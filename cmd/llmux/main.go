// Command llmux is an OpenAI-compatible proxy that routes chat requests to the
// configured model backends.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/gjcourt/llmux/internal/adapters/anthropic"
	"github.com/gjcourt/llmux/internal/adapters/httpapi"
	"github.com/gjcourt/llmux/internal/adapters/openaicompat"
	prommetrics "github.com/gjcourt/llmux/internal/adapters/prometheus"
	"github.com/gjcourt/llmux/internal/app"
	"github.com/gjcourt/llmux/internal/ports/outbound"
)

// envOr returns the value of key, or fallback when key is not set at all. A key
// that is set to the empty string returns "", which is how a backend is
// disabled (LLMUX_VLLM_URL=). The original code used a getter that treated ""
// as unset and fell back to the default, so an empty value could never
// actually disable a backend despite AGENTS.md promising it did.
func envOr(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return fallback
}

func main() {
	if err := run(); err != nil {
		slog.Error("server error", "err", err)
		os.Exit(1)
	}
}

func run() error {
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})))

	addr := envOr("LLMUX_ADDR", ":8080")

	providers, err := providersFromEnv()
	if err != nil {
		return err
	}
	if len(providers) == 0 {
		slog.Warn("no model backends configured; every chat request will return 404")
	}

	svc := app.New(providers...)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Metrics get their own listener so the chat port can stay reachable
	// from Open WebUI alone, and the scrape port from Prometheus alone.
	// The metrics server has its own context, cancelled only after the chat
	// server has drained, so counts from requests finishing during the
	// drain stay scrapeable for as long as possible.
	mctx, stopMetrics := context.WithCancel(context.Background())
	defer stopMetrics()
	metricsDone := make(chan error, 1)
	if maddr := envOr("LLMUX_METRICS_ADDR", ":9090"); maddr != "" {
		metrics := prommetrics.New()
		// Anthropic's model list is static config; pre-create its series.
		// vLLM/Ollama serve whatever id they're sent, so theirs can't be.
		for _, p := range providers {
			if p.Name() == "anthropic" {
				ms, _ := p.Models(ctx)
				ids := make([]string, 0, len(ms))
				for _, m := range ms {
					ids = append(ids, m.ID)
				}
				metrics.Declare(p.Name(), ids)
			}
		}
		svc.WithMetrics(metrics)
		mux := http.NewServeMux()
		mux.Handle("GET /metrics", metrics.Handler())
		mln, err := net.Listen("tcp", maddr)
		if err != nil {
			return err
		}
		slog.Info("metrics listening", "addr", mln.Addr().String())
		msrv := &http.Server{Handler: mux, ReadHeaderTimeout: 10 * time.Second}
		go func() {
			err := serve(mctx, msrv, mln, 5*time.Second)
			if err != nil {
				// Chat keeps serving; say so now rather than only at exit.
				slog.Error("metrics server failed", "err", err)
			}
			metricsDone <- err
		}()
	} else {
		metricsDone <- nil
	}

	srv := &http.Server{
		Handler:           httpapi.New(svc),
		ReadHeaderTimeout: 10 * time.Second,
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	slog.Info("llmux listening", "addr", ln.Addr().String(), "providers", len(providers))
	err = serve(ctx, srv, ln, 25*time.Second)
	stopMetrics() // chat has drained (or failed on its own): metrics go last
	// A metrics server that failed at runtime makes even a clean shutdown
	// exit non-zero: the failure was logged when it happened, and the exit
	// status is where a supervisor looks.
	return errors.Join(err, <-metricsDone)
}

// providersFromEnv builds the providers from the environment, in routing
// order.
func providersFromEnv() ([]outbound.ChatProvider, error) {
	// Order matters: the first provider that Handles a model wins, so the
	// specific Anthropic list goes before the vLLM/Ollama catch-all.
	var providers []outbound.ChatProvider

	if key := os.Getenv("LLMUX_ANTHROPIC_API_KEY"); key != "" {
		maxTokens, err := strconv.Atoi(envOr("LLMUX_ANTHROPIC_MAX_TOKENS", "8192"))
		if err != nil || maxTokens <= 0 {
			return nil, errors.New("LLMUX_ANTHROPIC_MAX_TOKENS must be a positive integer")
		}
		searches, err := strconv.Atoi(envOr("LLMUX_WEB_SEARCH_MAX_USES", "3"))
		if err != nil || searches < 0 {
			return nil, errors.New("LLMUX_WEB_SEARCH_MAX_USES must be a non-negative integer (0 disables web search)")
		}
		models := splitList(envOr("LLMUX_ANTHROPIC_MODELS", "claude-sonnet-5,claude-opus-5,claude-haiku-4-5"))
		providers = append(providers, anthropic.New(anthropic.Config{
			APIKey:           key,
			BaseURL:          envOr("LLMUX_ANTHROPIC_URL", "https://api.anthropic.com"),
			Models:           models,
			DefaultMaxTokens: maxTokens,
			WebSearchMaxUses: searches,
			Client:           anthropic.HTTPClient(),
		}))
		slog.Info("anthropic provider enabled", "models", models, "web_search_max_uses", searches)
	}

	// Both backends left with the homelab GPUs, so they now default to off.
	compat := openaicompat.New(openaicompat.Config{
		VLLMURL:   envOr("LLMUX_VLLM_URL", ""),
		OllamaURL: envOr("LLMUX_OLLAMA_URL", ""),
		Client:    &http.Client{Timeout: 120 * time.Second},
	})
	if compat.Enabled() {
		providers = append(providers, compat)
	}
	return providers, nil
}

// serve runs srv on ln until ctx is done, then drains in-flight requests for
// up to grace. Serve returns ErrServerClosed the moment Shutdown *starts*, so
// serve waits for Shutdown to *finish* — otherwise the process would exit
// mid-stream. The 25s grace sits under Kubernetes' default 30s.
func serve(ctx context.Context, srv *http.Server, ln net.Listener, grace time.Duration) error {
	drained := make(chan struct{})
	go func() {
		defer close(drained)
		<-ctx.Done()
		slog.Info("shutting down, draining in-flight requests")
		shutdown, cancel := context.WithTimeout(context.Background(), grace)
		defer cancel()
		if err := srv.Shutdown(shutdown); err != nil {
			slog.Warn("shutdown did not drain cleanly", "err", err)
		}
	}()

	if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	<-drained
	return nil
}

// splitList parses a comma-separated list, dropping blanks.
func splitList(s string) []string {
	var out []string
	for part := range strings.SplitSeq(s, ",") {
		if p := strings.TrimSpace(part); p != "" {
			out = append(out, p)
		}
	}
	return out
}
