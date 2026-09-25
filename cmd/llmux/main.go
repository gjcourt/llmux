// Command llmux is an OpenAI-compatible proxy that routes chat requests to the
// configured model backends.
package main

import (
	"context"
	"errors"
	"fmt"
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

	clientKeys, err := clientKeysFromEnv()
	if err != nil {
		return err
	}
	clients := []string{httpapi.Anonymous}
	if len(clientKeys) > 0 {
		clients = clients[:0]
		for name := range clientKeys {
			clients = append(clients, name)
		}
		slog.Info("client keys on", "clients", clients)
	} else {
		slog.Warn("no client keys configured: llmux accepts unauthenticated requests")
	}

	svc := app.New(providers...)
	var authHook func(reason string)

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
		metrics := prommetrics.New(clients)
		// Anthropic's model list is static config; pre-create its series.
		// vLLM/Ollama serve whatever id they're sent, so theirs can't be.
		for _, p := range providers {
			if p.Name() == "anthropic" {
				ms, err := p.Models(ctx)
				if err != nil {
					slog.Warn("could not list models to pre-create their metric series", "provider", p.Name(), "err", err)
				}
				ids := make([]string, 0, len(ms))
				for _, m := range ms {
					ids = append(ids, m.ID)
				}
				metrics.Declare(p.Name(), ids)
			}
		}
		svc.WithMetrics(metrics)
		authHook = metrics.AuthFailed
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
		Handler:           httpapi.New(svc, httpapi.WithClientKeys(clientKeys), httpapi.WithAuthFailureHook(authHook)),
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
		streamOnly, err := strconv.ParseBool(envOr("LLMUX_WEB_SEARCH_STREAM_ONLY", "true"))
		if err != nil {
			return nil, errors.New("LLMUX_WEB_SEARCH_STREAM_ONLY must be a boolean (true/false)")
		}
		models := splitList(envOr("LLMUX_ANTHROPIC_MODELS", "claude-sonnet-5,claude-opus-5,claude-haiku-4-5"))
		providers = append(providers, anthropic.New(anthropic.Config{
			APIKey:           key,
			BaseURL:          envOr("LLMUX_ANTHROPIC_URL", "https://api.anthropic.com"),
			Models:           models,
			DefaultMaxTokens: maxTokens,
			WebSearchMaxUses: searches,
			// Background calls (titles, tags) are non-streamed; don't pay
			// the search tool's per-request tokens on them.
			WebSearchStreamOnly: streamOnly,
			Client:              anthropic.HTTPClient(),
		}))
		slog.Info("anthropic provider enabled", "models", models, "web_search_max_uses", searches, "web_search_stream_only", streamOnly)
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

// clientKeysFromEnv reads LLMUX_CLIENT_KEYS, "name=key,name=key" (from a
// secret). Names label telemetry, so they're restricted to [a-z0-9-]; keys
// must be long enough not to be guessable. With LLMUX_REQUIRE_CLIENT_KEYS
// true, an empty list is an error rather than an unauthenticated llmux.
func clientKeysFromEnv() (map[string]string, error) {
	keys := map[string]string{}
	seen := map[string]bool{}
	for _, entry := range splitList(os.Getenv("LLMUX_CLIENT_KEYS")) {
		name, key, ok := strings.Cut(entry, "=")
		name, key = strings.TrimSpace(name), strings.TrimSpace(key)
		switch {
		case !ok || name == "" || key == "":
			return nil, errors.New("LLMUX_CLIENT_KEYS: each entry must be name=key")
		case !validClientName(name):
			return nil, fmt.Errorf("LLMUX_CLIENT_KEYS: client name %q must be lowercase letters, digits and hyphens", name)
		case name == httpapi.Anonymous:
			return nil, fmt.Errorf("LLMUX_CLIENT_KEYS: %q is reserved", name)
		case len(key) < 32:
			return nil, fmt.Errorf("LLMUX_CLIENT_KEYS: key for %q is shorter than 32 characters", name)
		case !validClientKey(key):
			return nil, fmt.Errorf("LLMUX_CLIENT_KEYS: key for %q must be letters, digits, '-' or '_' (generate with: openssl rand -hex 32)", name)
		case keys[name] != "":
			return nil, fmt.Errorf("LLMUX_CLIENT_KEYS: client %q listed twice", name)
		case seen[key]:
			return nil, fmt.Errorf("LLMUX_CLIENT_KEYS: two clients share a key (%q is the second)", name)
		}
		keys[name], seen[key] = key, true
	}
	required, err := strconv.ParseBool(envOr("LLMUX_REQUIRE_CLIENT_KEYS", "false"))
	if err != nil {
		return nil, errors.New("LLMUX_REQUIRE_CLIENT_KEYS must be a boolean (true/false)")
	}
	if required && len(keys) == 0 {
		return nil, errors.New("LLMUX_REQUIRE_CLIENT_KEYS is true but LLMUX_CLIENT_KEYS is empty")
	}
	return keys, nil
}

// validClientKey restricts keys to a URL- and list-safe alphabet: a key
// containing "," or "=" would otherwise be silently split into another entry.
func validClientKey(s string) bool {
	for _, r := range s {
		if (r < 'a' || r > 'z') && (r < 'A' || r > 'Z') && (r < '0' || r > '9') && r != '-' && r != '_' {
			return false
		}
	}
	return true
}

func validClientName(s string) bool {
	for _, r := range s {
		if (r < 'a' || r > 'z') && (r < '0' || r > '9') && r != '-' {
			return false
		}
	}
	return true
}
