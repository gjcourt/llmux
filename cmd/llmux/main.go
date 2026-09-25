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
	"syscall"
	"time"

	"github.com/gjcourt/llmux/internal/adapters/httpapi"
	"github.com/gjcourt/llmux/internal/adapters/openaicompat"
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
	compat := openaicompat.New(openaicompat.Config{
		VLLMURL:   envOr("LLMUX_VLLM_URL", "http://10.42.2.10:8000"),
		OllamaURL: envOr("LLMUX_OLLAMA_URL", "http://10.42.2.10:30068/v1"),
		Client:    &http.Client{Timeout: 120 * time.Second},
	})

	var providers []outbound.ChatProvider
	if compat.Enabled() {
		providers = append(providers, compat)
	}
	if len(providers) == 0 {
		slog.Warn("no model backends configured; every chat request will return 404")
	}

	srv := &http.Server{
		Handler:           httpapi.New(app.New(providers...)),
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	slog.Info("llmux listening", "addr", ln.Addr().String(), "providers", len(providers))
	return serve(ctx, srv, ln, 25*time.Second)
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
