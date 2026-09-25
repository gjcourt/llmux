package main

import (
	"context"
	"io"
	"net"
	"net/http"
	"testing"
	"time"
)

// A backend is disabled by setting its URL to the empty string. The original
// getter treated "" as unset and fell back to the default, so this never
// worked; AGENTS.md documented the intended behaviour.
func TestEnvOr_EmptyDisables(t *testing.T) {
	t.Setenv("LLMUX_TEST_URL", "")
	if got := envOr("LLMUX_TEST_URL", "http://default"); got != "" {
		t.Errorf("set-but-empty must return empty, got %q", got)
	}
}

func TestEnvOr_UnsetFallsBack(t *testing.T) {
	if got := envOr("LLMUX_TEST_URL_NEVER_SET", "http://default"); got != "http://default" {
		t.Errorf("unset must fall back, got %q", got)
	}
}

func TestEnvOr_SetWins(t *testing.T) {
	t.Setenv("LLMUX_TEST_URL", "http://custom")
	if got := envOr("LLMUX_TEST_URL", "http://default"); got != "http://custom" {
		t.Errorf("got %q", got)
	}
}

// Critique pass 1, finding 6: a SIGTERM must not cut off a response that is
// still streaming. serve must return only after the in-flight request ends.
func TestServe_DrainsInFlightRequest(t *testing.T) {
	started := make(chan struct{})
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("first ")) //nolint:errcheck
		w.(http.Flusher).Flush()
		close(started)
		time.Sleep(300 * time.Millisecond)
		w.Write([]byte("last")) //nolint:errcheck
	})
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- serve(ctx, &http.Server{Handler: mux}, ln, 5*time.Second) }()

	got := make(chan string, 1)
	go func() {
		resp, err := http.Get("http://" + ln.Addr().String())
		if err != nil {
			got <- "error: " + err.Error()
			return
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		got <- string(b)
	}()

	<-started
	cancel() // the signal arrives mid-response
	select {
	case err := <-served:
		t.Fatalf("serve returned (%v) while a request was still in flight", err)
	case <-time.After(100 * time.Millisecond):
	}
	if body := <-got; body != "first last" {
		t.Errorf("response was cut off: %q", body)
	}
	if err := <-served; err != nil {
		t.Errorf("serve: %v", err)
	}
}
