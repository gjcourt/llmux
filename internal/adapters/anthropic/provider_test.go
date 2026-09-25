package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/gjcourt/llmux/internal/domain"
)

func newTestProvider(t *testing.T, h http.HandlerFunc) *Provider {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return New(Config{
		APIKey: "test-key", BaseURL: srv.URL + "/", Models: []string{"claude-sonnet-5"},
		Now: func() time.Time { return time.Unix(42, 0) },
	})
}

func TestProvider_Chat(t *testing.T) {
	fixture, err := os.ReadFile("testdata/plain.sse")
	if err != nil {
		t.Fatal(err)
	}
	var gotBody map[string]any
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/messages" {
			t.Errorf("request %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("x-api-key") != "test-key" || r.Header.Get("anthropic-version") != APIVersion {
			t.Errorf("headers: %v", r.Header)
		}
		if r.Header.Get("Authorization") != "" {
			t.Error("Authorization must not be sent")
		}
		b, _ := io.ReadAll(r.Body)
		json.Unmarshal(b, &gotBody) //nolint:errcheck
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write(fixture) //nolint:errcheck
	})
	rec := &recorder{}
	err = p.Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", Messages: []domain.Message{user("colors?")}}, rec)
	if err != nil {
		t.Fatal(err)
	}
	if rec.text() != "Red and blue." || rec.events[0].Created != 42 {
		t.Errorf("events: %+v", rec.events)
	}
	if gotBody["model"] != "claude-sonnet-5" || gotBody["stream"] != true || gotBody["max_tokens"] != float64(8192) {
		t.Errorf("upstream body: %v", gotBody)
	}
}

func TestProvider_UpstreamErrorRelayed(t *testing.T) {
	body := `{"type":"error","error":{"type":"invalid_request_error","message":"bad"},"request_id":"req_1"}`
	p := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(body)) //nolint:errcheck
	})
	rec := &recorder{}
	err := p.Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", Messages: []domain.Message{user("x")}}, rec)
	var ue *domain.UpstreamError
	if !errors.As(err, &ue) || ue.Status != 400 || string(ue.Body) != body || ue.ContentType != "application/json" {
		t.Fatalf("got %v", err)
	}
	if len(rec.events) != 0 {
		t.Error("no events may be emitted before an upstream error")
	}
}

func TestProvider_InvalidRequestNeverCallsUpstream(t *testing.T) {
	p := newTestProvider(t, func(http.ResponseWriter, *http.Request) {
		t.Error("upstream must not be called")
	})
	err := p.Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", Tools: []json.RawMessage{json.RawMessage(`{}`)}, Messages: []domain.Message{user("x")}}, &recorder{})
	var ire *domain.InvalidRequestError
	if !errors.As(err, &ire) {
		t.Fatalf("got %v", err)
	}
}

func TestProvider_Unreachable(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()
	p := New(Config{BaseURL: url, Models: []string{"m"}})
	err := p.Chat(context.Background(), domain.ChatRequest{Model: "m", Messages: []domain.Message{user("x")}}, &recorder{})
	if !errors.Is(err, domain.ErrUnavailable) {
		t.Fatalf("got %v", err)
	}
}

func TestProvider_CanceledIsNotUnavailable(t *testing.T) {
	p := newTestProvider(t, func(http.ResponseWriter, *http.Request) {})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := p.Chat(ctx, domain.ChatRequest{Model: "claude-sonnet-5", Messages: []domain.Message{user("x")}}, &recorder{})
	if !errors.Is(err, context.Canceled) || errors.Is(err, domain.ErrUnavailable) {
		t.Fatalf("got %v", err)
	}
}

func TestProvider_HandlesAndModels(t *testing.T) {
	p := New(Config{Models: []string{"claude-sonnet-5", "claude-haiku-4-5"}})
	if !p.Handles("claude-haiku-4-5") || p.Handles("qwen3") {
		t.Error("Handles must match the configured list exactly")
	}
	ms, err := p.Models(context.Background())
	if err != nil || len(ms) != 2 || ms[0].ID != "claude-sonnet-5" || ms[0].OwnedBy != "anthropic" {
		t.Errorf("models: %+v %v", ms, err)
	}
}

// Critique #29 pass 1: Go strips Authorization on a cross-host redirect but
// not x-api-key. The provider must not follow redirects at all.
func TestProvider_DoesNotFollowRedirects(t *testing.T) {
	leaked := false
	other := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { leaked = true }))
	defer other.Close()
	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, other.URL+"/v1/messages", http.StatusTemporaryRedirect)
	}))
	defer redirector.Close()

	p := New(Config{APIKey: "sk-secret", BaseURL: redirector.URL, Models: []string{"m"}})
	err := p.Chat(context.Background(), domain.ChatRequest{Model: "m", Messages: []domain.Message{user("x")}}, &recorder{})
	var ue *domain.UpstreamError
	if !errors.As(err, &ue) || ue.Status != http.StatusTemporaryRedirect {
		t.Fatalf("want the 307 relayed as an upstream error, got %v", err)
	}
	if leaked {
		t.Fatal("redirect was followed: the API key reached another host")
	}
}

// A stream that goes silent is cut off with a distinct error — not
// context.Canceled, which the handler would take for a departed client.
func TestProvider_IdleTimeout(t *testing.T) {
	release := make(chan struct{})
	defer close(release)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(sse(evStart))) //nolint:errcheck
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	p := New(Config{BaseURL: srv.URL, Models: []string{"m"}, IdleTimeout: 150 * time.Millisecond})
	start := time.Now()
	err := p.Chat(context.Background(), domain.ChatRequest{Model: "m", Messages: []domain.Message{user("x")}}, &recorder{})
	if !errors.Is(err, errIdle) || errors.Is(err, context.Canceled) {
		t.Fatalf("want errIdle, got %v", err)
	}
	if time.Since(start) > 5*time.Second {
		t.Error("idle timeout did not fire promptly")
	}
}

// Data arriving keeps resetting the idle timer.
func TestProvider_IdleTimerResetsOnData(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range []string{evStart, evText, `{"type":"ping"}`, `{"type":"ping"}`, evDelta, `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`, evStop} {
			w.Write([]byte(sse(ev))) //nolint:errcheck
			w.(http.Flusher).Flush()
			time.Sleep(60 * time.Millisecond)
		}
	}))
	defer srv.Close()
	p := New(Config{BaseURL: srv.URL, Models: []string{"m"}, IdleTimeout: 150 * time.Millisecond})
	rec := &recorder{}
	if err := p.Chat(context.Background(), domain.ChatRequest{Model: "m", Messages: []domain.Message{user("x")}}, rec); err != nil {
		t.Fatalf("a slow but live stream must complete: %v", err)
	}
	if rec.text() != "hi" {
		t.Errorf("text %q", rec.text())
	}
}

// A client that goes away mid-stream surfaces as context.Canceled.
func TestProvider_CancelMidStream(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(sse(evStart, evText, evDelta))) //nolint:errcheck
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer srv.Close()
	p := New(Config{BaseURL: srv.URL, Models: []string{"m"}})
	sink := &cancelOnText{cancel: cancel}
	err := p.Chat(ctx, domain.ChatRequest{Model: "m", Messages: []domain.Message{user("x")}}, sink)
	if !errors.Is(err, context.Canceled) || errors.Is(err, errIdle) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
}

type cancelOnText struct{ cancel context.CancelFunc }

func (c *cancelOnText) Emit(e domain.Event) error {
	if e.Kind == domain.EventText {
		c.cancel()
	}
	return nil
}

func TestProvider_RetryAfterKept(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Retry-After", "17")
		w.WriteHeader(http.StatusTooManyRequests)
	})
	err := p.Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", Messages: []domain.Message{user("x")}}, &recorder{})
	var ue *domain.UpstreamError
	if !errors.As(err, &ue) || ue.RetryAfter != "17" {
		t.Fatalf("got %v", err)
	}
}
