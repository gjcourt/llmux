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
