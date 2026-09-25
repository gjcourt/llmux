package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
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

// scripted serves one canned SSE body per request, in order, and records
// each request body.
type scripted struct {
	bodies []string
	got    [][]byte
}

func (s *scripted) handler(t *testing.T) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		s.got = append(s.got, b)
		if len(s.got) > len(s.bodies) {
			t.Errorf("unexpected request %d", len(s.got))
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(s.bodies[len(s.got)-1])) //nolint:errcheck
	}
}

func pausedTurn(id string) string {
	return sse(
		`{"type":"message_start","message":{"id":"`+id+`","model":"claude-sonnet-5","usage":{"input_tokens":100,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srv_`+id+`","name":"web_search","input":{}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"q\"}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":5}}`,
		evStop)
}

func finalTurn(text string) string {
	return sse(
		`{"type":"message_start","message":{"id":"msg_final","model":"claude-sonnet-5","usage":{"input_tokens":300,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"`+text+`"}}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}`,
		evStop)
}

// A paused turn is resumed with the paused blocks appended as an assistant
// message, and the client sees one answer: one Start, one Finish, summed usage.
func TestProvider_ResumesPausedTurn(t *testing.T) {
	s := &scripted{bodies: []string{pausedTurn("m1"), finalTurn("done")}}
	srv := httptest.NewServer(s.handler(t))
	defer srv.Close()
	p := New(Config{BaseURL: srv.URL, Models: []string{"claude-sonnet-5"}, WebSearchMaxUses: 3})

	rec := &recorder{}
	if err := p.Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", Messages: []domain.Message{user("q?")}}, rec); err != nil {
		t.Fatal(err)
	}
	if len(s.got) != 2 {
		t.Fatalf("want 2 upstream calls, got %d", len(s.got))
	}
	var second struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
		Tools []map[string]any `json:"tools"`
	}
	if err := json.Unmarshal(s.got[1], &second); err != nil {
		t.Fatal(err)
	}
	if len(second.Messages) != 2 || second.Messages[1].Role != "assistant" ||
		string(second.Messages[1].Content) != `[{"id":"srv_m1","input":{"query":"q"},"name":"web_search","type":"server_tool_use"}]` {
		t.Errorf("resume request messages: %s", s.got[1])
	}
	if len(second.Tools) != 1 || second.Tools[0]["type"] != "web_search_20250305" || second.Tools[0]["max_uses"] != float64(3) {
		t.Errorf("resume must keep the tool: %v", second.Tools)
	}
	if n := len(rec.kinds(domain.EventStart)); n != 1 {
		t.Errorf("want one Start, got %d", n)
	}
	if rec.events[0].ID != "m1" || rec.text() != "done" {
		t.Errorf("events: %+v", rec.events)
	}
	fin, us := rec.kinds(domain.EventFinish), rec.kinds(domain.EventUsage)
	if len(fin) != 1 || fin[0].FinishReason != "stop" {
		t.Errorf("finish: %+v", fin)
	}
	if len(us) != 1 || us[0].Usage != (domain.Usage{PromptTokens: 400, CompletionTokens: 12, TotalTokens: 412}) {
		t.Errorf("usage must sum both calls: %+v", us)
	}
}

// Resumes are capped: after maxContinuations the answer finishes as length.
func TestProvider_ContinuationCap(t *testing.T) {
	bodies := make([]string, maxContinuations+1)
	for i := range bodies {
		bodies[i] = pausedTurn("m")
	}
	s := &scripted{bodies: bodies}
	srv := httptest.NewServer(s.handler(t))
	defer srv.Close()
	p := New(Config{BaseURL: srv.URL, Models: []string{"claude-sonnet-5"}, WebSearchMaxUses: 3})
	rec := &recorder{}
	if err := p.Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", Messages: []domain.Message{user("q?")}}, rec); err != nil {
		t.Fatal(err)
	}
	if len(s.got) != maxContinuations+1 {
		t.Errorf("want %d calls, got %d", maxContinuations+1, len(s.got))
	}
	if fin := rec.kinds(domain.EventFinish); len(fin) != 1 || fin[0].FinishReason != "length" {
		t.Errorf("finish: %+v", fin)
	}
	// Each resume appends to the same assistant message rather than adding
	// consecutive assistant turns.
	var last struct {
		Messages []struct {
			Role    string            `json:"role"`
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	json.Unmarshal(s.got[len(s.got)-1], &last) //nolint:errcheck
	if len(last.Messages) != 2 || len(last.Messages[1].Content) != maxContinuations {
		t.Errorf("want 1 user + 1 assistant with %d blocks, got %s", maxContinuations, s.got[len(s.got)-1])
	}
}

// A failed resume keeps the upstream error (status, Retry-After) wrapped, so
// the inbound adapter can still relay it to a client that has received
// nothing yet (JSON), and end a started stream with an error chunk.
func TestProvider_FailedResumeKeepsUpstreamError(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write([]byte(pausedTurn("m"))) //nolint:errcheck
			return
		}
		w.Header().Set("Retry-After", "7")
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`)) //nolint:errcheck
	}))
	defer srv.Close()
	p := New(Config{BaseURL: srv.URL, Models: []string{"claude-sonnet-5"}, WebSearchMaxUses: 3})
	err := p.Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", Messages: []domain.Message{user("q?")}}, &recorder{})
	var ue *domain.UpstreamError
	if !errors.As(err, &ue) || ue.Status != 429 || ue.RetryAfter != "7" || !strings.Contains(err.Error(), "resuming paused turn") {
		t.Fatalf("got %v", err)
	}
}

// The client's max_tokens caps the whole answer across resumes.
func TestProvider_ResumeSpendsMaxTokensBudget(t *testing.T) {
	s := &scripted{bodies: []string{pausedTurn("m1"), finalTurn("done")}} // pausedTurn uses 5 output tokens
	srv := httptest.NewServer(s.handler(t))
	defer srv.Close()
	p := New(Config{BaseURL: srv.URL, Models: []string{"claude-sonnet-5"}, WebSearchMaxUses: 3})
	if err := p.Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", MaxTokens: ptr(100), Messages: []domain.Message{user("q?")}}, &recorder{}); err != nil {
		t.Fatal(err)
	}
	var second struct {
		MaxTokens int `json:"max_tokens"`
	}
	json.Unmarshal(s.got[1], &second) //nolint:errcheck
	if second.MaxTokens != 95 {
		t.Errorf("resume max_tokens = %d, want 95", second.MaxTokens)
	}

	// Budget exhausted: no resume, answer ends as length.
	s2 := &scripted{bodies: []string{pausedTurn("m1")}}
	srv2 := httptest.NewServer(s2.handler(t))
	defer srv2.Close()
	p2 := New(Config{BaseURL: srv2.URL, Models: []string{"claude-sonnet-5"}, WebSearchMaxUses: 3})
	rec := &recorder{}
	if err := p2.Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", MaxTokens: ptr(5), Messages: []domain.Message{user("q?")}}, rec); err != nil {
		t.Fatal(err)
	}
	if len(s2.got) != 1 || rec.kinds(domain.EventFinish)[0].FinishReason != "length" {
		t.Errorf("calls %d, events %+v", len(s2.got), rec.events)
	}
}

// A turn whose blocks got a delta llmux can't fold in is not resumed: the
// replayed copy could be incomplete.
func TestProvider_LossyTurnNotResumed(t *testing.T) {
	lossy := sse(
		`{"type":"message_start","message":{"id":"m","model":"claude-sonnet-5","usage":{"input_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"future_delta","stuff":"x"}}`,
		`{"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":1}}`, evStop)
	s := &scripted{bodies: []string{lossy}}
	srv := httptest.NewServer(s.handler(t))
	defer srv.Close()
	p := New(Config{BaseURL: srv.URL, Models: []string{"claude-sonnet-5"}, WebSearchMaxUses: 3})
	rec := &recorder{}
	if err := p.Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", Messages: []domain.Message{user("q?")}}, rec); err != nil {
		t.Fatal(err)
	}
	if len(s.got) != 1 || rec.kinds(domain.EventFinish)[0].FinishReason != "length" {
		t.Errorf("calls %d, events %+v", len(s.got), rec.events)
	}
}

// A source cited in two turns of one answer is reported once.
func TestProvider_CitationsDedupedAcrossResumes(t *testing.T) {
	cite := `{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","url":"https://same.example/","title":"S"}}}`
	turn := func(stop string) string {
		return sse(`{"type":"message_start","message":{"id":"m","model":"claude-sonnet-5","usage":{"input_tokens":1}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"citations":[],"type":"text","text":""}}`, cite,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}`,
			`{"type":"message_delta","delta":{"stop_reason":"`+stop+`"},"usage":{"output_tokens":1}}`, evStop)
	}
	s := &scripted{bodies: []string{turn("pause_turn"), turn("end_turn")}}
	srv := httptest.NewServer(s.handler(t))
	defer srv.Close()
	p := New(Config{BaseURL: srv.URL, Models: []string{"claude-sonnet-5"}, WebSearchMaxUses: 3})
	rec := &recorder{}
	if err := p.Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", Messages: []domain.Message{user("q?")}}, rec); err != nil {
		t.Fatal(err)
	}
	if n := len(rec.kinds(domain.EventCitation)); n != 1 || len(s.got) != 2 {
		t.Errorf("citations %d, calls %d", n, len(s.got))
	}
}

// Prefill extension through the whole Chat loop.
func TestProvider_ResumeExtendsPrefill(t *testing.T) {
	s := &scripted{bodies: []string{pausedTurn("m1"), finalTurn("done")}}
	srv := httptest.NewServer(s.handler(t))
	defer srv.Close()
	p := New(Config{BaseURL: srv.URL, Models: []string{"claude-haiku-4-5"}, WebSearchMaxUses: 3})
	req := domain.ChatRequest{Model: "claude-haiku-4-5", Messages: []domain.Message{user("q?"), {Role: "assistant", Content: "Sure:"}}}
	if err := p.Chat(context.Background(), req, &recorder{}); err != nil {
		t.Fatal(err)
	}
	var second struct {
		Messages []struct {
			Role string `json:"role"`
		} `json:"messages"`
	}
	json.Unmarshal(s.got[1], &second) //nolint:errcheck
	if len(second.Messages) != 2 || second.Messages[1].Role != "assistant" {
		t.Errorf("want user + one assistant, got %s", s.got[1])
	}
}

func TestBuildRequest_WebSearchTool(t *testing.T) {
	off, _ := buildRequest(domain.ChatRequest{Model: "m", Messages: []domain.Message{user("x")}}, 10, 0)
	if off.Tools != nil {
		t.Error("0 must leave web search off")
	}
	on, _ := buildRequest(domain.ChatRequest{Model: "m", Messages: []domain.Message{user("x")}}, 10, 3)
	b, _ := json.Marshal(on.Tools)
	if string(b) != `[{"type":"web_search_20250305","name":"web_search","max_uses":3}]` {
		t.Errorf("tools: %s", b)
	}
}

// A client prefill (conversation ending in an assistant message) is extended,
// not followed by a second assistant message.
func TestWithContinuation_ExtendsPrefill(t *testing.T) {
	r, _ := buildRequest(domain.ChatRequest{Model: "m", Messages: []domain.Message{user("x"), {Role: "assistant", Content: "Sure:"}}}, 10, 3)
	r.withContinuation([]json.RawMessage{json.RawMessage(`{"type":"server_tool_use"}`)})
	b, _ := json.Marshal(r.Messages)
	if string(b) != `[{"role":"user","content":"x"},{"role":"assistant","content":[{"text":"Sure:","type":"text"},{"type":"server_tool_use"}]}]` {
		t.Errorf("messages: %s", b)
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
		select { // bounded, so a regression fails instead of deadlocking srv.Close
		case <-release:
		case <-r.Context().Done():
		case <-time.After(3 * time.Second):
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

// Critique #29 pass 2: the watchdog must also cover an error body — a proxy
// can send a 5xx status and then stall.
func TestProvider_IdleTimeoutCoversErrorBody(t *testing.T) {
	release := make(chan struct{})
	defer close(release)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"type":"error","err`)) //nolint:errcheck
		w.(http.Flusher).Flush()
		select { // bounded, so a regression fails instead of deadlocking srv.Close
		case <-release:
		case <-r.Context().Done():
		case <-time.After(10 * time.Second):
		}
	}))
	defer srv.Close()
	p := New(Config{BaseURL: srv.URL, Models: []string{"m"}, IdleTimeout: 150 * time.Millisecond})
	done := make(chan error, 1)
	go func() {
		done <- p.Chat(context.Background(), domain.ChatRequest{Model: "m", Messages: []domain.Message{user("x")}}, &recorder{})
	}()
	select {
	case err := <-done:
		if !errors.Is(err, errIdle) || !strings.Contains(err.Error(), "HTTP 500") {
			t.Fatalf("want the stall reported with the upstream status, got %v", err)
		}
		var ue *domain.UpstreamError
		if errors.As(err, &ue) {
			t.Error("a truncated body must not be relayed as the upstream's own error")
		}
	case <-time.After(2 * time.Second): // idle is 150ms; the handler stalls 10s
		t.Fatal("a stalled error body hung the request")
	}
}

type slowSink struct{ delay time.Duration }

func (s slowSink) Emit(domain.Event) error { time.Sleep(s.delay); return nil }

// Time spent writing to a slow client is not upstream silence.
func TestProvider_SlowClientIsNotIdle(t *testing.T) {
	// Events arrive promptly, one write each, so every one needs its own
	// Read — and each Read comes after a slow Emit.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range []string{evStart, evText, evDelta, evDelta, `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`, evStop} {
			w.Write([]byte(sse(ev))) //nolint:errcheck
			w.(http.Flusher).Flush()
			time.Sleep(20 * time.Millisecond)
		}
	}))
	defer srv.Close()
	p := New(Config{BaseURL: srv.URL, Models: []string{"m"}, IdleTimeout: 100 * time.Millisecond})
	if err := p.Chat(context.Background(), domain.ChatRequest{Model: "m", Messages: []domain.Message{user("x")}}, slowSink{delay: 250 * time.Millisecond}); err != nil {
		t.Fatalf("a slow client must not trip the upstream idle timeout: %v", err)
	}
}

// Critique #32 pass 1: tokens consumed before a failure are billed, so they
// must still be reported. Here the client cancels mid-answer, after
// message_start told us the input.
func TestProvider_CancelReportsInputConsumed(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(sse(`{"type":"message_start","message":{"id":"m","model":"m","usage":{"input_tokens":2800,"cache_read_input_tokens":200}}}`, evText, evDelta))) //nolint:errcheck
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer srv.Close()
	p := New(Config{BaseURL: srv.URL, Models: []string{"m"}})
	rec := &cancelRecorder{cancel: cancel}
	err := p.Chat(ctx, domain.ChatRequest{Model: "m", Messages: []domain.Message{user("x")}}, rec)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
	us := rec.kinds(domain.EventUsage)
	if len(us) != 1 || us[0].Usage.PromptTokens != 3000 || us[0].Usage.CacheReadTokens != 200 || !us[0].Partial {
		t.Errorf("usage after cancel must be reported, marked partial: %+v", us)
	}
	if len(rec.kinds(domain.EventFinish)) != 0 {
		t.Error("a cancelled answer must not report a finish")
	}
}

type cancelRecorder struct {
	recorder
	cancel context.CancelFunc
}

func (c *cancelRecorder) Emit(e domain.Event) error {
	if e.Kind == domain.EventText {
		c.cancel()
	}
	return c.recorder.Emit(e)
}

// A failure before message_start has consumed nothing and emits nothing —
// in particular it must not commit the client's response early.
func TestProvider_NoUsageBeforeStart(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusTooManyRequests) })
	rec := &recorder{}
	_ = p.Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", Messages: []domain.Message{user("x")}}, rec)
	if len(rec.events) != 0 {
		t.Errorf("events: %+v", rec.events)
	}
}

// A failed resume still reports the completed turn's usage.
func TestProvider_FailedResumeReportsCompletedTurn(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write([]byte(pausedTurn("m"))) //nolint:errcheck
			return
		}
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	p := New(Config{BaseURL: srv.URL, Models: []string{"claude-sonnet-5"}, WebSearchMaxUses: 3})
	rec := &recorder{}
	_ = p.Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", Messages: []domain.Message{user("q?")}}, rec)
	us := rec.kinds(domain.EventUsage)
	if len(us) != 1 || us[0].Usage.PromptTokens != 100 || us[0].Usage.CompletionTokens != 5 {
		t.Errorf("usage: %+v", us)
	}
}

// Cache and web-search counts sum across pause_turn resumes.
func TestProvider_ResumeSumsUsageBreakdown(t *testing.T) {
	turn := func(stop string) string {
		return sse(`{"type":"message_start","message":{"id":"m","model":"claude-sonnet-5","usage":{"input_tokens":1}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}`,
			`{"type":"message_delta","delta":{"stop_reason":"`+stop+`"},"usage":{"input_tokens":100,"cache_read_input_tokens":40,"cache_creation_input_tokens":10,"output_tokens":5,"server_tool_use":{"web_search_requests":2}}}`,
			evStop)
	}
	s := &scripted{bodies: []string{turn("pause_turn"), turn("end_turn")}}
	srv := httptest.NewServer(s.handler(t))
	defer srv.Close()
	p := New(Config{BaseURL: srv.URL, Models: []string{"claude-sonnet-5"}, WebSearchMaxUses: 3})
	rec := &recorder{}
	if err := p.Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", Messages: []domain.Message{user("q?")}}, rec); err != nil {
		t.Fatal(err)
	}
	want := domain.Usage{PromptTokens: 300, CompletionTokens: 10, TotalTokens: 310, CacheReadTokens: 80, CacheWriteTokens: 20, WebSearches: 4}
	if us := rec.kinds(domain.EventUsage); len(us) != 1 || us[0].Usage != want {
		t.Errorf("usage: %+v, want %+v", us, want)
	}
}

// A malformed stream (usage before message_start) must not emit anything
// that would commit the response before the error gets its status.
func TestProvider_NoPartialUsageBeforeStart(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(sse(`{"type":"message_delta","delta":{},"usage":{"input_tokens":50,"output_tokens":1}}`, //nolint:errcheck
			`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)))
	})
	rec := &recorder{}
	err := p.Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", Messages: []domain.Message{user("x")}}, rec)
	var ue *domain.UpstreamError
	if !errors.As(err, &ue) || len(rec.events) != 0 {
		t.Errorf("err %v, events %+v", err, rec.events)
	}
}

type failOnFinish struct{ recorder }

func (f *failOnFinish) Emit(e domain.Event) error {
	_ = f.recorder.Emit(e)
	if e.Kind == domain.EventFinish {
		return errors.New("client gone")
	}
	return nil
}

// A client that leaves between the last token and the finish chunk still
// leaves the answer's usage recorded.
func TestProvider_UsageEvenIfFinishWriteFails(t *testing.T) {
	fixture, _ := os.ReadFile("testdata/plain.sse")
	p := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write(fixture) //nolint:errcheck
	})
	rec := &failOnFinish{}
	if err := p.Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", Messages: []domain.Message{user("x")}}, rec); err == nil {
		t.Fatal("want the finish write error")
	}
	if us := rec.kinds(domain.EventUsage); len(us) != 1 || us[0].Usage.TotalTokens != 28 {
		t.Errorf("usage: %+v", us)
	}
}

// With WebSearchStreamOnly, a non-streamed request (Open WebUI's titles,
// tags, follow-ups) is not offered the search tool, so it doesn't pay the
// tool's per-request tokens; a streamed one is.
func TestProvider_WebSearchStreamOnly(t *testing.T) {
	var bodies [][]byte
	fixture, _ := os.ReadFile("testdata/plain.sse")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		bodies = append(bodies, b)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write(fixture) //nolint:errcheck
	}))
	defer srv.Close()
	tools := func(b []byte) int {
		var body struct {
			Tools []json.RawMessage `json:"tools"`
		}
		json.Unmarshal(b, &body) //nolint:errcheck
		return len(body.Tools)
	}
	for _, streamOnly := range []bool{true, false} {
		bodies = nil
		p := New(Config{BaseURL: srv.URL, Models: []string{"m"}, WebSearchMaxUses: 3, WebSearchStreamOnly: streamOnly})
		for _, stream := range []bool{true, false} {
			if err := p.Chat(context.Background(), domain.ChatRequest{Model: "m", Stream: stream, Messages: []domain.Message{user("x")}}, &recorder{}); err != nil {
				t.Fatal(err)
			}
		}
		streamed, background := tools(bodies[0]), tools(bodies[1])
		wantBackground := 1
		if streamOnly {
			wantBackground = 0
		}
		if streamed != 1 || background != wantBackground {
			t.Errorf("streamOnly=%v: streamed request got %d tools, non-streamed %d (want 1, %d)", streamOnly, streamed, background, wantBackground)
		}
	}
}
