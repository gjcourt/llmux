package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gjcourt/llmux/internal/domain"
)

// recorder is an EventSink that keeps every event.
type recorder struct{ events []domain.Event }

func (r *recorder) Emit(e domain.Event) error { r.events = append(r.events, e); return nil }

func (r *recorder) text() string {
	var b strings.Builder
	for _, e := range r.events {
		if e.Kind == domain.EventText {
			b.WriteString(e.Text)
		}
	}
	return b.String()
}

func (r *recorder) toolNames() []string {
	var out []string
	for _, e := range r.events {
		if e.Kind == domain.EventToolCall && e.ToolCall.Name != "" {
			out = append(out, e.ToolCall.Name)
		}
	}
	return out
}

func (r *recorder) finish() string {
	for _, e := range r.events {
		if e.Kind == domain.EventFinish {
			return e.FinishReason
		}
	}
	return ""
}

// fakeOllama returns a server whose handler calls fn for each request, and a
// counter of calls. Ported from the original main_test.go.
func fakeOllama(fn func(callN int, w http.ResponseWriter)) (*httptest.Server, *atomic.Int32) {
	var n atomic.Int32
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		c := n.Add(1)
		w.Header().Set("Content-Type", "application/json")
		fn(int(c), w)
	})), &n
}

func writeOllamaResp(w http.ResponseWriter, content string) {
	json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
		"id": "chatcmpl-test", "object": "chat.completion", "model": "test",
		"choices": []map[string]any{{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": content},
			"finish_reason": "stop",
		}},
	})
}

// Ported from TestProxyTransform_EmptyRetryExtractsToolCall: the model returns
// empty first (confused by tools), then on the plain-chat retry emits
// <tool_call> XML in text. The retry must go through the transform, so the XML
// becomes a tool call event — never raw text.
func TestTransform_EmptyRetryExtractsToolCall(t *testing.T) {
	upstream, _ := fakeOllama(func(callN int, w http.ResponseWriter) {
		if callN == 1 {
			writeOllamaResp(w, "")
		} else {
			writeOllamaResp(w, `Bing blocked. Let me try another way.<tool_call>{"name":"web_search","arguments":{"query":"test"}}</tool_call>`)
		}
	})
	defer upstream.Close()

	orig := []byte(`{"model":"test","messages":[{"role":"user","content":"search"}],"tools":[{"type":"function","function":{"name":"web_search","parameters":{"type":"object"}}}],"stream":true}`)
	p := New(Config{})
	rec := &recorder{}
	if err := p.transform(context.Background(), upstream.URL, forceNoStream(orig), orig, false, rec); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(rec.text(), "<tool_call>") {
		t.Error("raw <tool_call> XML leaked into text; expected a tool call event")
	}
	if names := rec.toolNames(); len(names) != 1 || names[0] != "web_search" {
		t.Errorf("tool calls: want [web_search], got %v", names)
	}
	if rec.finish() != "tool_calls" {
		t.Errorf("finish: want tool_calls, got %q", rec.finish())
	}
}

// Ported from TestProxyTransform_EmptyRetryDoesNotLoop.
func TestTransform_EmptyRetryDoesNotLoop(t *testing.T) {
	upstream, calls := fakeOllama(func(_ int, w http.ResponseWriter) { writeOllamaResp(w, "") })
	defer upstream.Close()

	orig := []byte(`{"model":"test","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"stream":true}`)
	if err := New(Config{}).transform(context.Background(), upstream.URL, forceNoStream(orig), orig, false, &recorder{}); err != nil {
		t.Fatal(err)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("want exactly 2 upstream calls (original + one retry), got %d", got)
	}
}

// Ported from TestProxyTransform_NoRetryAfterToolResults.
func TestTransform_NoRetryAfterToolResults(t *testing.T) {
	upstream, calls := fakeOllama(func(_ int, w http.ResponseWriter) { writeOllamaResp(w, "") })
	defer upstream.Close()

	orig := []byte(`{"model":"test","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c1","content":"result"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"stream":true}`)
	if err := New(Config{}).transform(context.Background(), upstream.URL, forceNoStream(orig), orig, false, &recorder{}); err != nil {
		t.Fatal(err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("want exactly 1 upstream call (no retry after tool results), got %d", got)
	}
}

// emitMessage must reproduce the order the original writeAsSSE used.
func TestEmitMessage_Order(t *testing.T) {
	body, err := applyToolCallTransform(ollamaResp(`<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{"x":1}}</tool_call>`))
	if err != nil {
		t.Fatal(err)
	}
	var resp openAIResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		t.Fatal(err)
	}
	rec := &recorder{}
	if err := emitMessage(resp, rec); err != nil {
		t.Fatal(err)
	}
	var kinds []domain.EventKind
	for _, e := range rec.events {
		kinds = append(kinds, e.Kind)
	}
	want := []domain.EventKind{domain.EventStart, domain.EventToolCall, domain.EventToolCall, domain.EventToolCall, domain.EventToolCall, domain.EventFinish}
	if len(kinds) != len(want) {
		t.Fatalf("kinds: want %v, got %v", want, kinds)
	}
	for i := range want {
		if kinds[i] != want[i] {
			t.Fatalf("kinds: want %v, got %v", want, kinds)
		}
	}
	if rec.events[1].ToolCall.Name != "a" || rec.events[2].ToolCall.ArgumentsDelta != "{}" || rec.events[3].ToolCall.Index != 1 {
		t.Errorf("unexpected tool call deltas: %+v", rec.events[1:5])
	}
}

func TestEmitMessage_NoChoicesEmitsNothing(t *testing.T) {
	rec := &recorder{}
	if err := emitMessage(openAIResponse{ID: "x"}, rec); err != nil {
		t.Fatal(err)
	}
	if len(rec.events) != 0 {
		t.Errorf("want no events, got %+v", rec.events)
	}
}

func TestRelaySSE(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"id":"c1","model":"m","created":7,"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`data: {"id":"c1","model":"m","choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}`,
		`data: {"id":"c1","model":"m","choices":[{"delta":{"content":"lo"},"finish_reason":null}]}`,
		`data: {"id":"c1","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","type":"function","function":{"name":"f","arguments":""}}]},"finish_reason":null}]}`,
		`data: {"id":"c1","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":1}"}}]},"finish_reason":null}]}`,
		`data: {"id":"c1","model":"m","choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: {"id":"c1","model":"m","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`,
		`data: [DONE]`,
		`data: {"this":"is after DONE and must be ignored"}`,
	}, "\n\n")
	rec := &recorder{}
	if err := relaySSE(strings.NewReader(stream), rec); err != nil {
		t.Fatal(err)
	}
	if rec.events[0].Kind != domain.EventStart || rec.events[0].ID != "c1" || rec.events[0].Created != 7 {
		t.Errorf("first event: %+v", rec.events[0])
	}
	if rec.text() != "Hello" {
		t.Errorf("text: want Hello, got %q", rec.text())
	}
	var args string
	for _, e := range rec.events {
		if e.Kind == domain.EventToolCall {
			args += e.ToolCall.ArgumentsDelta
		}
	}
	if args != `{"a":1}` {
		t.Errorf("tool args: got %q", args)
	}
	if rec.finish() != "tool_calls" {
		t.Errorf("finish: got %q", rec.finish())
	}
	last := rec.events[len(rec.events)-1]
	if last.Kind != domain.EventUsage || last.Usage.TotalTokens != 7 {
		t.Errorf("last event should be usage total=7, got %+v", last)
	}
}

func TestChat_ToolsFallBackToOllamaTransformWhenVLLMDisabled(t *testing.T) {
	upstream, _ := fakeOllama(func(_ int, w http.ResponseWriter) {
		writeOllamaResp(w, `<tool_call>{"name":"f","arguments":{}}</tool_call>`)
	})
	defer upstream.Close()

	p := New(Config{VLLMURL: "", OllamaURL: upstream.URL})
	raw := []byte(`{"model":"m","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f"}}]}`)
	rec := &recorder{}
	if err := p.Chat(context.Background(), domain.ChatRequest{Raw: raw}, rec); err != nil {
		t.Fatal(err)
	}
	if names := rec.toolNames(); len(names) != 1 || names[0] != "f" {
		t.Errorf("want the transformed tool call f, got %v", names)
	}
}

func TestChat_PlainChatFallsBackToVLLM(t *testing.T) {
	vllm, _ := fakeOllama(func(_ int, w http.ResponseWriter) { writeOllamaResp(w, "from vllm") })
	defer vllm.Close()
	dead := httptest.NewServer(http.NotFoundHandler())
	dead.Close() // closed server: connection refused

	p := New(Config{VLLMURL: vllm.URL, OllamaURL: dead.URL})
	rec := &recorder{}
	if err := p.Chat(context.Background(), domain.ChatRequest{Raw: []byte(`{"model":"m","messages":[]}`)}, rec); err != nil {
		t.Fatal(err)
	}
	if rec.text() != "from vllm" {
		t.Errorf("want fallback to vLLM, got %q", rec.text())
	}
}

func TestChat_UpstreamErrorIsRelayedNotFallenBackFrom(t *testing.T) {
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":"bad model"}`)) //nolint:errcheck
	}))
	defer bad.Close()

	p := New(Config{OllamaURL: bad.URL})
	rec := &recorder{}
	err := p.Chat(context.Background(), domain.ChatRequest{Raw: []byte(`{"model":"m"}`)}, rec)
	var ue *domain.UpstreamError
	if !errors.As(err, &ue) || ue.Status != http.StatusBadRequest || string(ue.Body) != `{"error":"bad model"}` {
		t.Fatalf("want UpstreamError 400 with body, got %v", err)
	}
	if len(rec.events) != 0 {
		t.Errorf("no events may precede an UpstreamError, got %+v", rec.events)
	}
}

func TestChat_AllDisabledIsUnavailable(t *testing.T) {
	err := New(Config{}).Chat(context.Background(), domain.ChatRequest{Raw: []byte(`{"model":"m"}`)}, &recorder{})
	if !errors.Is(err, domain.ErrUnavailable) {
		t.Fatalf("want ErrUnavailable, got %v", err)
	}
}

func TestHandlesOnlyWhenEnabled(t *testing.T) {
	if New(Config{}).Handles("anything") {
		t.Error("a provider with no upstreams must not claim models")
	}
	if !New(Config{OllamaURL: "http://x"}).Handles("anything") {
		t.Error("an enabled provider is the catch-all")
	}
}

// Critique pass 1, finding 3. The original proxyTransformInner ran every JSON
// body through the transform whatever its status, so an Ollama error on the
// tool path (e.g. 400 "model does not support tools") parsed as an empty
// response and triggered the plain-chat retry. Returning the error first
// broke that recovery.
func TestTransform_ErrorOnToolPathRetriesAsPlainChat(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		b, _ := io.ReadAll(r.Body)
		if strings.Contains(string(b), `"tools"`) {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error":{"message":"model does not support tools"}}`)) //nolint:errcheck
			return
		}
		writeOllamaResp(w, "hello")
	}))
	defer up.Close()

	orig := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}]}`)
	rec := &recorder{}
	if err := New(Config{}).transform(context.Background(), up.URL, forceNoStream(orig), orig, false, rec); err != nil {
		t.Fatalf("want recovery via plain-chat retry, got %v", err)
	}
	if rec.text() != "hello" {
		t.Errorf("want the retried answer, got %q", rec.text())
	}
}

// An error that survives the retry is still reported, not swallowed.
func TestTransform_ErrorAfterRetryIsRelayed(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte(`{"error":{"message":"loading model"}}`)) //nolint:errcheck
	}))
	defer up.Close()

	orig := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}]}`)
	err := New(Config{}).transform(context.Background(), up.URL, forceNoStream(orig), orig, false, &recorder{})
	var ue *domain.UpstreamError
	if !errors.As(err, &ue) || ue.Status != http.StatusServiceUnavailable {
		t.Fatalf("want the upstream 503 relayed after the retry, got %v", err)
	}
}

// Critique pass 1, finding 5: an upstream error chunk must not become a silent
// empty answer.
func TestRelaySSE_ErrorChunk(t *testing.T) {
	first := "data: {\"error\":{\"message\":\"boom\"}}\n\n"
	err := relaySSE(strings.NewReader(first), &recorder{})
	var ue *domain.UpstreamError
	if !errors.As(err, &ue) || !strings.Contains(string(ue.Body), "boom") {
		t.Fatalf("error as first chunk: want UpstreamError carrying the body, got %v", err)
	}

	mid := strings.Join([]string{
		`data: {"id":"c","model":"m","choices":[{"delta":{"content":"par"},"finish_reason":null}]}`,
		`data: {"error":{"message":"boom"}}`,
		`data: [DONE]`,
	}, "\n\n")
	rec := &recorder{}
	err = relaySSE(strings.NewReader(mid), rec)
	if err == nil || !strings.Contains(err.Error(), "boom") || errors.As(err, &ue) {
		t.Fatalf("error mid-stream: want a plain error mentioning boom, got %v", err)
	}
	if rec.text() != "par" {
		t.Errorf("text before the error should still be relayed, got %q", rec.text())
	}
}

// Critique pass 2: a truncated tool call must not be reported as a finished
// one. A plain stop still becomes tool_calls.
func TestEmitMessage_ToolCallFinishReason(t *testing.T) {
	for upstream, want := range map[string]string{"length": "length", "stop": "tool_calls", "": "tool_calls", "tool_calls": "tool_calls"} {
		content := ""
		resp := openAIResponse{Choices: []openAIChoice{{
			Message:      openAIMsg{Role: "assistant", Content: &content, ToolCalls: []openAIToolCall{{ID: "c", Type: "function", Function: openAIToolFunction{Name: "f", Arguments: `{"q":`}}}},
			FinishReason: upstream,
		}}}
		rec := &recorder{}
		if err := emitMessage(resp, rec); err != nil {
			t.Fatal(err)
		}
		if got := rec.finish(); got != want {
			t.Errorf("upstream %q: finish = %q, want %q", upstream, got, want)
		}
	}
}

// Critique pass 2: a stream that stops without [DONE] was cut off, and must
// surface as an error rather than a clean finish.
func TestRelaySSE_TruncatedIsError(t *testing.T) {
	stream := `data: {"id":"c","model":"m","choices":[{"delta":{"content":"hal"}}]}` + "\n\n"
	rec := &recorder{}
	err := relaySSE(strings.NewReader(stream), rec)
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("want ErrUnexpectedEOF, got %v", err)
	}
	if rec.text() != "hal" {
		t.Errorf("text before the cut must still be relayed, got %q", rec.text())
	}
}
