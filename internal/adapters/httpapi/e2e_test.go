package httpapi_test

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gjcourt/llmux/internal/adapters/httpapi"
	"github.com/gjcourt/llmux/internal/adapters/openaicompat"
	"github.com/gjcourt/llmux/internal/app"
	"github.com/gjcourt/llmux/internal/domain"
	"github.com/gjcourt/llmux/internal/testdoubles"
)

type chunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Model   string `json:"model"`
	Choices []struct {
		Delta struct {
			Role      string  `json:"role"`
			Content   *string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function *struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// sseEvents returns the data payloads of an SSE body.
func sseEvents(t *testing.T, body string) (chunks []chunk, done bool) {
	t.Helper()
	for line := range strings.SplitSeq(body, "\n") {
		line = strings.TrimRight(line, "\r")
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			done = true
			continue
		}
		if done {
			t.Fatalf("data after [DONE]: %s", data)
		}
		var c chunk
		if err := json.Unmarshal([]byte(data), &c); err != nil {
			t.Fatalf("invalid SSE JSON: %v\n%s", err, data)
		}
		chunks = append(chunks, c)
	}
	return chunks, done
}

// ollama serves one fixed chat.completion with the given assistant content.
func ollama(content string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
			"id": "chatcmpl-1", "object": "chat.completion", "model": "qwen",
			"choices": []map[string]any{{"index": 0, "message": map[string]any{"role": "assistant", "content": content}, "finish_reason": "stop"}},
		})
	}))
}

// serve runs the full stack — handler → app → openaicompat → fake Ollama —
// with vLLM disabled, so a request with tools takes the transform path.
func serve(t *testing.T, upstream *httptest.Server, body string) *http.Response {
	t.Helper()
	p := openaicompat.New(openaicompat.Config{OllamaURL: upstream.URL})
	srv := httptest.NewServer(httpapi.New(app.New(p)))
	t.Cleanup(srv.Close)
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

const toolsReq = `{"model":"qwen","stream":true,"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"web_search"}}]}`

// Ported from TestWriteAsSSE_ToolCall.
func TestE2E_StreamToolCall(t *testing.T) {
	up := ollama(`<tool_call>{"name":"web_search","arguments":{"query":"cats"}}</tool_call>`)
	defer up.Close()
	resp := serve(t, up, toolsReq)
	b, _ := io.ReadAll(resp.Body)

	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("content type: %q", ct)
	}
	chunks, done := sseEvents(t, string(b))
	if !done {
		t.Fatal("stream must end with [DONE]")
	}
	if chunks[0].Choices[0].Delta.Role != "assistant" {
		t.Errorf("first chunk must carry the assistant role, got %+v", chunks[0])
	}
	var name, args, finish string
	for _, c := range chunks {
		if c.Object != "chat.completion.chunk" || c.ID != "chatcmpl-1" {
			t.Errorf("chunk envelope wrong: %+v", c)
		}
		for _, tc := range c.Choices[0].Delta.ToolCalls {
			if tc.Function != nil {
				name += tc.Function.Name
				args += tc.Function.Arguments
			}
		}
		if fr := c.Choices[0].FinishReason; fr != nil {
			finish = *fr
		}
	}
	if name != "web_search" {
		t.Errorf("tool name: %q", name)
	}
	var a map[string]string
	if err := json.Unmarshal([]byte(args), &a); err != nil || a["query"] != "cats" {
		t.Errorf("tool args %q: %v", args, err)
	}
	if finish != "tool_calls" {
		t.Errorf("finish_reason: %q", finish)
	}
	if strings.Contains(string(b), "<tool_call>") {
		t.Error("raw <tool_call> XML leaked to the client")
	}
}

// Ported from TestWriteAsSSE_PlainText.
func TestE2E_StreamPlainText(t *testing.T) {
	up := ollama("Hello there!")
	defer up.Close()
	resp := serve(t, up, toolsReq)
	b, _ := io.ReadAll(resp.Body)
	chunks, done := sseEvents(t, string(b))
	if !done {
		t.Fatal("stream must end with [DONE]")
	}
	var text string
	for _, c := range chunks {
		if c.Choices[0].Delta.Content != nil {
			text += *c.Choices[0].Delta.Content
		}
	}
	if text != "Hello there!" {
		t.Errorf("content: %q", text)
	}
}

// Ported from TestWriteAsSSE_MultipleToolCalls.
func TestE2E_StreamMultipleToolCalls(t *testing.T) {
	up := ollama(`<tool_call>{"name":"a","arguments":{}}</tool_call>` + "\n" + `<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>`)
	defer up.Close()
	resp := serve(t, up, toolsReq)
	b, _ := io.ReadAll(resp.Body)
	chunks, _ := sseEvents(t, string(b))
	names := map[string]bool{}
	for _, c := range chunks {
		for _, tc := range c.Choices[0].Delta.ToolCalls {
			if tc.Function != nil && tc.Function.Name != "" {
				names[tc.Function.Name] = true
			}
		}
	}
	if !names["a"] || !names["b"] {
		t.Errorf("want tool calls a and b, got %v", names)
	}
}

func TestE2E_NonStreamToolCallJSON(t *testing.T) {
	up := ollama(`<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>`)
	defer up.Close()
	resp := serve(t, up, strings.Replace(toolsReq, `"stream":true`, `"stream":false`, 1))
	var out struct {
		Object  string `json:"object"`
		Choices []struct {
			Message struct {
				Content   *string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Function struct{ Name, Arguments string }
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	c := out.Choices[0]
	if out.Object != "chat.completion" || c.FinishReason != "tool_calls" || c.Message.Content != nil {
		t.Errorf("envelope: %+v", out)
	}
	if len(c.Message.ToolCalls) != 1 || c.Message.ToolCalls[0].Function.Name != "a" || c.Message.ToolCalls[0].Function.Arguments != `{"x":1}` || c.Message.ToolCalls[0].ID == "" {
		t.Errorf("tool calls: %+v", c.Message.ToolCalls)
	}
}

// fakeServer wires the handler to a scripted provider.
func fakeServer(t *testing.T, p *testdoubles.Provider) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(httpapi.New(app.New(p)))
	t.Cleanup(srv.Close)
	return srv
}

func post(t *testing.T, srv *httptest.Server, body string) (*http.Response, string) {
	t.Helper()
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp, string(b)
}

func TestE2E_UpstreamErrorRelayedVerbatim(t *testing.T) {
	srv := fakeServer(t, &testdoubles.Provider{Err: &domain.UpstreamError{Status: 429, ContentType: "application/json", Body: []byte(`{"error":"slow down"}`)}})
	for _, stream := range []string{"true", "false"} {
		resp, body := post(t, srv, `{"model":"m","stream":`+stream+`}`)
		if resp.StatusCode != 429 || body != `{"error":"slow down"}` {
			t.Errorf("stream=%s: want the upstream's 429 and body, got %d %q", stream, resp.StatusCode, body)
		}
	}
}

func TestE2E_NoProviderIs404(t *testing.T) {
	srv := fakeServer(t, &testdoubles.Provider{ModelIDs: []string{"only-this"}})
	resp, _ := post(t, srv, `{"model":"other","stream":true}`)
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("want 404, got %d", resp.StatusCode)
	}
}

func TestE2E_MidStreamErrorEndsStreamCleanly(t *testing.T) {
	srv := fakeServer(t, &testdoubles.Provider{
		Events: []domain.Event{{Kind: domain.EventStart, ID: "x"}, {Kind: domain.EventText, Text: "partial"}},
		Err:    errors.New("connection reset"),
	})
	resp, body := post(t, srv, `{"model":"m","stream":true}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("headers were already sent; want 200, got %d", resp.StatusCode)
	}
	chunks, done := sseEvents(t, body)
	if !done {
		t.Error("a failed stream must still be terminated with [DONE]")
	}
	last := chunks[len(chunks)-1]
	if last.Error == nil || !strings.Contains(last.Error.Message, "connection reset") {
		t.Errorf("want an error chunk before [DONE], got %+v", last)
	}
}

func TestE2E_UsageOnlyWhenRequested(t *testing.T) {
	events := []domain.Event{{Kind: domain.EventStart, ID: "x"}, {Kind: domain.EventText, Text: "hi"}, {Kind: domain.EventFinish, FinishReason: "stop"}, {Kind: domain.EventUsage, Usage: domain.Usage{TotalTokens: 9}}}
	srv := fakeServer(t, &testdoubles.Provider{Events: events})

	_, body := post(t, srv, `{"model":"m","stream":true}`)
	chunks, _ := sseEvents(t, body)
	for _, c := range chunks {
		if c.Usage != nil {
			t.Error("usage chunk sent without stream_options.include_usage")
		}
	}

	_, body = post(t, srv, `{"model":"m","stream":true,"stream_options":{"include_usage":true}}`)
	chunks, _ = sseEvents(t, body)
	if u := chunks[len(chunks)-1].Usage; u == nil || u.TotalTokens != 9 {
		t.Errorf("want a final usage chunk with total 9, got %+v", chunks[len(chunks)-1])
	}
}

func TestE2E_BadJSONIs400(t *testing.T) {
	srv := fakeServer(t, &testdoubles.Provider{})
	resp, _ := post(t, srv, `{not json`)
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("want 400, got %d", resp.StatusCode)
	}
}

func TestE2E_HealthzAndModels(t *testing.T) {
	srv := fakeServer(t, &testdoubles.Provider{ModelIDs: []string{"m1", "m2"}})

	resp, err := http.Get(srv.URL + "/healthz")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("healthz: %v %v", resp, err)
	}
	resp.Body.Close()

	resp, err = http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var list struct {
		Object string `json:"object"`
		Data   []struct{ ID, Object string }
	}
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		t.Fatal(err)
	}
	if list.Object != "list" || len(list.Data) != 2 || list.Data[0].ID != "m1" || list.Data[0].Object != "model" {
		t.Errorf("models: %+v", list)
	}
}

func TestE2E_HealthzWithNoProviders(t *testing.T) {
	// The homelab deployment relies on this: staging runs with no API key and
	// must still pass its readiness probe.
	srv := httptest.NewServer(httpapi.New(app.New()))
	defer srv.Close()
	resp, err := http.Get(srv.URL + "/healthz")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("healthz with no providers: %v %v", resp, err)
	}
	resp.Body.Close()
}
