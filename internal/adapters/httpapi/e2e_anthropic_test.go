package httpapi_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/gjcourt/llmux/internal/adapters/anthropic"
	"github.com/gjcourt/llmux/internal/adapters/httpapi"
	"github.com/gjcourt/llmux/internal/app"
)

// anthropicStack runs handler → app → anthropic against a fake Messages API
// that replays a captured stream. It records the upstream request body.
func anthropicStack(t *testing.T, fixture string, upstreamBody *[]byte) *httptest.Server {
	t.Helper()
	sse, err := os.ReadFile("../anthropic/testdata/" + fixture)
	if err != nil {
		t.Fatal(err)
	}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if upstreamBody != nil {
			*upstreamBody, _ = io.ReadAll(r.Body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write(sse) //nolint:errcheck
	}))
	t.Cleanup(up.Close)
	p := anthropic.New(anthropic.Config{APIKey: "k", BaseURL: up.URL, Models: []string{"claude-sonnet-5"}})
	srv := httptest.NewServer(httpapi.New(app.New(p)))
	t.Cleanup(srv.Close)
	return srv
}

func TestE2E_AnthropicStream(t *testing.T) {
	srv := anthropicStack(t, "plain.sse", nil)
	resp, body := post(t, srv, `{"model":"claude-sonnet-5","stream":true,"stream_options":{"include_usage":true},"temperature":0.7,
		"messages":[{"role":"system","content":"brief"},{"role":"user","content":"colors?"}]}`)
	if resp.StatusCode != 200 || resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status %d, type %q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	chunks, done := sseEvents(t, body)
	if !done {
		t.Fatal("stream must end with [DONE]")
	}
	var text, finish string
	var total int
	for _, c := range chunks {
		if c.Model != "claude-sonnet-5" || c.ID != "msg_011CfPd6DCknge5HVTivUBdY" {
			t.Errorf("envelope: %+v", c)
		}
		if c.Usage != nil {
			total = c.Usage.TotalTokens
		}
		if len(c.Choices) == 0 {
			continue
		}
		if len(c.Choices[0].Delta.ToolCalls) > 0 {
			t.Error("no tool calls may be emitted")
		}
		if s := c.Choices[0].Delta.Content; s != nil {
			text += *s
		}
		if fr := c.Choices[0].FinishReason; fr != nil {
			finish = *fr
		}
	}
	if text != "Red and blue." || finish != "stop" || total != 28 {
		t.Errorf("text %q finish %q total %d", text, finish, total)
	}
}

// Open WebUI's non-streaming calls (titles, tags, follow-ups) get one JSON
// chat.completion assembled from the upstream stream.
func TestE2E_AnthropicNonStream(t *testing.T) {
	var up []byte
	srv := anthropicStack(t, "websearch.sse", &up)
	resp, body := post(t, srv, `{"model":"claude-sonnet-5","messages":[{"role":"user","content":"latest open webui?"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %s", resp.StatusCode, body)
	}
	var out struct {
		Object  string `json:"object"`
		Choices []struct {
			Message struct {
				Content   string            `json:"content"`
				ToolCalls []json.RawMessage `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(body), &out); err != nil {
		t.Fatal(err)
	}
	want, _ := os.ReadFile("../anthropic/testdata/websearch.text")
	if out.Object != "chat.completion" || out.Choices[0].Message.Content != string(want) ||
		len(out.Choices[0].Message.ToolCalls) != 0 || out.Choices[0].FinishReason != "stop" || out.Usage.PromptTokens != 29268 {
		t.Errorf("got %+v", out)
	}
	if !strings.Contains(string(up), `"stream":true`) {
		t.Errorf("upstream call must always stream: %s", up)
	}
}

func TestE2E_AnthropicRejectsToolsAndImages(t *testing.T) {
	srv := anthropicStack(t, "plain.sse", nil)
	for name, body := range map[string]string{
		"tools": `{"model":"claude-sonnet-5","stream":true,"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f"}}]}`,
		"image": `{"model":"claude-sonnet-5","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"what is this"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}`,
	} {
		resp, b := post(t, srv, body)
		if resp.StatusCode != 400 || resp.Header.Get("Content-Type") != "application/json" || !strings.Contains(b, "not supported") {
			t.Errorf("%s: status %d type %q body %s", name, resp.StatusCode, resp.Header.Get("Content-Type"), b)
		}
	}
}

// Citations reach the client as OpenAI url_citation annotations — what Open
// WebUI renders as source chips — and never as tool_calls, which it would try
// to execute.
func TestE2E_AnthropicCitationsStream(t *testing.T) {
	srv := anthropicStack(t, "websearch.sse", nil)
	_, body := post(t, srv, `{"model":"claude-sonnet-5","stream":true,"messages":[{"role":"user","content":"latest open webui?"}]}`)
	var urls []string
	for line := range strings.SplitSeq(body, "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var c struct {
			Choices []struct {
				Delta map[string]json.RawMessage `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &c); err != nil {
			t.Fatal(err)
		}
		for _, ch := range c.Choices {
			if _, bad := ch.Delta["tool_calls"]; bad {
				t.Fatalf("tool_calls emitted: %s", data)
			}
			if raw, ok := ch.Delta["annotations"]; ok {
				var anns []struct {
					Type        string `json:"type"`
					URLCitation struct {
						URL   string `json:"url"`
						Title string `json:"title"`
					} `json:"url_citation"`
				}
				if err := json.Unmarshal(raw, &anns); err != nil {
					t.Fatal(err)
				}
				for _, a := range anns {
					if a.Type != "url_citation" || a.URLCitation.URL == "" {
						t.Errorf("bad annotation: %s", raw)
					}
					urls = append(urls, a.URLCitation.URL)
				}
			}
		}
	}
	if len(urls) == 0 {
		t.Fatal("no url_citation annotations in the stream")
	}
}

func TestE2E_AnthropicCitationsNonStream(t *testing.T) {
	srv := anthropicStack(t, "websearch.sse", nil)
	_, body := post(t, srv, `{"model":"claude-sonnet-5","messages":[{"role":"user","content":"latest open webui?"}]}`)
	var out struct {
		Choices []struct {
			Message struct {
				Annotations []struct {
					Type string `json:"type"`
				} `json:"annotations"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(body), &out); err != nil {
		t.Fatal(err)
	}
	if len(out.Choices[0].Message.Annotations) == 0 || out.Choices[0].Message.Annotations[0].Type != "url_citation" {
		t.Errorf("want url_citation annotations on the message: %s", body)
	}
}

// fakeAnthropic answers every request with status, headers and body.
func fakeAnthropic(t *testing.T, h http.HandlerFunc) *httptest.Server {
	t.Helper()
	up := httptest.NewServer(h)
	t.Cleanup(up.Close)
	p := anthropic.New(anthropic.Config{APIKey: "k", BaseURL: up.URL, Models: []string{"claude-sonnet-5"}})
	srv := httptest.NewServer(httpapi.New(app.New(p)))
	t.Cleanup(srv.Close)
	return srv
}

const streamReq = `{"model":"claude-sonnet-5","stream":true,"messages":[{"role":"user","content":"x"}]}`

// llmux's own bad key is not the client's problem: 401/403 become 502. 529
// becomes a retryable 503. Retry-After survives.
func TestE2E_AnthropicStatusMapping(t *testing.T) {
	for upstream, want := range map[int]int{401: 502, 403: 502, 429: 429, 529: 503, 400: 400, 301: 502, 307: 502} {
		srv := fakeAnthropic(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Retry-After", "9")
			w.WriteHeader(upstream)
			w.Write([]byte(`{"type":"error","error":{"type":"x","message":"y"}}`)) //nolint:errcheck
		})
		resp, body := post(t, srv, streamReq)
		if resp.StatusCode != want || resp.Header.Get("Retry-After") != "9" || !strings.Contains(body, `"message":"y"`) {
			t.Errorf("upstream %d: got %d, Retry-After %q, body %s", upstream, resp.StatusCode, resp.Header.Get("Retry-After"), body)
		}
	}
}

func TestE2E_AnthropicOverloadedBeforeStartIs503(t *testing.T) {
	srv := fakeAnthropic(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n")) //nolint:errcheck
	})
	resp, body := post(t, srv, streamReq)
	if resp.StatusCode != 503 || !strings.Contains(body, "Overloaded") {
		t.Errorf("got %d %s", resp.StatusCode, body)
	}
}

// After output has started, an upstream error ends the stream with an error
// chunk and [DONE] — never silently, never as a clean finish.
func TestE2E_AnthropicMidStreamErrorChunk(t *testing.T) {
	srv := fakeAnthropic(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range []string{
			`{"type":"message_start","message":{"id":"m","model":"claude-sonnet-5","usage":{"input_tokens":1}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`,
			`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`,
		} {
			w.Write([]byte("data: " + ev + "\n\n")) //nolint:errcheck
		}
	})
	resp, body := post(t, srv, streamReq)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	chunks, done := sseEvents(t, body)
	if !done {
		t.Error("stream must still end with [DONE]")
	}
	last := chunks[len(chunks)-1]
	if last.Error == nil || !strings.Contains(last.Error.Message, "Overloaded") {
		t.Errorf("last chunk must be the error: %+v", last)
	}
	for _, c := range chunks {
		for _, ch := range c.Choices {
			if ch.FinishReason != nil {
				t.Errorf("an errored stream must not carry a finish_reason: %+v", c)
			}
		}
	}
}
