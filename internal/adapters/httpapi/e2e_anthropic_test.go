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
