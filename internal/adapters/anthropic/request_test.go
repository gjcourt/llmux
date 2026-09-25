package anthropic

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/gjcourt/llmux/internal/domain"
)

func ptr[T any](v T) *T { return &v }

func user(s string) domain.Message { return domain.Message{Role: "user", Content: s} }

func TestBuildRequest_Basic(t *testing.T) {
	req := domain.ChatRequest{
		Model: "claude-sonnet-5",
		Messages: []domain.Message{
			{Role: "system", Content: "be brief"},
			{Role: "developer", Content: "use metric"},
			user("hi"),
			{Role: "assistant", Content: "hello"},
			user("again"),
		},
		MaxTokens: ptr(100),
		Stop:      []string{"END", "  ", ""},
	}
	got, err := buildRequest(req, 8192, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	want := messagesRequest{
		Model:     "claude-sonnet-5",
		MaxTokens: 100,
		System:    "be brief\n\nuse metric",
		Messages: []message{
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "hello"},
			{Role: "user", Content: "again"},
		},
		StopSequences: []string{"END"},
		Stream:        true,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got  %+v\nwant %+v", got, want)
	}
}

func TestBuildRequest_DefaultMaxTokens(t *testing.T) {
	for _, mt := range []*int{nil, ptr(0), ptr(-1)} {
		got, err := buildRequest(domain.ChatRequest{Model: "m", Messages: []domain.Message{user("x")}, MaxTokens: mt}, 8192, 0, false)
		if err != nil {
			t.Fatal(err)
		}
		if got.MaxTokens != 8192 {
			t.Errorf("max_tokens %v: got %d", mt, got.MaxTokens)
		}
	}
}

// The Claude 5 family rejects temperature and top_p with a 400, so they are
// never sent. Pinned on the wire, not just the struct.
func TestBuildRequest_DropsSampling(t *testing.T) {
	got, err := buildRequest(domain.ChatRequest{Model: "m", Messages: []domain.Message{user("x")}, Temperature: ptr(0.2), TopP: ptr(0.9)}, 10, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(got)
	if strings.Contains(string(b), "temperature") || strings.Contains(string(b), "top_p") {
		t.Errorf("sampling params sent: %s", b)
	}
}

func TestBuildRequest_SkipsEmptyAssistantAndSystem(t *testing.T) {
	got, err := buildRequest(domain.ChatRequest{Model: "m", Messages: []domain.Message{
		{Role: "system", Content: "  "}, user("a"), {Role: "assistant", Content: ""}, user("b"),
	}}, 10, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if got.System != "" || len(got.Messages) != 2 {
		t.Errorf("got %+v", got)
	}
}

func TestBuildRequest_Rejects(t *testing.T) {
	cases := map[string]domain.ChatRequest{
		"tools":           {Tools: []json.RawMessage{json.RawMessage(`{}`)}, Messages: []domain.Message{user("x")}},
		"tool role":       {Messages: []domain.Message{user("x"), {Role: "tool", ToolCallID: "c", Content: "42"}}},
		"tool calls":      {Messages: []domain.Message{user("x"), {Role: "assistant", ToolCalls: []domain.ToolCall{{ID: "c", Name: "f"}}}}},
		"empty user":      {Messages: []domain.Message{user("  ")}},
		"image":           {Messages: []domain.Message{{Role: "user", Content: "what is this?", NonText: true}}},
		"image in system": {Messages: []domain.Message{{Role: "system", Content: "x", NonText: true}, user("y")}},
		"unknown role":    {Messages: []domain.Message{{Role: "function", Content: "x"}}},
		"no messages":     {},
		"only system":     {Messages: []domain.Message{{Role: "system", Content: "x"}}},
	}
	for name, req := range cases {
		req.Model = "m"
		_, err := buildRequest(req, 10, 0, false)
		var ire *domain.InvalidRequestError
		if !errors.As(err, &ire) {
			t.Errorf("%s: want InvalidRequestError, got %v", name, err)
		}
	}
}

// Measured: claude-haiku-4-5 400s on a final assistant turn ending in
// whitespace. Earlier assistant turns are left alone.
func TestBuildRequest_TrimsTrailingPrefillWhitespace(t *testing.T) {
	got, err := buildRequest(domain.ChatRequest{Model: "m", Messages: []domain.Message{
		user("a"), {Role: "assistant", Content: "keep  "}, user("b"), {Role: "assistant", Content: "The colour is \u00a0\n"},
	}}, 10, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if got.Messages[1].Content != "keep  " || got.Messages[3].Content != "The colour is" {
		t.Errorf("messages: %+v", got.Messages)
	}
}

// Open WebUI sends its built-in tools on every browser chat. In drop mode the
// definitions are ignored and tool traffic in the history is removed, keeping
// the text — so the conversation reads as plain turns.
func TestBuildRequest_DropClientTools(t *testing.T) {
	req := domain.ChatRequest{Model: "m",
		Tools: []json.RawMessage{json.RawMessage(`{"type":"function","function":{"name":"get_current_timestamp","parameters":{"type":"object","properties":{}}}}`)},
		Messages: []domain.Message{
			user("what time is it?"),
			{Role: "assistant", ToolCalls: []domain.ToolCall{{ID: "c1", Name: "get_current_timestamp", Arguments: "{}"}}},
			{Role: "tool", ToolCallID: "c1", Content: "2026-09-25T15:00:00Z"},
			{Role: "assistant", Content: "It's 3pm UTC."},
			user("thanks"),
			{Role: "assistant", Content: "Let me check.", ToolCalls: []domain.ToolCall{{ID: "c2", Name: "search_chats"}}},
			{Role: "tool", ToolCallID: "c2", Content: "[]"},
			{Role: "assistant", Content: "Nothing found."},
			user("ok"),
		}}
	if _, err := buildRequest(req, 10, 3, false); err == nil {
		t.Fatal("reject mode must still refuse tools")
	}
	got, err := buildRequest(req, 10, 3, true)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(got.Messages)
	want := `[{"role":"user","content":"what time is it?"},{"role":"assistant","content":"It's 3pm UTC."},{"role":"user","content":"thanks"},{"role":"assistant","content":"Let me check.\n\nNothing found."},{"role":"user","content":"ok"}]`
	if string(b) != want {
		t.Errorf("messages:\n got %s\nwant %s", b, want)
	}
	tb, _ := json.Marshal(got.Tools)
	if string(tb) != `[{"type":"web_search_20250305","name":"web_search","max_uses":3}]` {
		t.Errorf("only llmux's own web search may be sent, got %s", tb)
	}
}
