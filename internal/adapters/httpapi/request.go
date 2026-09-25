package httpapi

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/gjcourt/llmux/internal/domain"
)

// wireRequest is the subset of an OpenAI chat-completions request llmux reads.
type wireRequest struct {
	Model         string        `json:"model"`
	Messages      []wireMessage `json:"messages"`
	Stream        bool          `json:"stream"`
	StreamOptions *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	N                   *float64          `json:"n"`
	MaxTokens           *float64          `json:"max_tokens"`
	MaxCompletionTokens *float64          `json:"max_completion_tokens"`
	Temperature         *float64          `json:"temperature"`
	TopP                *float64          `json:"top_p"`
	Stop                json.RawMessage   `json:"stop"`
	Tools               []json.RawMessage `json:"tools"`
}

type wireMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCallID string          `json:"tool_call_id"`
	ToolCalls  []struct {
		ID       string `json:"id"`
		Function struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls"`
}

// parseRequest converts an OpenAI chat-completions body into a ChatRequest.
// The body is kept verbatim in Raw for providers that forward it.
//
// Only a body that isn't a JSON object, or that asks for something llmux
// cannot represent (n > 1), is rejected. Fields llmux merely relays — odd
// content shapes, tool-call arguments sent as objects (Ollama's native shape),
// integral numbers written as 1024.0, malformed stop — are read leniently,
// because a provider that forwards Raw never looks at them and the original
// proxy forwarded such bodies untouched.
func parseRequest(body []byte) (domain.ChatRequest, error) {
	var w wireRequest
	if err := json.Unmarshal(body, &w); err != nil {
		return domain.ChatRequest{}, fmt.Errorf("invalid JSON body: %w", err)
	}
	if w.N != nil && *w.N > 1 {
		return domain.ChatRequest{}, fmt.Errorf("n > 1 is not supported: llmux relays a single choice")
	}
	req := domain.ChatRequest{
		Model:       w.Model,
		Stream:      w.Stream,
		MaxTokens:   intPtr(w.MaxTokens),
		Temperature: w.Temperature,
		TopP:        w.TopP,
		Tools:       w.Tools,
		Raw:         body,
	}
	if req.MaxTokens == nil {
		req.MaxTokens = intPtr(w.MaxCompletionTokens)
	}
	if w.StreamOptions != nil {
		req.IncludeUsage = w.StreamOptions.IncludeUsage
	}
	req.Stop = parseStop(w.Stop)

	for _, m := range w.Messages {
		text, nonText := messageText(m.Content)
		dm := domain.Message{Role: m.Role, Content: text, NonText: nonText, ToolCallID: m.ToolCallID}
		for _, tc := range m.ToolCalls {
			dm.ToolCalls = append(dm.ToolCalls, domain.ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: argumentsText(tc.Function.Arguments)})
		}
		req.Messages = append(req.Messages, dm)
	}
	return req, nil
}

// messageText flattens OpenAI message content — a string, null, or an array of
// parts — into its text. Non-text parts (images, audio) are skipped and
// reported as nonText; they survive in ChatRequest.Raw for providers that
// forward the original body. Any other shape yields "".
func messageText(raw json.RawMessage) (text string, nonText bool) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", false
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, false
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", false
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			b.WriteString(p.Text)
		} else {
			nonText = true
		}
	}
	return b.String(), nonText
}

// argumentsText returns tool-call arguments as a JSON string, whether the
// client sent a string (OpenAI) or an object (Ollama's native shape).
func argumentsText(raw json.RawMessage) string {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	return string(raw)
}

func intPtr(f *float64) *int {
	if f == nil {
		return nil
	}
	i := int(*f)
	return &i
}

// parseStop accepts OpenAI's stop, a string or an array of strings. Anything
// else is ignored.
func parseStop(raw json.RawMessage) []string {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []string{s}
	}
	var ss []string
	if err := json.Unmarshal(raw, &ss); err != nil {
		return nil
	}
	return ss
}
