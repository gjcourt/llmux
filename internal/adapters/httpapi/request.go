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
	MaxTokens           *int              `json:"max_tokens"`
	MaxCompletionTokens *int              `json:"max_completion_tokens"`
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
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls"`
}

// parseRequest converts an OpenAI chat-completions body into a ChatRequest.
// The body is kept verbatim in Raw for providers that forward it.
func parseRequest(body []byte) (domain.ChatRequest, error) {
	var w wireRequest
	if err := json.Unmarshal(body, &w); err != nil {
		return domain.ChatRequest{}, fmt.Errorf("invalid JSON body: %w", err)
	}
	req := domain.ChatRequest{
		Model:       w.Model,
		Stream:      w.Stream,
		MaxTokens:   w.MaxTokens,
		Temperature: w.Temperature,
		TopP:        w.TopP,
		Tools:       w.Tools,
		Raw:         body,
	}
	if req.MaxTokens == nil {
		req.MaxTokens = w.MaxCompletionTokens
	}
	if w.StreamOptions != nil {
		req.IncludeUsage = w.StreamOptions.IncludeUsage
	}
	stop, err := parseStop(w.Stop)
	if err != nil {
		return domain.ChatRequest{}, err
	}
	req.Stop = stop

	for i, m := range w.Messages {
		text, err := messageText(m.Content)
		if err != nil {
			return domain.ChatRequest{}, fmt.Errorf("messages[%d]: %w", i, err)
		}
		dm := domain.Message{Role: m.Role, Content: text, ToolCallID: m.ToolCallID}
		for _, tc := range m.ToolCalls {
			dm.ToolCalls = append(dm.ToolCalls, domain.ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
		}
		req.Messages = append(req.Messages, dm)
	}
	return req, nil
}

// messageText flattens OpenAI message content — a string, null, or an array of
// parts — into its text. Non-text parts (images, audio) are skipped here; they
// survive in ChatRequest.Raw for providers that forward the original body.
func messageText(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", fmt.Errorf("content must be a string or an array of parts")
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}
	return b.String(), nil
}

// parseStop accepts OpenAI's stop, which is a string or an array of strings.
func parseStop(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []string{s}, nil
	}
	var ss []string
	if err := json.Unmarshal(raw, &ss); err != nil {
		return nil, fmt.Errorf("stop must be a string or an array of strings")
	}
	return ss, nil
}
