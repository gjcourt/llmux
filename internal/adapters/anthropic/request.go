package anthropic

import (
	"encoding/json"
	"log/slog"
	"strconv"
	"strings"
	"unicode"

	"github.com/gjcourt/llmux/internal/domain"
)

// messagesRequest is the body of POST /v1/messages.
type messagesRequest struct {
	Model         string      `json:"model"`
	MaxTokens     int         `json:"max_tokens"`
	System        string      `json:"system,omitempty"`
	Messages      []message   `json:"messages"`
	StopSequences []string    `json:"stop_sequences,omitempty"`
	Tools         []webSearch `json:"tools,omitempty"`
	Stream        bool        `json:"stream"`
}

// message content is a string for a translated client message, or the raw
// content blocks of a paused assistant turn being resumed.
type message struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

// webSearch is Anthropic's server-side web search tool. Anthropic runs the
// searches itself; llmux only relays the answer and its citations.
type webSearch struct {
	Type    string `json:"type"`
	Name    string `json:"name"`
	MaxUses int    `json:"max_uses"`
}

// withContinuation appends a paused assistant turn so the API can resume it.
// If the conversation already ends with an assistant message (a client
// prefill), the blocks extend that message rather than starting another.
func (r *messagesRequest) withContinuation(blocks []json.RawMessage) {
	if n := len(r.Messages); n > 0 && r.Messages[n-1].Role == "assistant" {
		last := &r.Messages[n-1]
		var content []json.RawMessage
		switch c := last.Content.(type) {
		case string:
			b, _ := json.Marshal(map[string]string{"type": "text", "text": c})
			content = append(content, b)
		case []json.RawMessage:
			content = c
		}
		content = append(content, blocks...)
		last.Content = content
		return
	}
	r.Messages = append(r.Messages, message{Role: "assistant", Content: blocks})
}

// buildRequest translates a ChatRequest into a Messages API request.
//
// What is deliberately not translated, and why:
//   - temperature and top_p are always dropped. Measured 2026-09-25: the whole
//     Claude 5 family (claude-sonnet-5, claude-opus-5, claude-fable-5-1)
//     rejects either one with HTTP 400 "`temperature` is deprecated for this
//     model"; only claude-haiku-4-5 accepts them. Open WebUI sends them only
//     when a user sets them, but a per-model allow-list would rot, and losing
//     a sampling tweak on Haiku is harmless next to a 400 on everything else.
//   - tools are rejected, not dropped. Silently dropping them would make the
//     model answer as if the user's tools didn't exist.
//   - images and other non-text parts are rejected, for the same reason: a
//     message whose image was dropped would be answered as if it had none.
//
// maxSearches > 0 offers the model the web search tool, capped at that many
// searches per request; 0 leaves it off.
func buildRequest(req domain.ChatRequest, defaultMaxTokens, maxSearches int) (messagesRequest, error) {
	if len(req.Tools) > 0 {
		return messagesRequest{}, &domain.InvalidRequestError{Msg: "tools are not supported for Anthropic models in llmux yet"}
	}
	if req.Temperature != nil || req.TopP != nil {
		slog.Debug("dropping temperature/top_p: rejected by Claude 5 models", "model", req.Model)
	}

	out := messagesRequest{Model: req.Model, MaxTokens: defaultMaxTokens, Stream: true}
	if maxSearches > 0 {
		out.Tools = []webSearch{{Type: "web_search_20250305", Name: "web_search", MaxUses: maxSearches}}
	}
	if req.MaxTokens != nil && *req.MaxTokens > 0 {
		out.MaxTokens = *req.MaxTokens
	}

	var system []string
	for i, m := range req.Messages {
		if m.NonText {
			return messagesRequest{}, &domain.InvalidRequestError{Msg: "message " + strconv.Itoa(i) + " has an image or other non-text part; those are not supported for Anthropic models in llmux yet"}
		}
		switch m.Role {
		case "system", "developer":
			if strings.TrimSpace(m.Content) != "" {
				system = append(system, m.Content)
			}
		case "user", "assistant":
			if len(m.ToolCalls) > 0 {
				return messagesRequest{}, &domain.InvalidRequestError{Msg: "tool calls in history are not supported for Anthropic models in llmux yet"}
			}
			if strings.TrimSpace(m.Content) == "" {
				if m.Role == "assistant" {
					continue // an empty assistant turn carries nothing
				}
				return messagesRequest{}, &domain.InvalidRequestError{Msg: "message " + strconv.Itoa(i) + " is empty"}
			}
			out.Messages = append(out.Messages, message{Role: m.Role, Content: m.Content})
		case "tool":
			return messagesRequest{}, &domain.InvalidRequestError{Msg: "tool results are not supported for Anthropic models in llmux yet"}
		default:
			return messagesRequest{}, &domain.InvalidRequestError{Msg: "unsupported message role " + m.Role}
		}
	}
	if len(out.Messages) == 0 {
		return messagesRequest{}, &domain.InvalidRequestError{Msg: "request has no user or assistant messages"}
	}
	// A conversation ending in an assistant turn (Open WebUI's "continue
	// response") is a prefill. Measured 2026-09-25: claude-haiku-4-5 rejects
	// one ending in whitespace, so trim it; the Claude 5 family rejects
	// prefill outright, and that 400 is relayed as-is — its message says
	// exactly what's wrong.
	if last := &out.Messages[len(out.Messages)-1]; last.Role == "assistant" {
		if text, ok := last.Content.(string); ok {
			last.Content = strings.TrimRightFunc(text, unicode.IsSpace)
		}
	}
	out.System = strings.Join(system, "\n\n")

	// Anthropic rejects whitespace-only stop sequences (HTTP 400).
	for _, s := range req.Stop {
		if strings.TrimSpace(s) != "" {
			out.StopSequences = append(out.StopSequences, s)
		}
	}
	return out, nil
}
