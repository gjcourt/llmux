package anthropic

import (
	"log/slog"
	"strconv"
	"strings"
	"unicode"

	"github.com/gjcourt/llmux/internal/domain"
)

// messagesRequest is the body of POST /v1/messages.
type messagesRequest struct {
	Model         string    `json:"model"`
	MaxTokens     int       `json:"max_tokens"`
	System        string    `json:"system,omitempty"`
	Messages      []message `json:"messages"`
	StopSequences []string  `json:"stop_sequences,omitempty"`
	Stream        bool      `json:"stream"`
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
func buildRequest(req domain.ChatRequest, defaultMaxTokens int) (messagesRequest, error) {
	if len(req.Tools) > 0 {
		return messagesRequest{}, &domain.InvalidRequestError{Msg: "tools are not supported for Anthropic models in llmux yet"}
	}
	if req.Temperature != nil || req.TopP != nil {
		slog.Debug("dropping temperature/top_p: rejected by Claude 5 models", "model", req.Model)
	}

	out := messagesRequest{Model: req.Model, MaxTokens: defaultMaxTokens, Stream: true}
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
		last.Content = strings.TrimRightFunc(last.Content, unicode.IsSpace)
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
