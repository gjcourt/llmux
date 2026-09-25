package anthropic

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"strings"

	"github.com/gjcourt/llmux/internal/domain"
)

// streamEvent is one Messages API server-sent event. Only the fields llmux
// reads are declared.
type streamEvent struct {
	Type    string `json:"type"`
	Message *struct {
		ID    string   `json:"id"`
		Model string   `json:"model"`
		Usage apiUsage `json:"usage"`
	} `json:"message"`
	Index        int `json:"index"`
	ContentBlock *struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content_block"`
	Delta *struct {
		Type       string `json:"type"`
		Text       string `json:"text"`
		StopReason string `json:"stop_reason"`
	} `json:"delta"`
	Usage *apiUsage `json:"usage"`
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

type apiUsage struct {
	InputTokens              int `json:"input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	OutputTokens             int `json:"output_tokens"`
}

// StreamError is an `error` event received after the stream started, e.g.
// overloaded_error.
type StreamError struct {
	Type    string
	Message string
}

func (e *StreamError) Error() string { return "anthropic " + e.Type + ": " + e.Message }

// finishReason maps an Anthropic stop_reason to OpenAI's finish_reason.
func finishReason(stop string) string {
	switch stop {
	case "max_tokens":
		return "length"
	case "refusal":
		return "content_filter"
	case "tool_use":
		return "tool_calls"
	default: // end_turn, stop_sequence, pause_turn, ""
		return "stop"
	}
}

// parseStream reads a Messages API SSE stream and emits events. It returns nil
// only after message_stop; a stream that ends early is an error, so a
// truncated answer is never reported as complete.
//
// Only text content is relayed. Thinking blocks (claude-sonnet-5 emits them
// by default) and every other block type are dropped: the OpenAI wire format
// has no place for them that Open WebUI renders as intended, and tool-use
// blocks must never surface as tool_calls — Open WebUI would try to execute
// them.
func parseStream(r io.Reader, sink domain.EventSink, created int64) error {
	sc := bufio.NewScanner(r)
	// Large blocks (web search results carry encrypted page content) can make
	// a single data line far bigger than bufio's 64 KiB default.
	sc.Buffer(make([]byte, 64*1024), 16<<20)

	textBlocks := map[int]bool{}
	var in, out apiUsage
	stop := ""

	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue // event:, id:, comments, blank separators
		}
		var ev streamEvent
		if err := json.Unmarshal([]byte(strings.TrimSpace(line[len("data:"):])), &ev); err != nil {
			return fmt.Errorf("decode anthropic event: %w", err)
		}

		switch ev.Type {
		case "message_start":
			if ev.Message == nil {
				return fmt.Errorf("message_start without message")
			}
			in = ev.Message.Usage
			if err := sink.Emit(domain.Event{Kind: domain.EventStart, ID: ev.Message.ID, Model: ev.Message.Model, Created: created}); err != nil {
				return err
			}

		case "content_block_start":
			if ev.ContentBlock != nil && ev.ContentBlock.Type == "text" {
				textBlocks[ev.Index] = true
				if ev.ContentBlock.Text != "" {
					if err := sink.Emit(domain.Event{Kind: domain.EventText, Text: ev.ContentBlock.Text}); err != nil {
						return err
					}
				}
			}

		case "content_block_delta":
			if ev.Delta != nil && ev.Delta.Type == "text_delta" && textBlocks[ev.Index] && ev.Delta.Text != "" {
				if err := sink.Emit(domain.Event{Kind: domain.EventText, Text: ev.Delta.Text}); err != nil {
					return err
				}
			}

		case "message_delta":
			if ev.Delta != nil && ev.Delta.StopReason != "" {
				stop = ev.Delta.StopReason
			}
			if ev.Usage != nil {
				// message_delta usage is cumulative and supersedes
				// message_start's: after a server tool runs, input_tokens
				// here includes the search results (measured: 2,822 at
				// start, 29,268 here). Older API versions sent only
				// output_tokens, so keep start's input counts in that case.
				out = *ev.Usage
				if u := *ev.Usage; u.InputTokens+u.CacheCreationInputTokens+u.CacheReadInputTokens > 0 {
					in = u
				}
			}

		case "message_stop":
			if err := sink.Emit(domain.Event{Kind: domain.EventFinish, FinishReason: finishReason(stop)}); err != nil {
				return err
			}
			prompt := in.InputTokens + in.CacheCreationInputTokens + in.CacheReadInputTokens
			usage := domain.Usage{PromptTokens: prompt, CompletionTokens: out.OutputTokens, TotalTokens: prompt + out.OutputTokens}
			return sink.Emit(domain.Event{Kind: domain.EventUsage, Usage: usage})

		case "error":
			se := &StreamError{Type: "error", Message: "unknown error"}
			if ev.Error != nil {
				se.Type, se.Message = ev.Error.Type, ev.Error.Message
			}
			return se

		case "ping", "content_block_stop":
			// nothing to relay
		default:
			slog.Debug("ignoring anthropic event", "type", ev.Type)
		}
	}
	if err := sc.Err(); err != nil {
		return fmt.Errorf("read anthropic stream: %w", err)
	}
	return fmt.Errorf("anthropic stream ended before message_stop: %w", io.ErrUnexpectedEOF)
}
