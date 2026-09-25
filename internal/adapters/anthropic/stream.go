package anthropic

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/gjcourt/llmux/internal/domain"
)

// streamEvent is one Messages API server-sent event. Only the fields llmux
// reads are declared; content_block is also kept raw, because a paused turn
// must be sent back to the API exactly as received.
type streamEvent struct {
	Type    string `json:"type"`
	Message *struct {
		ID    string   `json:"id"`
		Model string   `json:"model"`
		Usage apiUsage `json:"usage"`
	} `json:"message"`
	Index        int             `json:"index"`
	ContentBlock json.RawMessage `json:"content_block"`
	Delta        *struct {
		Type        string          `json:"type"`
		Text        string          `json:"text"`
		Thinking    string          `json:"thinking"`
		Signature   string          `json:"signature"`
		PartialJSON string          `json:"partial_json"`
		Citation    json.RawMessage `json:"citation"`
		StopReason  string          `json:"stop_reason"`
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
	ServerToolUse            struct {
		WebSearchRequests int `json:"web_search_requests"`
	} `json:"server_tool_use"`
}

// StreamError is an `error` event received after the stream started, e.g.
// overloaded_error.
type StreamError struct {
	Type    string
	Message string
}

func (e *StreamError) Error() string { return "anthropic " + e.Type + ": " + e.Message }

// errorStatus is the HTTP status for an in-stream error type, matching what
// the API returns for the same error as a response status.
func errorStatus(errType string) int {
	switch errType {
	case "overloaded_error":
		return 529
	case "rate_limit_error":
		return http.StatusTooManyRequests
	case "api_error":
		return http.StatusInternalServerError
	}
	return http.StatusBadGateway
}

// finishReason maps an Anthropic stop_reason to OpenAI's finish_reason.
// pause_turn only reaches here when the continuation cap was hit, so the
// answer is incomplete: "length" says so.
func finishReason(stop string) string {
	switch stop {
	case "max_tokens", "pause_turn":
		return "length"
	case "refusal":
		return "content_filter"
	case "tool_use":
		return "tool_calls"
	default: // end_turn, stop_sequence, ""
		return "stop"
	}
}

// streamState carries what spans every turn of one answer: whether Start was
// sent, which sources were already cited, and the running token counts. A
// server-tool turn can pause (stop_reason pause_turn) and be resumed with a
// new request; the client sees one continuous response.
type streamState struct {
	created int64
	started bool
	cited   map[string]bool
	usage   domain.Usage // completed turns

	// The turn being read: its input is known from message_start, its
	// output only from message_delta. Kept so a turn that fails part-way
	// still reports the (billed) input it consumed.
	turnIn, turnOut apiUsage
}

// add returns u plus one turn's usage.
func add(u domain.Usage, in, out apiUsage) domain.Usage {
	prompt := in.InputTokens + in.CacheCreationInputTokens + in.CacheReadInputTokens
	u.PromptTokens += prompt
	u.CompletionTokens += out.OutputTokens
	u.TotalTokens += prompt + out.OutputTokens
	u.CacheReadTokens += in.CacheReadInputTokens
	u.CacheWriteTokens += in.CacheCreationInputTokens
	u.WebSearches += out.ServerToolUse.WebSearchRequests
	return u
}

// spent is everything consumed so far, the unfinished turn included.
func (st *streamState) spent() domain.Usage { return add(st.usage, st.turnIn, st.turnOut) }

func newStreamState(created int64) *streamState {
	return &streamState{created: created, cited: map[string]bool{}}
}

// turn is one Messages API response: why it stopped, and its content blocks
// as the API sent them, for resuming a paused turn.
type turn struct {
	stopReason   string
	blocks       []json.RawMessage
	outputTokens int
	// lossy is set when a block got a delta type llmux doesn't know, so its
	// replayed copy may be incomplete and the turn must not be resumed.
	lossy bool
}

// parseStream reads a single-turn stream and finishes the answer. It is the
// whole story when no server tool can pause.
func parseStream(r io.Reader, sink domain.EventSink, created int64) error {
	st := newStreamState(created)
	t, err := st.parseTurn(r, sink)
	if err != nil {
		return err
	}
	return st.finish(sink, t.stopReason)
}

// finish emits the final Finish and Usage events.
func (st *streamState) finish(sink domain.EventSink, stopReason string) error {
	ferr := sink.Emit(domain.Event{Kind: domain.EventFinish, FinishReason: finishReason(stopReason)})
	// Usage is emitted even when the client is already gone, so telemetry
	// still counts the (complete, billed) answer.
	uerr := sink.Emit(domain.Event{Kind: domain.EventUsage, Usage: st.usage})
	if ferr != nil {
		return ferr
	}
	return uerr
}

// parseTurn reads one Messages API SSE stream, emitting events as it goes,
// and returns once message_stop arrives; a stream that ends before that is an
// error, so a truncated answer is never reported as complete.
//
// Relayed: text, and each cited source once per answer (as a Citation).
// Dropped: thinking blocks (claude-sonnet-5 emits them by default) and the
// server tool's own blocks. Nothing is ever emitted as a tool call — Open
// WebUI would try to execute it.
func (st *streamState) parseTurn(r io.Reader, sink domain.EventSink) (turn, error) {
	sc := bufio.NewScanner(r)
	// Web search results carry encrypted page content, which can make one
	// data line far bigger than bufio's 64 KiB default.
	sc.Buffer(make([]byte, 64*1024), 16<<20)

	var (
		blocks  = map[int]map[string]any{}
		order   []int
		partial = map[int]*strings.Builder{} // input_json_delta per block
		lossy   bool
		stop    string
	)
	st.turnIn, st.turnOut = apiUsage{}, apiUsage{}

	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue // event:, id:, comments, blank separators
		}
		var ev streamEvent
		if err := json.Unmarshal([]byte(strings.TrimSpace(line[len("data:"):])), &ev); err != nil {
			return turn{}, fmt.Errorf("decode anthropic event: %w", err)
		}

		switch ev.Type {
		case "message_start":
			if ev.Message == nil {
				return turn{}, fmt.Errorf("message_start without message")
			}
			st.turnIn = ev.Message.Usage
			if !st.started {
				st.started = true
				if err := sink.Emit(domain.Event{Kind: domain.EventStart, ID: ev.Message.ID, Model: ev.Message.Model, Created: st.created}); err != nil {
					return turn{}, err
				}
			}

		case "content_block_start":
			var b map[string]any
			if err := json.Unmarshal(ev.ContentBlock, &b); err != nil || b == nil {
				return turn{}, fmt.Errorf("decode content block %d: %v", ev.Index, err)
			}
			if _, dup := blocks[ev.Index]; !dup {
				order = append(order, ev.Index)
			}
			blocks[ev.Index] = b
			delete(partial, ev.Index)
			if b["type"] == "text" {
				if text, _ := b["text"].(string); text != "" {
					if err := sink.Emit(domain.Event{Kind: domain.EventText, Text: text}); err != nil {
						return turn{}, err
					}
				}
			}

		case "content_block_delta":
			b := blocks[ev.Index]
			if b == nil || ev.Delta == nil {
				lossy = true // content we can't place; never resume this turn
				continue
			}
			known, err := st.applyDelta(b, ev, partial, sink)
			if err != nil {
				return turn{}, err
			}
			if !known {
				lossy = true
			}

		case "content_block_stop":
			if pj, ok := partial[ev.Index]; ok && blocks[ev.Index] != nil {
				if s := pj.String(); s != "" {
					if !json.Valid([]byte(s)) {
						return turn{}, fmt.Errorf("content block %d: invalid tool input JSON", ev.Index)
					}
					blocks[ev.Index]["input"] = json.RawMessage(s)
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
				st.turnOut = *ev.Usage
				if u := *ev.Usage; u.InputTokens+u.CacheCreationInputTokens+u.CacheReadInputTokens > 0 {
					st.turnIn = u
				}
			}

		case "message_stop":
			st.usage = add(st.usage, st.turnIn, st.turnOut)
			t := turn{stopReason: stop, outputTokens: st.turnOut.OutputTokens, lossy: lossy}
			st.turnIn, st.turnOut = apiUsage{}, apiUsage{}
			for _, i := range order {
				raw, err := json.Marshal(blocks[i])
				if err != nil {
					return turn{}, err
				}
				t.blocks = append(t.blocks, raw)
			}
			return t, nil

		case "error":
			se := &StreamError{Type: "error", Message: "unknown error"}
			if ev.Error != nil {
				se.Type, se.Message = ev.Error.Type, ev.Error.Message
			}
			ue := &domain.UpstreamError{Status: errorStatus(se.Type), ContentType: "application/json", Body: []byte(strings.TrimSpace(line[len("data:"):]))}
			if !st.started {
				// Nothing has reached the client, so it can still get a real
				// status — one that says whether retrying makes sense.
				return turn{}, ue
			}
			// Mid-answer: a streaming client is already answered, but a JSON
			// client has received nothing, so keep the status available.
			return turn{}, fmt.Errorf("%w: %w", se, ue)

		case "ping":
			// keep-alive
		default:
			slog.Debug("ignoring anthropic event", "type", ev.Type)
		}
	}
	if err := sc.Err(); err != nil {
		return turn{}, fmt.Errorf("read anthropic stream: %w", err)
	}
	return turn{}, fmt.Errorf("anthropic stream ended before message_stop: %w", io.ErrUnexpectedEOF)
}

// applyDelta folds one content_block_delta into its block, so a paused turn
// can be replayed, and emits what the client should see.
// It reports whether the delta type was known, i.e. fully folded in.
func (st *streamState) applyDelta(b map[string]any, ev streamEvent, partial map[int]*strings.Builder, sink domain.EventSink) (bool, error) {
	d := ev.Delta
	switch d.Type {
	case "text_delta":
		text, _ := b["text"].(string)
		b["text"] = text + d.Text
		if b["type"] == "text" && d.Text != "" {
			return true, sink.Emit(domain.Event{Kind: domain.EventText, Text: d.Text})
		}
	case "thinking_delta":
		thinking, _ := b["thinking"].(string)
		b["thinking"] = thinking + d.Thinking
	case "signature_delta":
		b["signature"] = d.Signature
	case "input_json_delta":
		if partial[ev.Index] == nil {
			partial[ev.Index] = &strings.Builder{}
		}
		partial[ev.Index].WriteString(d.PartialJSON)
	case "citations_delta":
		if len(d.Citation) == 0 {
			return true, nil
		}
		cites, _ := b["citations"].([]any)
		b["citations"] = append(cites, d.Citation)
		var c struct {
			URL   string `json:"url"`
			Title string `json:"title"`
		}
		if json.Unmarshal(d.Citation, &c) == nil && c.URL != "" && !st.cited[c.URL] {
			st.cited[c.URL] = true
			return true, sink.Emit(domain.Event{Kind: domain.EventCitation, Citation: domain.Citation{URL: c.URL, Title: c.Title}})
		}
	default:
		slog.Debug("ignoring anthropic delta", "type", d.Type)
		return false, nil
	}
	return true, nil
}
