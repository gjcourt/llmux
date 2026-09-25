package httpapi

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"

	"github.com/gjcourt/llmux/internal/domain"
)

// OpenAI wire shapes for streamed chunks.
type sseChunk struct {
	ID      string      `json:"id"`
	Object  string      `json:"object"`
	Created int64       `json:"created,omitempty"`
	Model   string      `json:"model"`
	Choices []sseChoice `json:"choices"`
	Usage   *wireUsage  `json:"usage,omitempty"`
}

type sseChoice struct {
	Index        int      `json:"index"`
	Delta        sseDelta `json:"delta"`
	FinishReason *string  `json:"finish_reason"`
}

type sseDelta struct {
	Role      string             `json:"role,omitempty"`
	Content   *string            `json:"content,omitempty"`
	ToolCalls []sseToolCallDelta `json:"tool_calls,omitempty"`
}

type sseToolCallDelta struct {
	Index    int               `json:"index"`
	ID       string            `json:"id,omitempty"`
	Type     string            `json:"type,omitempty"`
	Function *sseToolFuncDelta `json:"function,omitempty"`
}

type sseToolFuncDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type wireUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// sseSink writes events to a client as OpenAI chat.completion.chunk SSE. It
// sends response headers on the first event, so an error returned before any
// event can still be turned into a normal HTTP error response.
type sseSink struct {
	w            http.ResponseWriter
	flusher      http.Flusher
	includeUsage bool
	started      bool
	id, model    string
	created      int64
}

func newSSESink(w http.ResponseWriter, includeUsage bool) *sseSink {
	f, _ := w.(http.Flusher)
	return &sseSink{w: w, flusher: f, includeUsage: includeUsage}
}

func (s *sseSink) begin() {
	if s.started {
		return
	}
	s.started = true
	h := s.w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	s.w.WriteHeader(http.StatusOK)
}

func (s *sseSink) write(v any) error {
	s.begin()
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", b); err != nil {
		return err
	}
	if s.flusher != nil {
		s.flusher.Flush()
	}
	return nil
}

func (s *sseSink) chunk(d sseDelta, finish *string) sseChunk {
	return sseChunk{
		ID: s.id, Object: "chat.completion.chunk", Created: s.created, Model: s.model,
		Choices: []sseChoice{{Index: 0, Delta: d, FinishReason: finish}},
	}
}

// Emit implements domain.EventSink. Headers are committed by the first chunk
// actually written, not by the first event: an event that writes nothing
// (suppressed usage, a kind this encoder doesn't render) must not turn a later
// error into a 200.
func (s *sseSink) Emit(e domain.Event) error {
	switch e.Kind {
	case domain.EventStart:
		s.id, s.model, s.created = e.ID, e.Model, e.Created
		return s.write(s.chunk(sseDelta{Role: "assistant"}, nil))
	case domain.EventText:
		text := e.Text
		return s.write(s.chunk(sseDelta{Content: &text}, nil))
	case domain.EventToolCall:
		tc := sseToolCallDelta{Index: e.ToolCall.Index, ID: e.ToolCall.ID, Type: e.ToolCall.Type}
		if e.ToolCall.Name != "" || e.ToolCall.ArgumentsDelta != "" {
			tc.Function = &sseToolFuncDelta{Name: e.ToolCall.Name, Arguments: e.ToolCall.ArgumentsDelta}
		}
		return s.write(s.chunk(sseDelta{ToolCalls: []sseToolCallDelta{tc}}, nil))
	case domain.EventFinish:
		fr := e.FinishReason
		return s.write(s.chunk(sseDelta{}, &fr))
	case domain.EventUsage:
		if !s.includeUsage {
			return nil
		}
		u := wireUsage{PromptTokens: e.Usage.PromptTokens, CompletionTokens: e.Usage.CompletionTokens, TotalTokens: e.Usage.TotalTokens}
		return s.write(sseChunk{ID: s.id, Object: "chat.completion.chunk", Created: s.created, Model: s.model, Choices: []sseChoice{}, Usage: &u})
	}
	return nil
}

// done terminates the stream.
func (s *sseSink) done() {
	s.begin()
	fmt.Fprint(s.w, "data: [DONE]\n\n")
	if s.flusher != nil {
		s.flusher.Flush()
	}
}

// fail reports an error inside an already-started stream, then terminates it.
func (s *sseSink) fail(msg string) {
	_ = s.write(map[string]any{"error": map[string]string{"message": msg, "type": "upstream_error"}})
	s.done()
}

// jsonSink accumulates events into a single chat.completion response.
type jsonSink struct {
	id, model    string
	created      int64
	content      strings.Builder
	hasContent   bool
	toolCalls    map[int]*wireToolCall
	finishReason string
	usage        *wireUsage
}

type wireToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// Emit implements domain.EventSink.
func (j *jsonSink) Emit(e domain.Event) error {
	switch e.Kind {
	case domain.EventStart:
		j.id, j.model, j.created = e.ID, e.Model, e.Created
	case domain.EventText:
		j.content.WriteString(e.Text)
		j.hasContent = true
	case domain.EventToolCall:
		if j.toolCalls == nil {
			j.toolCalls = map[int]*wireToolCall{}
		}
		tc, ok := j.toolCalls[e.ToolCall.Index]
		if !ok {
			tc = &wireToolCall{Type: "function"}
			j.toolCalls[e.ToolCall.Index] = tc
		}
		if e.ToolCall.ID != "" {
			tc.ID = e.ToolCall.ID
		}
		if e.ToolCall.Type != "" {
			tc.Type = e.ToolCall.Type
		}
		tc.Function.Name += e.ToolCall.Name
		tc.Function.Arguments += e.ToolCall.ArgumentsDelta
	case domain.EventFinish:
		j.finishReason = e.FinishReason
	case domain.EventUsage:
		j.usage = &wireUsage{PromptTokens: e.Usage.PromptTokens, CompletionTokens: e.Usage.CompletionTokens, TotalTokens: e.Usage.TotalTokens}
	}
	return nil
}

// response renders the accumulated chat.completion. content is a string —
// "" when the answer had no text — except when the answer is only tool calls,
// where OpenAI (and the original transform) use null.
func (j *jsonSink) response() map[string]any {
	msg := map[string]any{"role": "assistant", "content": j.content.String()}
	if len(j.toolCalls) > 0 {
		idx := make([]int, 0, len(j.toolCalls))
		for i := range j.toolCalls {
			idx = append(idx, i)
		}
		sort.Ints(idx)
		calls := make([]*wireToolCall, 0, len(idx))
		for _, i := range idx {
			calls = append(calls, j.toolCalls[i])
		}
		msg["tool_calls"] = calls
		if !j.hasContent {
			msg["content"] = nil
		}
	}
	fr := j.finishReason
	if fr == "" {
		fr = "stop"
	}
	out := map[string]any{
		"id": j.id, "object": "chat.completion", "created": j.created, "model": j.model,
		"choices": []map[string]any{{"index": 0, "message": msg, "finish_reason": fr}},
	}
	if j.usage != nil {
		out["usage"] = j.usage
	}
	return out
}
