// Package domain holds llmux's provider-neutral chat model: the request a client
// sent, the stream of events a provider produces in answer, and the errors that
// cross the port boundary. It imports nothing internal and only the standard
// library.
package domain

import (
	"encoding/json"
	"errors"
	"fmt"
)

// ChatRequest is a chat-completion request, parsed from the OpenAI wire format
// by the inbound adapter.
type ChatRequest struct {
	Model        string
	Messages     []Message
	Stream       bool
	IncludeUsage bool
	MaxTokens    *int
	Temperature  *float64
	TopP         *float64
	Stop         []string
	Tools        []json.RawMessage

	// Raw is the original OpenAI-format request body. Providers that speak the
	// OpenAI wire format forward it verbatim, so fields llmux does not model
	// (multimodal parts, provider extensions) survive the trip. Providers that
	// do not speak it use the parsed fields above.
	Raw []byte
}

// Message is one conversation turn. Content is the concatenation of the
// message's text parts.
type Message struct {
	Role    string
	Content string // the text parts, concatenated
	// NonText is set when the content also had parts that are not text
	// (images, audio, files). Content alone then under-represents the
	// message, so a provider that cannot forward those parts must refuse it
	// rather than send the text as if it were everything.
	NonText    bool
	ToolCallID string
	ToolCalls  []ToolCall
}

// ToolCall is a complete tool invocation recorded in conversation history.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// EventKind identifies which fields of an Event are meaningful.
type EventKind int

// Event kinds, in the order a well-formed response produces them: one Start,
// then any mix of Text, ToolCall and Citation, then Finish, optionally
// followed by Usage.
const (
	EventStart EventKind = iota + 1
	EventText
	EventToolCall
	EventFinish
	EventUsage
	// EventCitation is a source the answer draws on, e.g. a web search
	// result. It is informational: never something the client should act on.
	EventCitation
)

// Event is one step of a provider's answer.
type Event struct {
	Kind EventKind

	// Start
	ID      string
	Model   string
	Created int64

	// Text
	Text string

	// ToolCall
	ToolCall ToolCallDelta

	// Finish
	FinishReason string

	// Usage
	Usage Usage

	// Citation
	Citation Citation
}

// Citation is a source a provider cites.
type Citation struct {
	URL   string
	Title string
}

// ToolCallDelta is an incremental piece of a tool call. The first delta for a
// given Index carries ID, Type and Name; later deltas append to Arguments.
type ToolCallDelta struct {
	Index          int
	ID             string
	Type           string
	Name           string
	ArgumentsDelta string
}

// Usage is token accounting for one response.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// EventSink receives a provider's events in order. Emit returns an error when
// the consumer can no longer accept events — typically because the client went
// away — and the provider must stop.
type EventSink interface {
	Emit(Event) error
}

// Model is a model a provider can serve. Raw, when set, is the provider's own
// JSON description and is relayed unchanged.
type Model struct {
	ID      string
	OwnedBy string
	Raw     json.RawMessage
}

// ErrNoProvider means no configured provider serves the requested model.
var ErrNoProvider = errors.New("no provider serves this model")

// ErrUnavailable means every upstream a provider could use was unreachable.
var ErrUnavailable = errors.New("upstream unavailable")

// InvalidRequestError is a request llmux can parse but a provider cannot
// serve as asked — for example tools sent to a provider that doesn't support
// them. It maps to HTTP 400.
type InvalidRequestError struct {
	Msg string
}

func (e *InvalidRequestError) Error() string { return e.Msg }

// UpstreamError is a non-success response from an upstream, returned before any
// event was emitted so the inbound adapter can relay its status and body.
type UpstreamError struct {
	Status      int
	ContentType string
	Body        []byte
	RetryAfter  string // the upstream's Retry-After header, if any
}

func (e *UpstreamError) Error() string {
	return fmt.Sprintf("upstream returned HTTP %d", e.Status)
}
