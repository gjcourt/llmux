package outbound

import (
	"time"

	"github.com/gjcourt/llmux/internal/domain"
)

// Outcome classifies how a chat request ended, for telemetry.
type Outcome string

// Outcomes. The set is closed so metric label cardinality stays bounded.
const (
	OutcomeOK Outcome = "ok"
	// A provider refused the request (400: tools, images, …). The HTTP
	// adapter's own 400/413s (bad JSON, n>1, oversized body) happen before
	// routing and are not observed.
	OutcomeInvalidRequest Outcome = "invalid_request"
	OutcomeNoProvider     Outcome = "no_provider" // no provider serves the model (404)
	OutcomeUpstream4xx    Outcome = "upstream_4xx"
	// Upstream 5xx, and Anthropic's in-stream error events (overloaded, …),
	// which carry a status.
	OutcomeUpstream5xx Outcome = "upstream_5xx"
	OutcomeUnavailable Outcome = "unavailable" // upstream unreachable
	OutcomeCanceled    Outcome = "canceled"    // the client went away
	// Anything else: a truncated or stalled stream, a vLLM/Ollama mid-stream
	// error chunk, an upstream 3xx, a provider panic.
	OutcomeError Outcome = "error"
)

// ChatObservation is everything telemetry learns about one chat request.
type ChatObservation struct {
	Provider string // provider Name(), or "none" when nothing served it
	Model    string // the requested model id, or "unrouted" when nothing served it
	Stream   bool
	Outcome  Outcome

	Duration time.Duration // request start to the provider returning
	// TimeToFirstToken is the delay until the first text or tool-call event;
	// zero when none arrived.
	TimeToFirstToken time.Duration

	// Usage is the provider's token accounting; nil when it reported none.
	Usage        *domain.Usage
	Citations    int    // distinct sources cited
	FinishReason string // "" when the answer never finished
}

// Metrics receives telemetry from the application core. Implementations must
// be safe for concurrent use and must not block: they sit on the request path.
type Metrics interface {
	// ChatStarted is called when a provider starts serving a request.
	ChatStarted(provider, model string)
	// ChatFinished is called once per request, after ChatStarted if a
	// provider served it.
	ChatFinished(ChatObservation)
}
