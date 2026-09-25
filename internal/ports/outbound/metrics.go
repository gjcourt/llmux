package outbound

import (
	"time"

	"github.com/gjcourt/llmux/internal/domain"
)

// Outcome classifies how a chat request ended, for telemetry.
type Outcome string

// Outcomes. The set is closed so metric label cardinality stays bounded.
const (
	OutcomeOK             Outcome = "ok"
	OutcomeInvalidRequest Outcome = "invalid_request" // llmux refused it (400)
	OutcomeNoProvider     Outcome = "no_provider"     // no provider serves the model (404)
	OutcomeUpstream4xx    Outcome = "upstream_4xx"
	OutcomeUpstream5xx    Outcome = "upstream_5xx" // includes mid-stream errors (overloaded, …)
	OutcomeUnavailable    Outcome = "unavailable"  // upstream unreachable
	OutcomeCanceled       Outcome = "canceled"     // the client went away
	OutcomeError          Outcome = "error"        // anything else (truncated stream, idle timeout, …)
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
