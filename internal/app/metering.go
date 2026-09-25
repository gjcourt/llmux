package app

import (
	"context"
	"errors"
	"time"

	"github.com/gjcourt/llmux/internal/domain"
	"github.com/gjcourt/llmux/internal/ports/outbound"
)

// meteringSink passes events through unchanged and notes what telemetry
// needs: when the first token arrived, the usage, citations, finish reason.
// It sees every event, including Usage that the inbound adapter may drop
// because the client didn't ask for it.
type meteringSink struct {
	next      domain.EventSink
	start     time.Time
	now       func() time.Time
	ttft      time.Duration
	usage     *domain.Usage
	citations int
	finish    string
}

func (m *meteringSink) Emit(e domain.Event) error {
	switch e.Kind {
	case domain.EventText, domain.EventToolCall:
		if m.ttft == 0 {
			m.ttft = m.now().Sub(m.start)
		}
	case domain.EventUsage:
		u := e.Usage
		m.usage = &u
	case domain.EventCitation:
		m.citations++
	case domain.EventFinish:
		m.finish = e.FinishReason
	}
	return m.next.Emit(e)
}

// classify maps a provider's error to a telemetry outcome.
func classify(ctx context.Context, err error) outbound.Outcome {
	var ue *domain.UpstreamError
	var ie *domain.InvalidRequestError
	switch {
	case err == nil:
		return outbound.OutcomeOK
	case errors.Is(err, context.Canceled) || ctx.Err() != nil:
		return outbound.OutcomeCanceled
	case errors.As(err, &ie):
		return outbound.OutcomeInvalidRequest
	case errors.Is(err, domain.ErrNoProvider):
		return outbound.OutcomeNoProvider
	case errors.As(err, &ue):
		if ue.Status >= 400 && ue.Status < 500 {
			return outbound.OutcomeUpstream4xx
		}
		return outbound.OutcomeUpstream5xx
	case errors.Is(err, domain.ErrUnavailable):
		return outbound.OutcomeUnavailable
	}
	return outbound.OutcomeError
}

// nopMetrics is the default: telemetry off.
type nopMetrics struct{}

func (nopMetrics) ChatStarted(string, string)            {}
func (nopMetrics) ChatFinished(outbound.ChatObservation) {}
