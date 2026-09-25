// Package app is the application core: it routes each request to the provider
// that serves its model.
package app

import (
	"context"
	"log/slog"
	"time"

	"github.com/gjcourt/llmux/internal/domain"
	"github.com/gjcourt/llmux/internal/ports/inbound"
	"github.com/gjcourt/llmux/internal/ports/outbound"
)

// Service implements inbound.ChatService over an ordered list of providers.
type Service struct {
	providers []outbound.ChatProvider
	metrics   outbound.Metrics
	now       func() time.Time
}

var _ inbound.ChatService = (*Service)(nil)

// New returns a Service. Order matters: a request goes to the first provider
// that Handles its model, so list specific providers before catch-alls.
func New(providers ...outbound.ChatProvider) *Service {
	return &Service{providers: providers, metrics: nopMetrics{}, now: time.Now}
}

// WithMetrics sets where telemetry goes (default: nowhere) and returns s.
func (s *Service) WithMetrics(m outbound.Metrics) *Service {
	s.metrics = m
	return s
}

// Chat routes req to the first provider that serves req.Model, recording
// telemetry for every request.
func (s *Service) Chat(ctx context.Context, req domain.ChatRequest, sink domain.EventSink) (err error) {
	start := s.now()
	for _, p := range s.providers {
		if !p.Handles(req.Model) {
			continue
		}
		slog.Info("routing request", "provider", p.Name(), "model", req.Model, "stream", req.Stream)
		s.metrics.ChatStarted(p.Name(), req.Model)
		ms := &meteringSink{next: sink, start: start, now: s.now}
		// Deferred so a panicking provider still balances ChatStarted
		// (net/http recovers handler panics; the in-flight gauge would
		// otherwise stay up for the life of the process).
		defer func() {
			outcome := classify(ctx, err)
			r := recover()
			if r != nil {
				outcome = outbound.OutcomeError
			}
			s.metrics.ChatFinished(outbound.ChatObservation{
				Provider: p.Name(), Model: req.Model, Stream: req.Stream,
				Outcome:  outcome,
				Duration: s.now().Sub(start), TimeToFirstToken: ms.ttft,
				Usage: ms.usage, Citations: ms.citations, FinishReason: ms.finish,
			})
			if r != nil {
				panic(r)
			}
		}()
		return p.Chat(ctx, req, ms)
	}
	// The requested model is client input with no provider behind it; it
	// is not used as a label, so a client can't grow the series count.
	s.metrics.ChatFinished(outbound.ChatObservation{
		Provider: "none", Model: "unrouted", Stream: req.Stream,
		Outcome: outbound.OutcomeNoProvider, Duration: s.now().Sub(start),
	})
	return domain.ErrNoProvider
}

// Models concatenates every provider's models. A provider that fails to list
// is logged and skipped, so one unreachable backend doesn't hide the others.
func (s *Service) Models(ctx context.Context) ([]domain.Model, error) {
	var all []domain.Model
	for _, p := range s.providers {
		ms, err := p.Models(ctx)
		if err != nil {
			slog.Warn("provider failed to list models", "provider", p.Name(), "err", err)
			continue
		}
		all = append(all, ms...)
	}
	return all, nil
}
