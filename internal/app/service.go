// Package app is the application core: it routes each request to the provider
// that serves its model.
package app

import (
	"context"
	"log/slog"

	"github.com/gjcourt/llmux/internal/domain"
	"github.com/gjcourt/llmux/internal/ports/inbound"
	"github.com/gjcourt/llmux/internal/ports/outbound"
)

// Service implements inbound.ChatService over an ordered list of providers.
type Service struct {
	providers []outbound.ChatProvider
}

var _ inbound.ChatService = (*Service)(nil)

// New returns a Service. Order matters: a request goes to the first provider
// that Handles its model, so list specific providers before catch-alls.
func New(providers ...outbound.ChatProvider) *Service {
	return &Service{providers: providers}
}

// Chat routes req to the first provider that serves req.Model.
func (s *Service) Chat(ctx context.Context, req domain.ChatRequest, sink domain.EventSink) error {
	for _, p := range s.providers {
		if p.Handles(req.Model) {
			slog.Info("routing request", "provider", p.Name(), "model", req.Model, "stream", req.Stream)
			return p.Chat(ctx, req, sink)
		}
	}
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
