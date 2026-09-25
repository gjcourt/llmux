// Package testdoubles holds fakes for the outbound ports.
package testdoubles

import (
	"context"
	"slices"

	"github.com/gjcourt/llmux/internal/domain"
	"github.com/gjcourt/llmux/internal/ports/outbound"
)

// Provider is a scripted outbound.ChatProvider. It emits Events in order, then
// returns Err. Requests records every request it was asked to serve.
type Provider struct {
	ProviderName string
	ModelIDs     []string // Handles reports true for these; empty means all
	Events       []domain.Event
	Err          error
	ModelsErr    error

	Requests []domain.ChatRequest
}

var _ outbound.ChatProvider = (*Provider)(nil)

// Name implements outbound.ChatProvider.
func (p *Provider) Name() string {
	if p.ProviderName == "" {
		return "fake"
	}
	return p.ProviderName
}

// Handles implements outbound.ChatProvider.
func (p *Provider) Handles(model string) bool {
	if len(p.ModelIDs) == 0 {
		return true
	}
	return slices.Contains(p.ModelIDs, model)
}

// Chat implements outbound.ChatProvider.
func (p *Provider) Chat(_ context.Context, req domain.ChatRequest, sink domain.EventSink) error {
	p.Requests = append(p.Requests, req)
	for _, e := range p.Events {
		if err := sink.Emit(e); err != nil {
			return err
		}
	}
	return p.Err
}

// Models implements outbound.ChatProvider.
func (p *Provider) Models(context.Context) ([]domain.Model, error) {
	if p.ModelsErr != nil {
		return nil, p.ModelsErr
	}
	out := make([]domain.Model, 0, len(p.ModelIDs))
	for _, id := range p.ModelIDs {
		out = append(out, domain.Model{ID: id, OwnedBy: p.Name()})
	}
	return out, nil
}
