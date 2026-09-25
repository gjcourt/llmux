// Package outbound declares the driven ports: what the application needs from
// the model backends it talks to.
package outbound

import (
	"context"

	"github.com/gjcourt/llmux/internal/domain"
)

// ChatProvider is one model backend.
type ChatProvider interface {
	// Name identifies the provider in logs.
	Name() string

	// Handles reports whether this provider serves model.
	Handles(model string) bool

	// Chat streams the answer to req into sink. It must not emit anything
	// before deciding the request can be served: an error returned before the
	// first event (for example *domain.UpstreamError) lets the inbound adapter
	// relay a clean HTTP status.
	Chat(ctx context.Context, req domain.ChatRequest, sink domain.EventSink) error

	// Models lists the models this provider serves.
	Models(ctx context.Context) ([]domain.Model, error)
}
