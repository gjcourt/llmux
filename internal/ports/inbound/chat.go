// Package inbound declares the driving ports: what the application offers to
// the adapters that call it.
package inbound

import (
	"context"

	"github.com/gjcourt/llmux/internal/domain"
)

// ChatService answers chat requests and lists the models it can serve.
type ChatService interface {
	// Chat streams the answer to req into sink. An error returned before any
	// event was emitted describes why no answer was produced; an error after
	// events were emitted means the answer is incomplete.
	Chat(ctx context.Context, req domain.ChatRequest, sink domain.EventSink) error

	// Models lists every model any configured provider can serve.
	Models(ctx context.Context) ([]domain.Model, error)
}
