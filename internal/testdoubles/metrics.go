package testdoubles

import (
	"sync"

	"github.com/gjcourt/llmux/internal/ports/outbound"
)

// Metrics records what the application core reports to outbound.Metrics.
type Metrics struct {
	mu       sync.Mutex
	Started  []string // "provider/model"
	Finished []outbound.ChatObservation
}

var _ outbound.Metrics = (*Metrics)(nil)

// ChatStarted implements outbound.Metrics.
func (m *Metrics) ChatStarted(provider, model string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Started = append(m.Started, provider+"/"+model)
}

// ChatFinished implements outbound.Metrics.
func (m *Metrics) ChatFinished(o outbound.ChatObservation) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Finished = append(m.Finished, o)
}
