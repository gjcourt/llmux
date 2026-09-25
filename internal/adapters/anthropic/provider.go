// Package anthropic is the outbound adapter for Anthropic's native Messages
// API. It is used instead of Anthropic's OpenAI-compatible endpoint because
// only the native API exposes server tools such as web search (added in a
// later phase) — the compat endpoint silently ignores web_search_options.
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/gjcourt/llmux/internal/domain"
	"github.com/gjcourt/llmux/internal/ports/outbound"
)

// APIVersion is the anthropic-version header llmux speaks.
const APIVersion = "2023-06-01"

// Config configures the provider.
type Config struct {
	APIKey           string
	BaseURL          string   // default https://api.anthropic.com
	Models           []string // model ids this provider serves and lists
	DefaultMaxTokens int      // used when the request sets none; Anthropic requires one
	Client           *http.Client
	Now              func() time.Time // for tests
}

// Provider implements outbound.ChatProvider for Anthropic.
type Provider struct {
	cfg Config
}

var _ outbound.ChatProvider = (*Provider)(nil)

// New returns a Provider, filling defaults.
func New(cfg Config) *Provider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.anthropic.com"
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	if cfg.DefaultMaxTokens <= 0 {
		cfg.DefaultMaxTokens = 8192
	}
	if cfg.Client == nil {
		cfg.Client = http.DefaultClient
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	return &Provider{cfg: cfg}
}

// Name implements outbound.ChatProvider.
func (p *Provider) Name() string { return "anthropic" }

// Handles implements outbound.ChatProvider.
func (p *Provider) Handles(model string) bool { return slices.Contains(p.cfg.Models, model) }

// Models implements outbound.ChatProvider.
func (p *Provider) Models(context.Context) ([]domain.Model, error) {
	out := make([]domain.Model, 0, len(p.cfg.Models))
	for _, id := range p.cfg.Models {
		out = append(out, domain.Model{ID: id, OwnedBy: "anthropic"})
	}
	return out, nil
}

// Chat implements outbound.ChatProvider. The upstream call always streams;
// the inbound adapter assembles a single JSON response when the client asked
// for one.
func (p *Provider) Chat(ctx context.Context, req domain.ChatRequest, sink domain.EventSink) error {
	body, err := buildRequest(req, p.cfg.DefaultMaxTokens)
	if err != nil {
		return err
	}
	b, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("encode anthropic request: %w", err)
	}

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL+"/v1/messages", bytes.NewReader(b))
	if err != nil {
		return err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")
	hreq.Header.Set("x-api-key", p.cfg.APIKey)
	hreq.Header.Set("anthropic-version", APIVersion)

	resp, err := p.cfg.Client.Do(hreq)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return fmt.Errorf("%w: %v", domain.ErrUnavailable, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return &domain.UpstreamError{Status: resp.StatusCode, ContentType: resp.Header.Get("Content-Type"), Body: raw}
	}
	return parseStream(resp.Body, sink, p.cfg.Now().Unix())
}
