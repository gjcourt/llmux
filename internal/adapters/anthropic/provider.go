// Package anthropic is the outbound adapter for Anthropic's native Messages
// API. It is used instead of Anthropic's OpenAI-compatible endpoint because
// only the native API exposes server tools such as web search (added in a
// later phase) — the compat endpoint silently ignores web_search_options.
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
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
	BaseURL          string           // default https://api.anthropic.com
	Models           []string         // model ids this provider serves and lists
	DefaultMaxTokens int              // used when the request sets none; Anthropic requires one
	Client           *http.Client     // default HTTPClient()
	IdleTimeout      time.Duration    // cancel after this long with no bytes; default 90s
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
		cfg.Client = HTTPClient()
	}
	if cfg.IdleTimeout <= 0 {
		cfg.IdleTimeout = 90 * time.Second
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
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
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
	// Everything after the headers is read through the watchdog, error
	// bodies included: a proxy can send a 5xx status and then stall.
	stream := newIdleReader(resp.Body, p.cfg.IdleTimeout, cancel)
	defer stream.stop()

	if resp.StatusCode >= 300 {
		// On a stall this keeps the status and whatever body arrived.
		raw, _ := io.ReadAll(io.LimitReader(stream, 1<<20))
		return &domain.UpstreamError{Status: resp.StatusCode, ContentType: resp.Header.Get("Content-Type"), Body: raw, RetryAfter: resp.Header.Get("Retry-After")}
	}
	err = parseStream(stream, sink, p.cfg.Now().Unix())
	if err != nil && errors.Is(context.Cause(ctx), errIdle) {
		return fmt.Errorf("anthropic stream: %w", errIdle)
	}
	return err
}

// errIdle is the cause when the upstream stream goes silent. It is distinct
// from context.Canceled, which the handler reads as "the client went away".
var errIdle = errors.New("no data from upstream within the idle timeout")

// idleReader cancels the request when a single Read waits longer than d.
// Anthropic sends ping events while it works (including during web
// searches), so a read that long means a stalled connection, which would
// otherwise hold the client's request open indefinitely — there is
// deliberately no overall timeout. The clock runs only inside Read: time
// spent writing to a slow client is not blamed on the upstream.
type idleReader struct {
	r     io.Reader
	d     time.Duration
	timer *time.Timer
}

func newIdleReader(r io.Reader, d time.Duration, cancel context.CancelCauseFunc) *idleReader {
	t := time.AfterFunc(d, func() { cancel(errIdle) })
	t.Stop()
	return &idleReader{r: r, d: d, timer: t}
}

func (ir *idleReader) Read(p []byte) (int, error) {
	ir.timer.Reset(ir.d)
	n, err := ir.r.Read(p)
	ir.timer.Stop()
	return n, err
}

func (ir *idleReader) stop() { ir.timer.Stop() }

// HTTPClient returns the client the provider should use in production: no
// overall timeout (answers stream as long as they take; the caller's context
// and the idle timeout end them), bounded dial and header waits, and no
// redirects — Go strips Authorization on a cross-host redirect but not
// x-api-key, so following one could hand the key to another host. A 3xx is
// relayed as an upstream error instead.
func HTTPClient() *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			Proxy:                 http.ProxyFromEnvironment,
			DialContext:           (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
			TLSHandshakeTimeout:   10 * time.Second,
			ResponseHeaderTimeout: 60 * time.Second,
			IdleConnTimeout:       90 * time.Second,
		},
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
}
