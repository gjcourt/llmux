// Package anthropic is the outbound adapter for Anthropic's native Messages
// API. It is used instead of Anthropic's OpenAI-compatible endpoint because
// only the native API exposes server tools such as web search — the compat
// endpoint silently ignores web_search_options.
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
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

// maxContinuations bounds how many times a paused server-tool turn is resumed
// within one answer. Each resume re-sends the whole conversation, search
// results included, so each costs a full request's input tokens.
const maxContinuations = 3

// Config configures the provider.
type Config struct {
	APIKey           string
	BaseURL          string           // default https://api.anthropic.com
	Models           []string         // model ids this provider serves and lists
	DefaultMaxTokens int              // used when the request sets none; Anthropic requires one
	WebSearchMaxUses int              // searches allowed per request; 0 disables web search
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
// for one. A turn that pauses mid-search (stop_reason pause_turn) is resumed
// with a follow-up request, up to maxContinuations times, and streams on into
// the same response.
func (p *Provider) Chat(ctx context.Context, req domain.ChatRequest, sink domain.EventSink) error {
	body, err := buildRequest(req, p.cfg.DefaultMaxTokens, p.cfg.WebSearchMaxUses)
	if err != nil {
		return err
	}
	st := newStreamState(p.cfg.Now().Unix())
	for resumes := 0; ; resumes++ {
		t, err := p.send(ctx, body, st, sink)
		if err != nil {
			return err
		}
		if t.stopReason != "pause_turn" || resumes == maxContinuations {
			if t.stopReason == "pause_turn" {
				slog.Warn("anthropic turn still paused after max continuations; answer may be incomplete", "model", req.Model, "continuations", resumes)
			}
			return st.finish(sink, t.stopReason)
		}
		slog.Debug("resuming paused anthropic turn", "model", req.Model, "continuation", resumes+1)
		body.withContinuation(t.blocks)
	}
}

// send makes one Messages API call and parses its stream.
func (p *Provider) send(ctx context.Context, body messagesRequest, st *streamState, sink domain.EventSink) (turn, error) {
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	b, err := json.Marshal(body)
	if err != nil {
		return turn{}, fmt.Errorf("encode anthropic request: %w", err)
	}

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL+"/v1/messages", bytes.NewReader(b))
	if err != nil {
		return turn{}, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")
	hreq.Header.Set("x-api-key", p.cfg.APIKey)
	hreq.Header.Set("anthropic-version", APIVersion)

	resp, err := p.cfg.Client.Do(hreq)
	if err != nil {
		if ctx.Err() != nil {
			return turn{}, ctx.Err()
		}
		return turn{}, fmt.Errorf("%w: %v", domain.ErrUnavailable, err)
	}
	defer resp.Body.Close()
	// Everything after the headers is read through the watchdog, error
	// bodies included: a proxy can send a 5xx status and then stall.
	stream := newIdleReader(resp.Body, p.cfg.IdleTimeout, cancel)
	defer stream.stop()

	if resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(stream, 1<<20))
		if errors.Is(context.Cause(ctx), errIdle) {
			// The body stalled part-way; relaying a truncated JSON error
			// would only fail to parse. Report the stall instead (502).
			return turn{}, fmt.Errorf("anthropic returned HTTP %d, then its error body stalled: %w", resp.StatusCode, errIdle)
		}
		// Relayed verbatim only while nothing has been sent; a failed
		// continuation is mid-answer, so it becomes an in-stream error.
		ue := &domain.UpstreamError{Status: resp.StatusCode, ContentType: resp.Header.Get("Content-Type"), Body: raw, RetryAfter: resp.Header.Get("Retry-After")}
		if st.started {
			return turn{}, fmt.Errorf("resuming paused turn: HTTP %d: %s", ue.Status, ue.Body)
		}
		return turn{}, ue
	}
	t, err := st.parseTurn(stream, sink)
	if err != nil && errors.Is(context.Cause(ctx), errIdle) {
		return turn{}, fmt.Errorf("anthropic stream: %w", errIdle)
	}
	return t, err
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
