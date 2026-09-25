package app_test

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/gjcourt/llmux/internal/app"
	"github.com/gjcourt/llmux/internal/domain"
	"github.com/gjcourt/llmux/internal/ports/outbound"
	"github.com/gjcourt/llmux/internal/testdoubles"
)

// recordingSink keeps what reaches the inbound side, to prove metering
// passes events through untouched.
type recordingSink struct{ kinds []domain.EventKind }

func (r *recordingSink) Emit(e domain.Event) error { r.kinds = append(r.kinds, e.Kind); return nil }

func TestMetering_Observation(t *testing.T) {
	usage := domain.Usage{PromptTokens: 100, CompletionTokens: 20, TotalTokens: 120, CacheReadTokens: 30, WebSearches: 2}
	p := &testdoubles.Provider{ProviderName: "anthropic", ModelIDs: []string{"claude-sonnet-5"}, Events: []domain.Event{
		{Kind: domain.EventStart, ID: "m"},
		{Kind: domain.EventCitation, Citation: domain.Citation{URL: "https://a"}},
		{Kind: domain.EventText, Text: "hi"},
		{Kind: domain.EventCitation, Citation: domain.Citation{URL: "https://b"}},
		{Kind: domain.EventFinish, FinishReason: "stop"},
		{Kind: domain.EventUsage, Usage: usage},
	}}
	m := &testdoubles.Metrics{}
	sink := &recordingSink{}
	if err := app.New(p).WithMetrics(m).Chat(context.Background(), domain.ChatRequest{Model: "claude-sonnet-5", Stream: true}, sink); err != nil {
		t.Fatal(err)
	}
	if len(sink.kinds) != 6 {
		t.Errorf("events must pass through untouched: %v", sink.kinds)
	}
	if len(m.Started) != 1 || m.Started[0] != "anthropic/claude-sonnet-5" || len(m.Finished) != 1 {
		t.Fatalf("started %v finished %d", m.Started, len(m.Finished))
	}
	o := m.Finished[0]
	if o.Provider != "anthropic" || o.Model != "claude-sonnet-5" || !o.Stream || o.Outcome != outbound.OutcomeOK ||
		o.Citations != 2 || o.FinishReason != "stop" || o.Usage == nil || *o.Usage != usage {
		t.Errorf("observation: %+v", o)
	}
	if o.TimeToFirstToken < 0 || o.TimeToFirstToken > o.Duration {
		t.Errorf("ttft %v not within duration %v", o.TimeToFirstToken, o.Duration)
	}
}

// Time to first token counts only text or tool calls, not Start or citations.
func TestMetering_TimeToFirstToken(t *testing.T) {
	p := &slowProvider{delay: 30 * time.Millisecond}
	m := &testdoubles.Metrics{}
	if err := app.New(p).WithMetrics(m).Chat(context.Background(), domain.ChatRequest{Model: "x"}, &recordingSink{}); err != nil {
		t.Fatal(err)
	}
	o := m.Finished[0]
	if o.TimeToFirstToken < 30*time.Millisecond || o.Duration < o.TimeToFirstToken {
		t.Errorf("ttft %v duration %v", o.TimeToFirstToken, o.Duration)
	}
}

type slowProvider struct{ delay time.Duration }

func (*slowProvider) Name() string                                   { return "slow" }
func (*slowProvider) Handles(string) bool                            { return true }
func (*slowProvider) Models(context.Context) ([]domain.Model, error) { return nil, nil }
func (s *slowProvider) Chat(_ context.Context, _ domain.ChatRequest, sink domain.EventSink) error {
	_ = sink.Emit(domain.Event{Kind: domain.EventStart})
	_ = sink.Emit(domain.Event{Kind: domain.EventCitation})
	time.Sleep(s.delay)
	_ = sink.Emit(domain.Event{Kind: domain.EventText, Text: "x"})
	return sink.Emit(domain.Event{Kind: domain.EventFinish, FinishReason: "stop"})
}

func TestMetering_Outcomes(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	cases := []struct {
		ctx  context.Context
		err  error
		want outbound.Outcome
	}{
		{context.Background(), &domain.InvalidRequestError{Msg: "x"}, outbound.OutcomeInvalidRequest},
		{context.Background(), &domain.UpstreamError{Status: 429}, outbound.OutcomeUpstream4xx},
		{context.Background(), &domain.UpstreamError{Status: 529}, outbound.OutcomeUpstream5xx},
		{context.Background(), fmt.Errorf("mid-stream: %w", &domain.UpstreamError{Status: 503}), outbound.OutcomeUpstream5xx},
		{context.Background(), fmt.Errorf("%w: boom", domain.ErrUnavailable), outbound.OutcomeUnavailable},
		{context.Background(), errors.New("stream ended early"), outbound.OutcomeError},
		{canceled, fmt.Errorf("read: %w", context.Canceled), outbound.OutcomeCanceled},
		{canceled, errors.New("write: broken pipe"), outbound.OutcomeCanceled},
	}
	for _, c := range cases {
		m := &testdoubles.Metrics{}
		p := &testdoubles.Provider{Err: c.err}
		_ = app.New(p).WithMetrics(m).Chat(c.ctx, domain.ChatRequest{Model: "m"}, &recordingSink{})
		if got := m.Finished[0].Outcome; got != c.want {
			t.Errorf("%v: outcome %q, want %q", c.err, got, c.want)
		}
		if m.Finished[0].Usage != nil {
			t.Error("no usage was reported")
		}
	}
}

// A model no provider serves is not used as a label value.
func TestMetering_NoProviderDoesNotLabelModel(t *testing.T) {
	m := &testdoubles.Metrics{}
	svc := app.New(&testdoubles.Provider{ModelIDs: []string{"a"}}).WithMetrics(m)
	if err := svc.Chat(context.Background(), domain.ChatRequest{Model: "attacker-chosen-123"}, &recordingSink{}); !errors.Is(err, domain.ErrNoProvider) {
		t.Fatal(err)
	}
	if len(m.Started) != 0 || len(m.Finished) != 1 {
		t.Fatalf("started %v finished %v", m.Started, m.Finished)
	}
	if o := m.Finished[0]; o.Provider != "none" || o.Model != "unrouted" || o.Outcome != outbound.OutcomeNoProvider {
		t.Errorf("observation: %+v", o)
	}
}

// Without WithMetrics, telemetry is off and nothing panics.
func TestMetering_DefaultIsNop(t *testing.T) {
	if err := app.New(&testdoubles.Provider{}).Chat(context.Background(), domain.ChatRequest{Model: "m"}, &recordingSink{}); err != nil {
		t.Fatal(err)
	}
}

type panicky struct{}

func (panicky) Name() string                                   { return "panicky" }
func (panicky) Handles(string) bool                            { return true }
func (panicky) Models(context.Context) ([]domain.Model, error) { return nil, nil }
func (panicky) Chat(context.Context, domain.ChatRequest, domain.EventSink) error {
	panic("boom")
}

// A panicking provider still balances ChatStarted, and the panic propagates.
func TestMetering_PanicStillFinishes(t *testing.T) {
	m := &testdoubles.Metrics{}
	func() {
		defer func() {
			if r := recover(); r != "boom" {
				t.Errorf("panic must propagate, got %v", r)
			}
		}()
		_ = app.New(panicky{}).WithMetrics(m).Chat(context.Background(), domain.ChatRequest{Model: "m"}, &recordingSink{})
	}()
	if len(m.Started) != 1 || len(m.Finished) != 1 || m.Finished[0].Outcome != outbound.OutcomeError {
		t.Errorf("started %v finished %+v", m.Started, m.Finished)
	}
}

func TestMetering_RedirectIsError(t *testing.T) {
	m := &testdoubles.Metrics{}
	_ = app.New(&testdoubles.Provider{Err: &domain.UpstreamError{Status: 307}}).WithMetrics(m).Chat(context.Background(), domain.ChatRequest{Model: "m"}, &recordingSink{})
	if got := m.Finished[0].Outcome; got != outbound.OutcomeError {
		t.Errorf("3xx outcome %q", got)
	}
}
