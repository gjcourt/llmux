package prometheus

import (
	"io"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/gjcourt/llmux/internal/domain"
	"github.com/gjcourt/llmux/internal/ports/outbound"
)

func TestMetrics_RecordsAnswer(t *testing.T) {
	m := New()
	m.ChatStarted("anthropic", "claude-sonnet-5")
	if v := testutil.ToFloat64(m.inFlight.WithLabelValues("anthropic", "claude-sonnet-5")); v != 1 {
		t.Errorf("in flight = %v", v)
	}
	m.ChatFinished(outbound.ChatObservation{
		Provider: "anthropic", Model: "claude-sonnet-5", Stream: true, Outcome: outbound.OutcomeOK,
		Duration: 3 * time.Second, TimeToFirstToken: 800 * time.Millisecond,
		Usage:     &domain.Usage{PromptTokens: 1000, CompletionTokens: 50, CacheReadTokens: 300, CacheWriteTokens: 200, WebSearches: 2},
		Citations: 3, FinishReason: "stop",
	})
	checks := map[string]float64{
		"in_flight":   testutil.ToFloat64(m.inFlight.WithLabelValues("anthropic", "claude-sonnet-5")),
		"requests":    testutil.ToFloat64(m.requests.WithLabelValues("anthropic", "claude-sonnet-5", "true", "ok")),
		"input":       testutil.ToFloat64(m.tokens.WithLabelValues("anthropic", "claude-sonnet-5", "input")),
		"cache_read":  testutil.ToFloat64(m.tokens.WithLabelValues("anthropic", "claude-sonnet-5", "cache_read")),
		"cache_write": testutil.ToFloat64(m.tokens.WithLabelValues("anthropic", "claude-sonnet-5", "cache_write")),
		"output":      testutil.ToFloat64(m.tokens.WithLabelValues("anthropic", "claude-sonnet-5", "output")),
		"searches":    testutil.ToFloat64(m.searches.WithLabelValues("anthropic", "claude-sonnet-5")),
		"citations":   testutil.ToFloat64(m.citations.WithLabelValues("anthropic", "claude-sonnet-5")),
		"finish_stop": testutil.ToFloat64(m.finishes.WithLabelValues("anthropic", "claude-sonnet-5", "stop")),
	}
	want := map[string]float64{"in_flight": 0, "requests": 1, "input": 500, "cache_read": 300, "cache_write": 200, "output": 50, "searches": 2, "citations": 3, "finish_stop": 1}
	for k, w := range want {
		if checks[k] != w {
			t.Errorf("%s = %v, want %v", k, checks[k], w)
		}
	}
	if n := testutil.CollectAndCount(m.duration); n != 1 {
		t.Errorf("duration series = %d", n)
	}
	if n := testutil.CollectAndCount(m.ttft); n != 1 {
		t.Errorf("ttft series = %d", n)
	}
}

// Tokens from an answer that failed after its usage arrived were still
// billed, so they count; a successful answer with no usage is flagged.
func TestMetrics_UsageEdgeCases(t *testing.T) {
	m := New()
	m.ChatStarted("p", "m")
	m.ChatFinished(outbound.ChatObservation{Provider: "p", Model: "m", Outcome: outbound.OutcomeUpstream5xx, Usage: &domain.Usage{PromptTokens: 10, CompletionTokens: 5}})
	if v := testutil.ToFloat64(m.tokens.WithLabelValues("p", "m", "output")); v != 5 {
		t.Errorf("output = %v", v)
	}
	m.ChatStarted("p", "m")
	m.ChatFinished(outbound.ChatObservation{Provider: "p", Model: "m", Outcome: outbound.OutcomeOK})
	if v := testutil.ToFloat64(m.noUsage.WithLabelValues("p", "m")); v != 1 {
		t.Errorf("usage missing = %v", v)
	}
	if v := testutil.ToFloat64(m.inFlight.WithLabelValues("p", "m")); v != 0 {
		t.Errorf("in flight = %v", v)
	}
}

// An unrouted request is counted, but never touches the in-flight gauge (it
// was never started) or the latency histograms.
func TestMetrics_NoProvider(t *testing.T) {
	m := New()
	m.ChatFinished(outbound.ChatObservation{Provider: "none", Model: "unrouted", Outcome: outbound.OutcomeNoProvider})
	if v := testutil.ToFloat64(m.requests.WithLabelValues("none", "unrouted", "false", "no_provider")); v != 1 {
		t.Errorf("requests = %v", v)
	}
	if testutil.CollectAndCount(m.inFlight) != 0 || testutil.CollectAndCount(m.duration) != 0 {
		t.Error("an unrouted request must not create in-flight or latency series")
	}
}

func TestMetrics_Handler(t *testing.T) {
	m := New()
	m.ChatStarted("anthropic", "claude-haiku-4-5")
	m.ChatFinished(outbound.ChatObservation{Provider: "anthropic", Model: "claude-haiku-4-5", Outcome: outbound.OutcomeOK, Usage: &domain.Usage{PromptTokens: 1}})
	rec := httptest.NewRecorder()
	m.Handler().ServeHTTP(rec, httptest.NewRequest("GET", "/metrics", nil))
	b, _ := io.ReadAll(rec.Body)
	for _, want := range []string{
		`llmux_chat_requests_total{model="claude-haiku-4-5",outcome="ok",provider="anthropic",stream="false"} 1`,
		`llmux_tokens_total{model="claude-haiku-4-5",provider="anthropic",type="input"} 1`,
		"go_goroutines", "process_resident_memory_bytes",
	} {
		if !strings.Contains(string(b), want) {
			t.Errorf("missing %q", want)
		}
	}
}
