package prometheus

import (
	"io"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/gjcourt/llmux/internal/domain"
	"github.com/gjcourt/llmux/internal/ports/outbound"
)

func TestMetrics_RecordsAnswer(t *testing.T) {
	m := New([]string{"web"})
	m.ChatStarted("web", "anthropic", "claude-sonnet-5")
	if v := testutil.ToFloat64(m.inFlight.WithLabelValues("web", "anthropic", "claude-sonnet-5")); v != 1 {
		t.Errorf("in flight = %v", v)
	}
	m.ChatFinished(outbound.ChatObservation{Client: "web",
		Provider: "anthropic", Model: "claude-sonnet-5", Stream: true, Outcome: outbound.OutcomeOK,
		Duration: 3 * time.Second, TimeToFirstToken: 800 * time.Millisecond,
		Usage:     &domain.Usage{PromptTokens: 1000, CompletionTokens: 50, CacheReadTokens: 300, CacheWriteTokens: 200, WebSearches: 2},
		Citations: 3, FinishReason: "stop",
	})
	checks := map[string]float64{
		"in_flight":   testutil.ToFloat64(m.inFlight.WithLabelValues("web", "anthropic", "claude-sonnet-5")),
		"requests":    testutil.ToFloat64(m.requests.WithLabelValues("web", "anthropic", "claude-sonnet-5", "true", "ok")),
		"input":       testutil.ToFloat64(m.tokens.WithLabelValues("web", "anthropic", "claude-sonnet-5", "input")),
		"cache_read":  testutil.ToFloat64(m.tokens.WithLabelValues("web", "anthropic", "claude-sonnet-5", "cache_read")),
		"cache_write": testutil.ToFloat64(m.tokens.WithLabelValues("web", "anthropic", "claude-sonnet-5", "cache_write")),
		"output":      testutil.ToFloat64(m.tokens.WithLabelValues("web", "anthropic", "claude-sonnet-5", "output")),
		"searches":    testutil.ToFloat64(m.searches.WithLabelValues("web", "anthropic", "claude-sonnet-5")),
		"citations":   testutil.ToFloat64(m.citations.WithLabelValues("web", "anthropic", "claude-sonnet-5")),
		"finish_stop": testutil.ToFloat64(m.finishes.WithLabelValues("web", "anthropic", "claude-sonnet-5", "stop")),
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
	m := New([]string{"web"})
	m.ChatStarted("web", "p", "m")
	m.ChatFinished(outbound.ChatObservation{Client: "web", Provider: "p", Model: "m", Outcome: outbound.OutcomeUpstream5xx, Usage: &domain.Usage{PromptTokens: 10, CompletionTokens: 5}})
	if v := testutil.ToFloat64(m.tokens.WithLabelValues("web", "p", "m", "output")); v != 5 {
		t.Errorf("output = %v", v)
	}
	m.ChatStarted("web", "p", "m")
	m.ChatFinished(outbound.ChatObservation{Client: "web", Provider: "p", Model: "m", Outcome: outbound.OutcomeOK})
	if v := testutil.ToFloat64(m.noUsage.WithLabelValues("web", "p", "m")); v != 1 {
		t.Errorf("usage missing = %v", v)
	}
	if v := testutil.ToFloat64(m.inFlight.WithLabelValues("web", "p", "m")); v != 0 {
		t.Errorf("in flight = %v", v)
	}
}

// An unrouted request is counted, but never touches the in-flight gauge (it
// was never started) or the latency histograms.
func TestMetrics_NoProvider(t *testing.T) {
	m := New([]string{"web"})
	m.ChatFinished(outbound.ChatObservation{Client: "web", Provider: "none", Model: "unrouted", Outcome: outbound.OutcomeNoProvider})
	if v := testutil.ToFloat64(m.requests.WithLabelValues("web", "none", "unrouted", "false", "no_provider")); v != 1 {
		t.Errorf("requests = %v", v)
	}
	if testutil.CollectAndCount(m.inFlight) != 0 || testutil.CollectAndCount(m.duration) != 0 { // New pre-creates only the unrouted request series
		t.Error("an unrouted request must not create in-flight or latency series")
	}
}

func TestMetrics_Handler(t *testing.T) {
	m := New([]string{"web"})
	m.ChatStarted("web", "anthropic", "claude-haiku-4-5")
	m.ChatFinished(outbound.ChatObservation{Client: "web", Provider: "anthropic", Model: "claude-haiku-4-5", Outcome: outbound.OutcomeOK, Usage: &domain.Usage{PromptTokens: 1}})
	rec := httptest.NewRecorder()
	m.Handler().ServeHTTP(rec, httptest.NewRequest("GET", "/metrics", nil))
	b, _ := io.ReadAll(rec.Body)
	for _, want := range []string{
		`llmux_chat_requests_total{client="web",model="claude-haiku-4-5",outcome="ok",provider="anthropic",stream="false"} 1`,
		`llmux_tokens_total{client="web",model="claude-haiku-4-5",provider="anthropic",type="input"} 1`,
		"go_goroutines", "process_resident_memory_bytes",
	} {
		if !strings.Contains(string(b), want) {
			t.Errorf("missing %q", want)
		}
	}
}

// Declared series exist at zero before any request, so the first request
// is visible to increase().
func TestMetrics_Declare(t *testing.T) {
	m := New([]string{"web"})
	m.Declare("anthropic", []string{"claude-sonnet-5", "claude-haiku-4-5"})
	// per model: 2 stream values × 7 outcomes; plus the client's 2 unrouted
	// series (pre-created by New). No phantom "anonymous" series.
	if n := testutil.CollectAndCount(m.requests); n != 2*2*7+2 {
		t.Errorf("request series = %d, want %d", n, 2*2*7+2)
	}
	if n := testutil.CollectAndCount(m.tokens); n != 2*4 {
		t.Errorf("token series = %d", n)
	}
	if v := testutil.ToFloat64(m.requests.WithLabelValues("web", "anthropic", "claude-sonnet-5", "true", "ok")); v != 0 {
		t.Errorf("declared series must start at 0, got %v", v)
	}
	if v := testutil.ToFloat64(m.requests.WithLabelValues("web", "none", "unrouted", "false", "no_provider")); v != 0 {
		t.Errorf("unrouted series must exist at 0, got %v", v)
	}
}

// Only successful answers feed the duration histogram.
func TestMetrics_DurationOnlyForOK(t *testing.T) {
	m := New([]string{"web"})
	m.ChatStarted("web", "p", "m")
	m.ChatFinished(outbound.ChatObservation{Client: "web", Provider: "p", Model: "m", Outcome: outbound.OutcomeUpstream5xx, Duration: time.Second})
	if n := testutil.CollectAndCount(m.duration); n != 0 {
		t.Errorf("a failed request created %d duration series", n)
	}
}

// Declare must use exactly the label values ChatFinished uses: real traffic
// on a declared model adds no series (a mismatch would leave zero series
// that never move, next to live ones that were born at 1).
func TestMetrics_DeclareMatchesTraffic(t *testing.T) {
	m := New([]string{"web"})
	m.Declare("anthropic", []string{"a"})
	count := func() int {
		n := 0
		for _, c := range []prometheus.Collector{m.requests, m.inFlight, m.tokens, m.searches, m.citations, m.finishes, m.noUsage} {
			n += testutil.CollectAndCount(c)
		}
		return n
	}
	before := count()
	outcomes := []outbound.Outcome{outbound.OutcomeOK, outbound.OutcomeInvalidRequest, outbound.OutcomeUpstream4xx, outbound.OutcomeUpstream5xx, outbound.OutcomeUnavailable, outbound.OutcomeCanceled, outbound.OutcomeError}
	for _, stream := range []bool{true, false} {
		for _, o := range outcomes {
			for _, fr := range []string{"stop", "length", "content_filter"} {
				m.ChatStarted("web", "anthropic", "a")
				m.ChatFinished(outbound.ChatObservation{Client: "web", Provider: "anthropic", Model: "a", Stream: stream, Outcome: o, FinishReason: fr,
					Duration: time.Second, TimeToFirstToken: time.Millisecond, Citations: 1,
					Usage: &domain.Usage{PromptTokens: 10, CompletionTokens: 1, CacheReadTokens: 2, CacheWriteTokens: 1, WebSearches: 1}})
			}
		}
		m.ChatStarted("web", "anthropic", "a")
		m.ChatFinished(outbound.ChatObservation{Client: "web", Provider: "anthropic", Model: "a", Stream: stream, Outcome: outbound.OutcomeOK})
		m.ChatFinished(outbound.ChatObservation{Client: "web", Provider: "none", Model: "unrouted", Stream: stream, Outcome: outbound.OutcomeNoProvider})
	}
	if after := count(); after != before {
		t.Errorf("traffic added %d series beyond the declared ones", after-before)
	}
}

// Every configured client gets its own series; the unrouted ones exist
// whatever providers are configured.
func TestMetrics_SeveralClients(t *testing.T) {
	m := New([]string{"openwebui", "renovate-review"})
	if n := testutil.CollectAndCount(m.requests); n != 2*2 {
		t.Errorf("unrouted series before Declare = %d, want 4", n)
	}
	m.Declare("anthropic", []string{"a", "b"})
	if n := testutil.CollectAndCount(m.tokens); n != 2*2*4 {
		t.Errorf("token series = %d, want 16", n)
	}
}

func TestMetrics_AuthFailed(t *testing.T) {
	m := New(nil)
	m.AuthFailed("missing")
	m.AuthFailed("invalid")
	m.AuthFailed("invalid")
	if v := testutil.ToFloat64(m.authFails.WithLabelValues("invalid")); v != 2 {
		t.Errorf("invalid = %v", v)
	}
	if n := testutil.CollectAndCount(m.authFails); n != 2 {
		t.Errorf("auth failure series = %d", n)
	}
}
