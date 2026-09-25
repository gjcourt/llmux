// Package prometheus is the outbound telemetry adapter: it records the
// application core's chat observations as Prometheus metrics and serves them.
//
// Every series is labelled by provider and model. The model label is the
// requested model id of a request some provider served; the application core
// never passes an unrouted model, so a client cannot grow the series count
// with made-up ids — except through a catch-all provider (vLLM/Ollama), which
// serves whatever id it is sent.
package prometheus

import (
	"net/http"
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/gjcourt/llmux/internal/ports/outbound"
)

// Metrics implements outbound.Metrics on its own registry.
type Metrics struct {
	reg *prometheus.Registry

	requests  *prometheus.CounterVec
	inFlight  *prometheus.GaugeVec
	duration  *prometheus.HistogramVec
	ttft      *prometheus.HistogramVec
	tokens    *prometheus.CounterVec
	searches  *prometheus.CounterVec
	citations *prometheus.CounterVec
	finishes  *prometheus.CounterVec
	noUsage   *prometheus.CounterVec
	authFails *prometheus.CounterVec

	clients []string // known client names; their series are pre-created
}

var _ outbound.Metrics = (*Metrics)(nil)

// New returns Metrics with Go runtime and process collectors registered.
// clients are the names that can appear in the client label (the configured
// client keys, or just "anonymous" when keys are off); see Declare.
func New(clients []string) *Metrics {
	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))
	f := func(c prometheus.Collector) { reg.MustRegister(c) }
	m := &Metrics{reg: reg, clients: clients}

	m.requests = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "llmux_chat_requests_total",
		Help: "Chat requests by provider, model, streaming mode and outcome.",
	}, []string{"client", "provider", "model", "stream", "outcome"})
	m.inFlight = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "llmux_chat_in_flight",
		Help: "Chat requests currently being served.",
	}, []string{"client", "provider", "model"})
	m.duration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "llmux_chat_duration_seconds",
		Help: "Time from request to the provider finishing a successful answer.",
		// Chat answers stream for seconds to minutes; searched ones longer.
		Buckets: []float64{0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120, 240, 480},
	}, []string{"client", "provider", "model", "stream"})
	m.ttft = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "llmux_chat_time_to_first_token_seconds",
		Help:    "Time from request to the first text or tool-call token, whatever the outcome (a cancelled answer that got a token counts). Web search happens before the first token. For a non-streamed vLLM/Ollama answer it is the whole answer's time.",
		Buckets: []float64{0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60},
	}, []string{"client", "provider", "model", "stream"})
	m.tokens = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "llmux_tokens_total",
		Help: "Tokens by provider, model and type: input (uncached), cache_read, cache_write, output.",
	}, []string{"client", "provider", "model", "type"})
	m.searches = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "llmux_web_searches_total",
		Help: "Server-side web searches the provider ran.",
	}, []string{"client", "provider", "model"})
	m.citations = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "llmux_citations_total",
		Help: "Distinct sources cited in answers.",
	}, []string{"client", "provider", "model"})
	m.finishes = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "llmux_chat_finish_reasons_total",
		Help: "Finished answers by finish_reason (stop, length, content_filter, tool_calls).",
	}, []string{"client", "provider", "model", "reason"})
	m.noUsage = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "llmux_chat_usage_missing_total",
		Help: "Successful answers whose provider reported no token usage, so llmux_tokens_total undercounts them. (Failed requests often legitimately have none.)",
	}, []string{"client", "provider", "model"})

	m.authFails = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "llmux_auth_failures_total",
		Help: "Requests rejected for a missing or invalid llmux client key. No client label: the caller is unknown.",
	}, []string{"reason"})
	for _, c := range []prometheus.Collector{m.requests, m.inFlight, m.duration, m.ttft, m.tokens, m.searches, m.citations, m.finishes, m.noUsage, m.authFails} {
		f(c)
	}
	// Series every deployment has, pre-created at zero (see Declare for why).
	for _, r := range []string{"missing", "invalid"} {
		m.authFails.WithLabelValues(r)
	}
	for _, c := range clients {
		for _, stream := range []string{"true", "false"} {
			m.requests.WithLabelValues(c, "none", "unrouted", stream, string(outbound.OutcomeNoProvider))
		}
	}
	return m
}

// Declare creates provider's series for models at zero. A series that first
// appears already at 1 has no earlier sample, so Prometheus' increase() and
// rate() can't count that first request — on a low-traffic deployment that
// loses the first request of every new model/outcome pair. Call it for
// providers whose models are known up front.
func (m *Metrics) Declare(provider string, models []string) {
	outcomes := []outbound.Outcome{
		outbound.OutcomeOK, outbound.OutcomeInvalidRequest, outbound.OutcomeUpstream4xx, outbound.OutcomeUpstream5xx,
		outbound.OutcomeUnavailable, outbound.OutcomeCanceled, outbound.OutcomeError,
	}
	for _, c := range m.clients {
		for _, mo := range models {
			for _, stream := range []string{"true", "false"} {
				for _, o := range outcomes {
					m.requests.WithLabelValues(c, provider, mo, stream, string(o))
				}
				m.duration.WithLabelValues(c, provider, mo, stream)
				m.ttft.WithLabelValues(c, provider, mo, stream)
			}
			for _, t := range []string{"input", "cache_read", "cache_write", "output"} {
				m.tokens.WithLabelValues(c, provider, mo, t)
			}
			for _, r := range []string{"stop", "length", "content_filter"} {
				m.finishes.WithLabelValues(c, provider, mo, r)
			}
			m.inFlight.WithLabelValues(c, provider, mo)
			m.searches.WithLabelValues(c, provider, mo)
			m.citations.WithLabelValues(c, provider, mo)
			m.noUsage.WithLabelValues(c, provider, mo)
		}
	}
}

// ChatStarted implements outbound.Metrics.
func (m *Metrics) ChatStarted(client, provider, model string) {
	m.inFlight.WithLabelValues(client, provider, model).Inc()
}

// ChatFinished implements outbound.Metrics.
func (m *Metrics) ChatFinished(o outbound.ChatObservation) {
	c, p, mo := o.Client, o.Provider, o.Model
	m.requests.WithLabelValues(c, p, mo, strconv.FormatBool(o.Stream), string(o.Outcome)).Inc()
	if o.Outcome == outbound.OutcomeNoProvider {
		return // nothing was served: no in-flight entry, no latency to speak of
	}
	m.inFlight.WithLabelValues(c, p, mo).Dec()
	if o.Outcome == outbound.OutcomeOK {
		// Successful answers only: fast 4xx/529s and cancellations would
		// otherwise drag the percentiles down and read as "faster".
		m.duration.WithLabelValues(c, p, mo, strconv.FormatBool(o.Stream)).Observe(o.Duration.Seconds())
	}
	if o.TimeToFirstToken > 0 {
		m.ttft.WithLabelValues(c, p, mo, strconv.FormatBool(o.Stream)).Observe(o.TimeToFirstToken.Seconds())
	}
	if o.FinishReason != "" {
		m.finishes.WithLabelValues(c, p, mo, o.FinishReason).Inc()
	}
	if o.Citations > 0 {
		m.citations.WithLabelValues(c, p, mo).Add(float64(o.Citations))
	}
	// Usage is counted whenever it was reported, failures included: tokens
	// consumed by an answer that later failed were still billed.
	if u := o.Usage; u != nil {
		uncached := max(u.PromptTokens-u.CacheReadTokens-u.CacheWriteTokens, 0)
		m.tokens.WithLabelValues(c, p, mo, "input").Add(float64(uncached))
		m.tokens.WithLabelValues(c, p, mo, "cache_read").Add(float64(u.CacheReadTokens))
		m.tokens.WithLabelValues(c, p, mo, "cache_write").Add(float64(u.CacheWriteTokens))
		m.tokens.WithLabelValues(c, p, mo, "output").Add(float64(u.CompletionTokens))
		if u.WebSearches > 0 {
			m.searches.WithLabelValues(c, p, mo).Add(float64(u.WebSearches))
		}
	} else if o.Outcome == outbound.OutcomeOK {
		m.noUsage.WithLabelValues(c, p, mo).Inc()
	}
}

// AuthFailed counts a request rejected for its client key; reason is
// "missing" or "invalid". Wire it to httpapi.WithAuthFailureHook.
func (m *Metrics) AuthFailed(reason string) {
	m.authFails.WithLabelValues(reason).Inc()
}

// Handler serves the metrics in the Prometheus text format.
func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.reg, promhttp.HandlerOpts{Registry: m.reg})
}
