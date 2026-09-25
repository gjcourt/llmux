---
title: Usage telemetry — tokens, latency and outcomes per provider and model
status: Implemented
created: 2026-09-25
updated: 2026-09-25
updated_by: gjcourt
tags: [telemetry, prometheus, grafana, hexagonal]
---

# Usage telemetry

## Goal

See what llmux is spending and how it's behaving, per provider and model:
input/output tokens (with cache and web-search detail), request outcomes,
latency and time to first token — in a Grafana dashboard fed by the homelab's
existing Prometheus.

## Design

**A new outbound port, observed in the application core.** Providers already
emit `domain.Event`s (Start, Text, ToolCall, Citation, Finish, Usage).
`app.Service` wraps each request's sink in a `meteringSink` that passes every
event through unchanged and notes the time of the first token, the usage, the
citations and the finish reason. When the provider returns, the service
classifies the error into a closed set of outcomes and hands one
`outbound.ChatObservation` to `outbound.Metrics`. Neither providers nor the
HTTP adapter know telemetry exists, and a new provider is metered for free as
long as it emits `Usage`.

The metering sink sits *inside* the inbound adapter's sink, so it sees the
Usage event even when the client didn't ask for usage and the SSE encoder
drops it.

`domain.Usage` gained `CacheReadTokens`, `CacheWriteTokens` (both included in
`PromptTokens`, OpenAI's meaning) and `WebSearches`. Anthropic reports all
three (`message_delta` usage, summed across `pause_turn` resumes); OpenAI-
compatible upstreams report cached tokens in `prompt_tokens_details`.

A provider panic still records an observation (outcome `error`) before the
panic continues, so the in-flight gauge can't stick.

**Adapter:** `internal/adapters/prometheus`, `prometheus/client_golang` on its
own registry (plus Go runtime and process collectors), served on a **separate
listener**, `LLMUX_METRICS_ADDR` (default `:9090`; empty disables). The chat
port stays reachable from Open WebUI only; the metrics port from Prometheus
only. On SIGTERM the metrics server shuts down only after the chat server has
drained. This is llmux's first third-party dependency — accepted for correct
histograms and the standard runtime metrics over hand-writing the exposition
format.

## Metrics

All labelled `provider`, `model`.

| Metric | Type | Extra labels | Notes |
|---|---|---|---|
| `llmux_chat_requests_total` | counter | `stream`, `outcome` | outcome ∈ ok, invalid_request, no_provider, upstream_4xx, upstream_5xx, unavailable, canceled, error — see `outbound.Outcome` for exactly what each covers. The HTTP adapter's own 400/413s happen before routing and aren't counted |
| `llmux_chat_in_flight` | gauge | | |
| `llmux_chat_duration_seconds` | histogram | `stream` | request → answer done, **successful answers only** (fast failures would read as "faster"); 0.25s–8m buckets |
| `llmux_chat_time_to_first_token_seconds` | histogram | `stream` | first text/tool-call token; search happens before it. For a non-streamed vLLM/Ollama answer it equals the duration |
| `llmux_tokens_total` | counter | `type` | input (uncached), cache_read, cache_write, output |
| `llmux_web_searches_total` | counter | | server-side searches run |
| `llmux_citations_total` | counter | | distinct sources cited |
| `llmux_chat_finish_reasons_total` | counter | `reason` | a rising `length` share means answers are being cut off |
| `llmux_chat_usage_missing_total` | counter | | successful answers with no usage — tokens undercount them |

**Cardinality.** `model` is the requested id of a request some provider
served. An unrouted request is recorded as `provider="none", model="unrouted"`,
so a client can't mint series with made-up ids. The one exception is the
vLLM/Ollama catch-all, which serves any id; it's off by default.

**Tokens are counted whenever reported, failures included** — tokens an
answer consumed before failing were still billed. The Anthropic adapter
reports what it knows on any failure after `message_start`: every completed
turn in full, plus the unfinished turn's input (from `message_start`). The
unfinished turn's *output* is only reported at its end, so a cancelled answer
undercounts output — typically the smaller share. This matters: pressing Stop
in Open WebUI is a cancellation. Such usage travels as a `Partial` event: the
metering sink records it, the wire encoders drop it, so a client never sees a
usage chunk that isn't the answer's real accounting.

## Not included

- **Dollar cost.** A price table in code or the dashboard would go stale
  silently; tokens by type are what a price is applied to.
- **Per-user attribution.** Open WebUI doesn't pass the user to llmux.
- **`pause_turn` resume counts.** Visible in logs; can be added if resumes
  turn out to matter.

## Finding along the way

Measured 2026-09-25 with this telemetry: **offering the web search tool adds
~2.2k (Haiku 4.5) – 2.8k (Sonnet 5) input tokens to every request**, searched
or not — "Say hello in five words" cost 13–14 input tokens with
`LLMUX_WEB_SEARCH_MAX_USES=0` and 2,219 / 2,808 with 3. Open WebUI's
background calls (titles, tags, follow-ups) pay it too.

**Acted on (2026-09-25):** `LLMUX_WEB_SEARCH_STREAM_ONLY` (default `true`)
offers web search to streamed requests only, so Open WebUI's non-streamed
background calls stop paying for it. The dashboard's "Input tokens per
request" panel shows the effect.
