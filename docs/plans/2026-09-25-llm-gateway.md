---
title: llmux as the one LLM gateway for the homelab
status: In-Progress
created: 2026-09-25
updated: 2026-09-25
updated_by: gjcourt
tags: [gateway, anthropic, auth, telemetry, homelab]
---

# llmux as the one LLM gateway

## Goal

Every workload in the homelab that calls an LLM goes through llmux. llmux is
the **only** thing that holds a provider API key, and it can say who spent
what. Today Open WebUI is being moved onto llmux (homelab#1471/#1472); this
plan covers everything else.

## Who calls LLMs today

Surveyed 2026-09-25 (every repo under `~/src`, plus the homelab manifests):

| Caller | State | How it calls | Needs |
|---|---|---|---|
| Open WebUI | moving to llmux (homelab#1472) | OpenAI format, streaming | a client key (phase 1) |
| **renovate-review** CronJob | running in the cluster | raw `POST https://api.anthropic.com/v1/messages`, text only, non-streaming, `REVIEW_MODEL=claude-opus-4-8` (→ `claude-opus-5-5` on the move) | Anthropic-format endpoint (phase 2) |
| cadence | being deployed (`homelab` branch `feat/cadence-app`) | `@anthropic-ai/sdk`, off unless `ANTHROPIC_API_KEY` is set | Anthropic-format endpoint |
| yc-starter | not deployed | Python `anthropic` SDK, `claude-sonnet-5`, optional | Anthropic-format endpoint |
| yc-prep | not deployed | commented-out stub | — |

Two facts shape the design:

1. **Everything except Open WebUI speaks Anthropic's native Messages format**,
   and llmux only accepts the OpenAI format.
2. Anthropic's SDKs take a base URL (`ANTHROPIC_BASE_URL`), so once llmux
   accepts the native format these callers move by **configuration only**.

## Design

### Anthropic-format front door: a native pass-through lane

A new inbound adapter serves `POST /v1/messages` (and `GET /v1/models` in
Anthropic's shape). Two ways to build it:

| | Translate through the domain | **Native pass-through (proposed)** |
|---|---|---|
| How | parse → `domain.ChatRequest` → Anthropic provider → re-encode Anthropic SSE | forward the client's body to Anthropic **verbatim**; relay the response bytes back unchanged |
| Fidelity | only what the domain models: no tools, images, thinking, `cache_control`, … — llmux rejects those today | everything the API supports, including features that don't exist yet |
| Work | large (two encoders, growing domain) | small |
| Telemetry | free | a tee: the relayed stream is also parsed for usage, same metrics |

Pass-through wins: these clients already speak Anthropic, so translating them
twice only loses features. Hexagonally it's a second, narrower port rather
than a hole in the architecture:

- `domain.NativeRequest{Format: "anthropic", Model, Stream, Body}`.
- `outbound.NativeProvider` (optional, alongside `ChatProvider`):
  `Relays(format, model) bool` and `Relay(ctx, req, w)`.
  The Anthropic provider implements it: it swaps the client's credentials for
  llmux's key, forwards the body, copies the response through, and reports
  usage from the stream (the same `message_start`/`message_delta` parsing
  used today) to the metering port.
- `app.Service.Relay` does routing, the model allowlist and metering, exactly
  as `Chat` does.

Only same-format relaying (Anthropic-format client → Anthropic). No
Anthropic-format → vLLM translation. llmux's own policy is **not** applied to
the native lane — it doesn't inject web search or change `max_tokens`. The
client decides; llmux routes, authenticates, meters.

### Know who's calling: client keys

llmux has no authentication today; the network policy is the only gate
(Open WebUI's pods alone can reach it). With several callers that stops
being enough, and "who spent what" needs an identity anyway.

- Each client gets its own random llmux key. llmux loads `name → key` from
  a SOPS secret (`LLMUX_CLIENT_KEYS`).
- Accepted as `Authorization: Bearer …` (OpenAI clients) or `x-api-key: …`
  (Anthropic SDKs), so clients put the llmux key where they'd put a
  provider key — no code change.
- Unknown or missing key → 401. With no keys configured, auth is off (local
  development).
- Every metric gains a `client` label; the dashboard gets a "by client" view.
  Revoking one app is removing one line.

The network policy stays as the second layer: llmux's ingress lists each
client namespace explicitly.

### One provider key

As each caller moves, its copy of the Anthropic key is deleted. When the
last one has moved, **rotate the Anthropic key** — it has lived in Open
WebUI's and renovate-review's secrets — and give the new one to llmux only.

## Phases (one PR each unless noted)

1. **Client keys + `client` label** (llmux) — built: `LLMUX_CLIENT_KEYS`, `LLMUX_REQUIRE_CLIENT_KEYS`. Then homelab: Open WebUI's
   `OPENAI_API_KEY` becomes its llmux client key — which also takes the real
   Anthropic key out of Open WebUI.
2. **Native `/v1/messages` pass-through lane** (llmux), with metering and
   the model allowlist.
3. **renovate-review cutover** (homelab): its script builds the URL from an
   env var instead of hardcoding `api.anthropic.com`; its secret holds an
   llmux client key instead of the Anthropic key; netpols both ways.
4. **cadence**, when it's deployed: `ANTHROPIC_BASE_URL` + an llmux client
   key in its manifests (coordinate with the `feat/cadence-app` branch).
5. **Rotate the Anthropic key**, after the end-to-end test; llmux is the
   only holder.

Later, and now simpler because everything is in one place: prompt caching,
per-client budgets or rate limits.

## Decisions (owner, 2026-09-25)

1. **Native pass-through** — yes.
2. **Client keys required in production** — yes.
3. **renovate-review's model** — upgrade to `claude-opus-5-5` (added to
   llmux's model list in homelab#1471).
4. **Rotate the Anthropic key** — yes, but only after an end-to-end test of
   the whole path; not as part of the migration PRs.

## Out of scope

- Callers outside the cluster (Claude Code, local scripts on the Mac). That
  needs llmux exposed beyond the cluster — a separate decision about
  ingress and auth.
- Tools and images on the OpenAI-format lane. Native clients get them via
  pass-through; Open WebUI doesn't need them today.
