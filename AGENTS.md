# AGENTS.md

> llmux is an OpenAI-compatible HTTP proxy that routes chat requests to model backends. Backends: Anthropic (native Messages API, with server-side web search and cited sources) and vLLM/Ollama, where it repairs malformed tool calls and strips `<think>` blocks. — https://github.com/gjcourt/llmux

## Commands

| Command | Use |
|---------|-----|
| `go build -o llmux ./cmd/llmux` | Compile to `./llmux` |
| `go test -race ./...` | Run all tests with race detector |
| `go vet ./...` | Static analysis |
| `go fmt ./...` | gofmt |
| `go-arch-lint check` | Enforce the hexagonal dependency rule |

Single test: `go test ./internal/adapters/openaicompat/ -run TestTransform_SingleToolCall -v`
Pre-push: `go fmt ./... && go vet ./... && go test -race ./... && go-arch-lint check`

## Architecture

Hexagonal, enforced by `go-arch-lint` (`.go-arch-lint.yml`, golinks' config):

```
cmd/llmux/                      config + wiring
internal/domain/                ChatRequest, Event, EventSink, errors — stdlib only
internal/ports/inbound/         ChatService
internal/ports/outbound/        ChatProvider
internal/app/                   routes a request to the first provider that Handles(model)
internal/adapters/httpapi/      inbound: OpenAI-compatible /v1/chat/completions, /v1/models, /healthz
internal/adapters/anthropic/    outbound: Anthropic Messages API — text, web search, citations
internal/adapters/openaicompat/ outbound: vLLM + Ollama, failover, tool-call transform
internal/adapters/prometheus/   outbound: telemetry (outbound.Metrics) + /metrics handler
internal/testdoubles/           scripted fake ChatProvider
```

**Providers speak events, not HTTP.** Every provider turns its upstream into
`domain.Event`s (Start, Text, ToolCall, Finish, Usage) pushed into an
`EventSink`; `httpapi` turns events into OpenAI SSE or JSON. A provider never
writes to the client, so adding one needs no HTTP code. A provider must not emit
anything before it knows it can answer — an error returned before the first
event (e.g. `*domain.UpstreamError`) becomes a clean HTTP status; one after it
becomes an in-stream error chunk.

**Provider order is routing.** `app` sends a request to the first provider
whose `Handles(model)` is true. Anthropic handles only its configured model
list; openaicompat handles everything, so it must be registered last
(`providersFromEnv`, pinned by a test).

The Anthropic provider always streams upstream, even for a non-streaming
client. It offers the model Anthropic's server-side web search
(`LLMUX_WEB_SEARCH_MAX_USES`, default 3 per request; 0 turns it off) and
relays text plus each cited source once, as an OpenAI `url_citation`
annotation, which Open WebUI shows as a source chip. Thinking and
server-tool blocks are dropped, and **nothing it emits is ever a tool call**
— Open WebUI executes tool calls it receives. A turn that pauses mid-search
(`pause_turn`) is resumed with its blocks sent back verbatim, up to 3 times,
into the same response. It always drops `temperature`/`top_p` (the Claude 5 family 400s
on either) and rejects client tools, tool history and images with a 400
rather than silently dropping them. Exception: `LLMUX_CLIENT_TOOLS=drop` (default `reject`) ignores client tools and removes tool traffic from the history instead — **required for Open WebUI**, which sends its built-in tools (time, memory, knowledge, chat search) on every browser chat. Those tools then do nothing until llmux forwards tools properly. Its HTTP client must never follow
redirects (`anthropic.HTTPClient`): `x-api-key` survives a cross-host redirect.

The vLLM/Ollama provider keeps the original behaviour: requests with tools go to
vLLM, falling back to Ollama through `applyToolCallTransform`; plain chat goes to
Ollama, falling back to vLLM. When tools are present but the model returns empty
content (a Qwen3 failure mode), it strips the tools and retries as plain chat.

Full description: [Architecture Overview](docs/architecture/2026-07-25-overview.md).

## Conventions

- **Hexagonal layout**: dependencies point inward. `domain` imports nothing internal; ports import only `domain`; adapters never import each other. `go-arch-lint check` enforces this in CI.
- **Transform stages are pure** — no I/O, no globals; they take bytes in and return bytes out.
- **Backends are off unless configured.** Anthropic needs `LLMUX_ANTHROPIC_API_KEY`; vLLM/Ollama default to empty (off) since 2026-09-25, when the homelab GPUs were sold. An empty value disables a backend; `envOr` uses `os.LookupEnv`.
- **Conventional Commits** for every commit (`feat:`, `fix:`, `chore:`, `refactor:`, `docs:`, `test:`, `ci:`).
- **Branch names** follow `<type>/<description>`.
- **All changes go through a branch and pull request** — never commit directly to the default branch (`master`); it is protected.

## Invariants

- Every transform stage must round-trip a clean tool-call response unchanged.
- A nil/empty input must not panic any transform stage; failures return original bytes and the original parse error.
- The compiled binary lives at `./llmux`; never committed.
- A provider emits nothing before it knows it can serve the request.
- `GET /healthz` returns 200 even with no providers configured — the planned homelab deployment (homelab#1471) uses it as its readiness probe.

## What NOT to Do

- Do not call upstream LLMs from a transform stage — transforms operate only on already-buffered bytes.
- Do not introduce shared mutable state — request handling is request-scoped.
- Do not commit the binary `llmux` artifact.
- Do not skip race testing — request handlers must be safe under concurrent traffic.

## Domain

Tool-calling models in vLLM/Ollama frequently return malformed JSON tool calls — `<tool_call>` XML wrappers, orphan `<think>` tags, missing terminators. llmux sits between a tool-aware client (Open Interpreter, Claude Code, etc.) and the model server, repairing the response so downstream parsers don't choke.

## Client keys

Every `/v1` request can be required to carry an llmux **client key**, as `Authorization: Bearer <key>` (OpenAI clients) or `x-api-key: <key>` (Anthropic SDKs) — wherever the client would put a provider key. `LLMUX_CLIENT_KEYS="name=key,name=key"` comes from a secret; the name labels every metric (`client`). No keys → authentication off, every caller `anonymous`. `LLMUX_REQUIRE_CLIENT_KEYS=true` refuses to start without keys (set it in production). Keys are `[A-Za-z0-9_-]`, at least 32 characters — generate with `openssl rand -hex 32`. Either header may carry the key (accepted if either matches); `Bearer` is case-insensitive. Keys are compared as SHA-256 digests in constant time; `/healthz` stays open for probes. Rejections are counted in `llmux_auth_failures_total{reason="missing|invalid"}` and logged at most every 10s with the peer address (behind the gateway, the gateway's) — never the token. `x-api-key` is how Anthropic SDKs will send the key once the native `/v1/messages` lane exists (gateway plan phase 2). Plan: `docs/plans/2026-09-25-llm-gateway.md`.

## Cross-service dependencies

| Service | Endpoint | Purpose |
|---|---|---|
| Anthropic | `LLMUX_ANTHROPIC_URL` (default `https://api.anthropic.com`) | Enabled by `LLMUX_ANTHROPIC_API_KEY`. `LLMUX_WEB_SEARCH_MAX_USES` (default 3; 0 = off) caps searches per request — a searched answer costs ~11–30k input tokens, and offering the tool at all adds ~2.2–2.8k to a request. `LLMUX_WEB_SEARCH_STREAM_ONLY` (default `true`) offers it to streamed requests only: Open WebUI's background calls (titles, tags, follow-ups) are non-streamed and never need it. llmux can't tell who is asking, so **every** non-streaming client loses web search, silently — set it `false` if a non-streaming client needs search. `LLMUX_ANTHROPIC_MODELS` (default `claude-sonnet-5,claude-opus-5,claude-haiku-4-5`) is both the routing list and what `/v1/models` shows. `LLMUX_ANTHROPIC_MAX_TOKENS` (default 8192) applies when a request sets none |
| vLLM | `LLMUX_VLLM_URL` (default empty = off) | Tool-capable local backend; was `http://10.42.2.10:8000` |
| Ollama | `LLMUX_OLLAMA_URL` (default empty = off) | Local-model backend; was `http://10.42.2.10:30068/v1` |

## Quality gate before push

1. `go fmt ./...`
2. `go vet ./...`
3. `go test -race ./...`
4. `go build -o llmux ./cmd/llmux`
5. `go-arch-lint check`

## Documentation

`docs/` taxonomy: `architecture/` · `design/` · `operations/` · `plans/` · `reference/` · `research/`. See each folder's `README.md` for scope. Index: `docs/README.md`.

Start here: [Architecture Overview](docs/architecture/2026-07-25-overview.md) — components, request flow (content-based routing + failover), and the tool-call transform pipeline.

## Observability

Logs to stderr in slog text format at debug level. `GET /healthz` for readiness.

Prometheus metrics on a **separate listener**, `LLMUX_METRICS_ADDR` (default `:9090`; empty disables) — `GET /metrics` is not on the chat port. Telemetry is an outbound port (`outbound.Metrics`): `app.Service` meters every request through a pass-through sink, so providers only need to emit `domain.Usage`. Never label a metric with an unrouted, client-supplied value. Design: [docs/design/2026-09-25-telemetry.md](docs/design/2026-09-25-telemetry.md).

When you learn a new convention or invariant in this repo, update this file.
