# llmux

llmux is an OpenAI-compatible HTTP proxy that routes `/v1/chat/completions`
requests to model backends: Anthropic's native Messages API (with
server-side web search and cited sources) and vLLM/Ollama, where it repairs
malformed tool calls and strips `<think>` blocks that local models leave in
their output.

## Backends

Routing is by model name: the first backend whose model list matches the
request's `model` field handles it.

- **Anthropic** — calls the native `/v1/messages` API (not the
  OpenAI-compatible endpoint, which ignores `web_search_options`). Enabled by
  `LLMUX_ANTHROPIC_API_KEY`. Offers the model server-side web search and
  relays cited sources as OpenAI `url_citation` annotations. Rejects client
  tools, tool history and images with a 400 by default, since it can't
  faithfully forward them; `LLMUX_CLIENT_TOOLS=drop` instead drops them
  silently, which is what Open WebUI needs since it sends its built-in tools
  on every chat.
- **vLLM / Ollama** — an OpenAI-compatible catch-all for everything else.
  Requests with tools go to vLLM, falling back to Ollama and repairing its
  response (`<tool_call>` XML, orphaned `<think>` tags, malformed JSON);
  plain chat goes to Ollama, falling back to vLLM. Off by default
  (`LLMUX_VLLM_URL` / `LLMUX_OLLAMA_URL` empty).

At least one backend must be configured or every chat request 404s.

## Configuration

All configuration is environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `LLMUX_ADDR` | `:8080` | Listen address for the chat API. |
| `LLMUX_ANTHROPIC_API_KEY` | unset | Enables the Anthropic backend. |
| `LLMUX_ANTHROPIC_URL` | `https://api.anthropic.com` | Anthropic API base URL. |
| `LLMUX_ANTHROPIC_MODELS` | `claude-sonnet-5,claude-opus-5,claude-haiku-4-5` | Models routed to Anthropic; also what `/v1/models` lists for it. |
| `LLMUX_ANTHROPIC_MAX_TOKENS` | `8192` | `max_tokens` sent upstream when the request specifies none. |
| `LLMUX_WEB_SEARCH_MAX_USES` | `3` | Server-side web searches offered per request; `0` disables the tool. |
| `LLMUX_WEB_SEARCH_STREAM_ONLY` | `true` | Offer web search to streamed requests only, so non-streamed background calls (Open WebUI's title/tag generation) don't pay its per-request token cost. |
| `LLMUX_CLIENT_TOOLS` | `reject` | `reject` 400s a request carrying client tools, tool history or images; `drop` strips them instead. |
| `LLMUX_VLLM_URL` | empty (off) | vLLM base URL. |
| `LLMUX_OLLAMA_URL` | empty (off) | Ollama base URL. |
| `LLMUX_CLIENT_KEYS` | empty (off) | `name=key,name=key` — llmux-level client keys. Unset means every caller is `anonymous` and unauthenticated. |
| `LLMUX_REQUIRE_CLIENT_KEYS` | `false` | Refuse to start if `LLMUX_CLIENT_KEYS` is empty. |
| `LLMUX_METRICS_ADDR` | `:9090` | Listen address for Prometheus `/metrics`; empty disables it. |

A client key goes in `Authorization: Bearer <key>` or `x-api-key: <key>`,
whichever header the client already sends its provider key in. Keys must be
at least 32 characters from `[A-Za-z0-9_-]` — generate one with
`openssl rand -hex 32`.

## Running it

```
go build -o llmux ./cmd/llmux
LLMUX_ANTHROPIC_API_KEY=sk-ant-... ./llmux
```

or with Docker:

```
docker build -t llmux .
docker run -p 8080:8080 -e LLMUX_ANTHROPIC_API_KEY=sk-ant-... llmux
```

`GET /healthz` returns 200 as soon as the process is up, even with no
backend configured, so it works as a readiness probe. The chat API is
`POST /v1/chat/completions` and `GET /v1/models`; metrics are on the
separate listener above, not the chat port.

## How it works

Backends speak `domain.Event`s, not HTTP: each one turns its upstream
response into a stream of Start/Text/ToolCall/Finish/Usage events pushed
into a sink, and the HTTP layer turns those events into OpenAI SSE or JSON.
A backend never writes to the client directly, so adding one needs no HTTP
code, and it must not emit anything before it knows it can serve the
request — an error returned before the first event becomes a clean HTTP
status, one after it becomes an in-stream error chunk.

```
cmd/llmux/                      config + wiring
internal/domain/                ChatRequest, Event, EventSink, errors — stdlib only
internal/ports/inbound/         ChatService
internal/ports/outbound/        ChatProvider, Metrics
internal/app/                   routes a request to the first provider that Handles(model); meters it
internal/adapters/httpapi/      inbound: /v1/chat/completions, /v1/models, /healthz
internal/adapters/anthropic/    outbound: Anthropic Messages API
internal/adapters/openaicompat/ outbound: vLLM + Ollama, failover, tool-call repair
internal/adapters/prometheus/   outbound: metrics, served on its own listener
internal/testdoubles/           scripted fake ChatProvider for tests
```

The layout is hexagonal and enforced by `go-arch-lint`
(`.go-arch-lint.yml`): `domain` imports nothing internal, ports import only
`domain`, and adapters never import each other. See
[docs/architecture/2026-07-25-overview.md](docs/architecture/2026-07-25-overview.md)
for the full request flow.

## Development

```
go build -o llmux ./cmd/llmux   # compile
go test -race ./...             # tests, with the race detector
go vet ./...                    # static analysis
go fmt ./...                    # gofmt
go-arch-lint check              # hexagonal dependency rule
```

CI (`.github/workflows/ci.yml`) runs all of the above plus `golangci-lint`
and a `go mod tidy` check on every push and pull request to `master`.
`.github/workflows/image.yml` builds and, on `master`, pushes a multi-arch
image to `ghcr.io/gjcourt/llmux`.

## Observability

Logs go to stderr as `slog` text at debug level. Prometheus metrics —
request counts and outcomes, in-flight requests, latency, time to first
token, tokens by type, web searches, citations, auth failures — are served
on `LLMUX_METRICS_ADDR`, labelled by client, provider and model. See
[docs/design/2026-09-25-telemetry.md](docs/design/2026-09-25-telemetry.md).
