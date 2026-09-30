<!-- readme-type: service -->
# llmux

OpenAI-compatible HTTP proxy that routes chat requests to Anthropic and vLLM/Ollama backends

Tool-calling models served through vLLM or Ollama often return malformed JSON
tool calls — `<tool_call>` XML wrappers, orphaned `<think>` tags, missing
terminators — that break OpenAI-compatible clients expecting a clean
`tool_calls` array. llmux sits between a client (Open WebUI, Open Interpreter,
Claude Code) and the model servers, repairing the response before it reaches
them. It also offers a native Anthropic backend behind the same OpenAI-shaped
API, with server-side web search and cited sources. Routing is by model name,
so a client picks a backend just by naming a model.

**Status:** in production and staging on the homelab since 2026-09-25
(gjcourt/homelab#1471), proxying Open WebUI's chats to Anthropic; the
vLLM/Ollama backends are configured off since the homelab GPUs were sold.

```text
go build -o llmux ./cmd/llmux
LLMUX_ADDR=127.0.0.1:18080 LLMUX_METRICS_ADDR=127.0.0.1:19090 ./llmux &
time=2026-09-30T06:04:00.011Z level=WARN msg="no model backends configured; every chat request will return 404"
time=2026-09-30T06:04:00.015Z level=INFO msg="llmux listening" addr=127.0.0.1:18080 providers=0

curl http://127.0.0.1:18080/healthz
ok

curl http://127.0.0.1:18080/v1/models
{"data":[],"object":"list"}
```

## Quick start

Needs: Go 1.25+.

```bash
git clone https://github.com/gjcourt/llmux && cd llmux
go build -o llmux ./cmd/llmux
./llmux
```

Then check `curl http://localhost:8080/healthz`, which answers `ok` as soon
as the process is up. Without `LLMUX_ANTHROPIC_API_KEY` (or a vLLM/Ollama
URL) set, llmux still starts but every chat request 404s — see
Configuration to enable a backend.

## Usage

Send a chat request the same way you would to OpenAI; llmux routes it by
model name to whichever backend is configured.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "hi"}]}'
```

With no backend configured this 404s (`no provider serves this model`); set
`LLMUX_ANTHROPIC_API_KEY` and it answers in the same shape OpenAI does.

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

## How it works

Backends speak `domain.Event`s, not HTTP: each one turns its upstream
response into a stream of Start/Text/ToolCall/Finish/Usage events pushed
into a sink, and the HTTP layer turns those into OpenAI SSE or JSON — a
backend never writes to the client directly, so adding one needs no HTTP
code. Provider order is routing: the first backend whose `Handles(model)` is
true serves the request, so Anthropic's specific model list is checked
before the vLLM/Ollama catch-all. The layout is hexagonal and enforced by
`go-arch-lint` (`.go-arch-lint.yml`); the same events feed Prometheus metrics
on a separate listener (`LLMUX_METRICS_ADDR`). See
[docs/architecture/2026-07-25-overview.md](docs/architecture/2026-07-25-overview.md)
for the full request flow and
[docs/design/2026-09-25-telemetry.md](docs/design/2026-09-25-telemetry.md)
for telemetry.

## Development

```bash
go build ./... && go build -o llmux ./cmd/llmux   # Build
go-arch-lint check                                # Arch guard
go test -race -count=1 -timeout 120s ./...        # Test
gofmt -l .                                        # Format (must print nothing)
go vet ./...                                      # Vet
golangci-lint run --timeout=5m                    # Lint
go mod tidy                                       # Tidy (must not change go.mod/go.sum)
```

Conventions for contributors and agents: [AGENTS.md](AGENTS.md).

## Deployment

Runs on the homelab as a Kubernetes Deployment behind its own Service and
NetworkPolicy, in front of Open WebUI. Production and staging overlays are
in `gjcourt/homelab`'s
[`apps/base/llmux`](https://github.com/gjcourt/homelab/tree/master/apps/base/llmux)
and
[`apps/production/llmux`](https://github.com/gjcourt/homelab/tree/master/apps/production/llmux);
there is no llmux-specific runbook yet.

## License

No licence file yet.
