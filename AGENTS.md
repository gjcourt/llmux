# AGENTS.md

> llmux is an OpenAI-compatible HTTP proxy that routes chat requests to model backends. Today's backend is vLLM/Ollama, where it repairs malformed tool calls and strips `<think>` blocks; an Anthropic backend with web search is planned (docs/plans/2026-09-25-hexagonal-anthropic-port.md). — https://github.com/gjcourt/llmux

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
internal/adapters/openaicompat/ outbound: vLLM + Ollama, failover, tool-call transform
internal/testdoubles/           scripted fake ChatProvider
```

**Providers speak events, not HTTP.** Every provider turns its upstream into
`domain.Event`s (Start, Text, ToolCall, Finish, Usage) pushed into an
`EventSink`; `httpapi` turns events into OpenAI SSE or JSON. A provider never
writes to the client, so adding one needs no HTTP code. A provider must not emit
anything before it knows it can answer — an error returned before the first
event (e.g. `*domain.UpstreamError`) becomes a clean HTTP status; one after it
becomes an in-stream error chunk.

The vLLM/Ollama provider keeps the original behaviour: requests with tools go to
vLLM, falling back to Ollama through `applyToolCallTransform`; plain chat goes to
Ollama, falling back to vLLM. When tools are present but the model returns empty
content (a Qwen3 failure mode), it strips the tools and retries as plain chat.

Full description: [Architecture Overview](docs/architecture/2026-07-25-overview.md).

## Conventions

- **Hexagonal layout**: dependencies point inward. `domain` imports nothing internal; ports import only `domain`; adapters never import each other. `go-arch-lint check` enforces this in CI.
- **Transform stages are pure** — no I/O, no globals; they take bytes in and return bytes out.
- **Empty env var disables that backend** — `LLMUX_VLLM_URL=` (set, but empty) turns vLLM off; an *unset* variable takes the default. `envOr` uses `os.LookupEnv`. (Before 2026-09-25 this line was documented but false — the old getter treated empty as unset.)
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

## Cross-service dependencies

| Service | Endpoint | Purpose |
|---|---|---|
| vLLM | `LLMUX_VLLM_URL` (default `http://10.42.2.10:8000`) | Primary inference backend (TrueNAS) |
| Ollama | `LLMUX_OLLAMA_URL` (default `http://10.42.2.10:30068/v1`) | Fallback / local-model backend |

Set either to empty string to disable that backend.

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

Logs to stderr in slog text format at debug level. `GET /healthz` for readiness. No metrics endpoint.

When you learn a new convention or invariant in this repo, update this file.
