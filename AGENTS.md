# AGENTS.md

> llmux is a small HTTP proxy in front of vLLM and Ollama — it buffers the model response and runs `applyToolCallTransform` to repair malformed tool calls and strip `<think>` blocks before returning to the client. — https://github.com/gjcourt/llmux

## Commands

| Command | Use |
|---------|-----|
| `go build .` | Compile to `./llmux` |
| `go test -race ./...` | Run all tests with race detector |
| `go vet ./...` | Static analysis |
| `go fmt ./...` | gofmt |

Single test: `go test -run TestTransform_SingleToolCall -v`
Pre-push: `go fmt ./... && go vet ./... && go test -race ./...`

## Architecture

Single-package Go tool — everything is in `main.go`.

The HTTP server proxies `POST /v1/chat/completions` and `GET /v1/models` to either vLLM (`LLMUX_VLLM_URL`) or Ollama (`LLMUX_OLLAMA_URL`). On chat completions it forces `stream:false` on the upstream call so the response can be buffered, runs `applyToolCallTransform` to repair malformed tool calls and strip `<think>` blocks, then returns the result. When tools are present but the model returns empty content (a known Qwen3 conversational-query failure mode), it strips the tools and retries as plain chat.

Tests live in `main_test.go`.

## Conventions

- **Single package**: `main`. Helpers and tests live in `main.go` / `main_test.go`.
- **Transform stages are pure** — no I/O, no globals; they take bytes in and return bytes out.
- **Empty env var disables that backend** — `LLMUX_VLLM_URL=` (with nothing) intentionally turns off vLLM routing.
- **Conventional Commits** for every commit (`feat:`, `fix:`, `chore:`, `refactor:`, `docs:`, `test:`, `ci:`).
- **Branch names** follow `<type>/<description>`.
- **All changes go through a branch and pull request** — never commit directly to the default branch (`master`); it is protected.

## Invariants

- Every transform stage must round-trip a clean tool-call response unchanged.
- A nil/empty input must not panic any transform stage; failures return original bytes and the original parse error.
- The compiled binary lives at `./llmux`; never committed.

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
4. `go build .`

## Documentation

`docs/` taxonomy: `architecture/` · `design/` · `operations/` · `plans/` · `reference/` · `research/`. See each folder's `README.md` for scope. Index: `docs/README.md`.

## Observability

Logs to stderr in slog text format at debug level. No metrics endpoint today.

When you learn a new convention or invariant in this repo, update this file.
