---
title: Hexagonal restructure + Anthropic outbound port with web search and sources
status: Draft
created: 2026-09-25
updated: 2026-09-25
updated_by: gjcourt
tags: [architecture, hexagonal, anthropic, web-search, citations, open-webui]
---

# Hexagonal restructure + Anthropic outbound port

## Goal

Make `chat.burntbytes.com` (Open WebUI) able to search the web, with **clickable
sources**, by putting llmux between Open WebUI and Anthropic and adding Anthropic's
native Messages API as a new **outbound port** — Claude decides when to search, and
llmux turns its citations into the source chips Open WebUI already knows how to render.

Along the way, restructure llmux from a single `main.go` into the hexagonal layout the
other Go repos use (golinks, soundbyte), enforced by `go-arch-lint`.

## Why llmux, not LiteLLM or an Open WebUI Pipe

Measured on 2026-09-24/25:

| Path | Claude can search | Sources render | In git |
|---|---|---|---|
| Open WebUI → Anthropic OpenAI-compat layer (today) | **No** — `web_search_options` silently ignored; Claude says it can't search | — | ✅ |
| Open WebUI Pipe function | Yes | Yes (hand-written) | ❌ lives in Open WebUI's DB |
| LiteLLM 1.102.1, `web_search_options` forced in config | Yes — tested, incl. streaming + 2nd turn | **No** — citations land in `provider_specific_fields` | ✅ |
| **llmux + Anthropic port (this plan)** | Yes | **Yes** | ✅ |

**LiteLLM also has a latent hazard this plan avoids.** It reports Anthropic's
server-side search as OpenAI `tool_calls` (`name: web_search`, id `srvtoolu_…`), in
both streaming and non-streaming responses. Open WebUI v0.11.3 collects streamed
`delta.tool_calls` to execute them (`utils/middleware.py` ~L5024, immediately after the
annotation handler). The API path tested clean, but the browser path runs Open WebUI's
native tool handling and was not tested. llmux must never emit server tool use as
`tool_calls`.

## What Open WebUI renders (the contract we emit)

`open_webui/utils/middleware.py` (v0.11.3), streaming path:

```python
annotations = delta.get('annotations')
for annotation in annotations:
    if annotation.get('type') == 'url_citation' and 'url_citation' in annotation:
        url_citation = annotation['url_citation']
        url   = url_citation.get('url', '')
        title = url_citation.get('title', url)
        await event_emitter({'type': 'source', ...})
```

So a source chip is exactly:

```json
{"choices":[{"index":0,"delta":{"annotations":[
  {"type":"url_citation","url_citation":{"url":"https://…","title":"…"}}
]}}]}
```

Non-streaming responses carry the same list on `choices[0].message.annotations`
(~L4887 reads a `url_citation` part there too).

## What Anthropic sends (captured, not from docs)

A real `claude-sonnet-5` stream with `web_search_20250305` enabled, 2026-09-25:

```
content blocks: thinking, server_tool_use, web_search_tool_result,
                thinking, server_tool_use, web_search_tool_result, text ×5
delta types:    text_delta 38, input_json_delta 19, citations_delta 2,
                thinking_delta 2, signature_delta 2
usage:          input_tokens 29,268   output_tokens 396   (one search-backed answer)
```

`citations_delta.citation` =
`{type: web_search_result_location, url, title, cited_text, encrypted_index}`.

## Translation table (Anthropic stream → OpenAI stream)

| Anthropic event | llmux emits | Notes |
|---|---|---|
| `message_start` | first chunk: `delta.role = "assistant"` | carries `id`, `model` |
| `content_block_delta` / `text_delta` | `delta.content` | |
| `content_block_delta` / `citations_delta` | `delta.annotations[url_citation]` | **dedupe by URL** per response; `title` falls back to `url` |
| `server_tool_use` (+ its `input_json_delta`s) | **nothing** | never `tool_calls` — see hazard above |
| `web_search_tool_result` | **nothing** | results reach the client only via citations |
| `thinking` / `thinking_delta` / `signature_delta` | nothing (default) — see decision 2 | |
| `message_delta.stop_reason` | `finish_reason`: `end_turn`→`stop`, `max_tokens`→`length`, `pause_turn`→ continue (below) | |
| `message_delta.usage` | final chunk `usage` (when `stream_options.include_usage`) | include `server_tool_use.web_search_requests` in logs |
| `error` event | SSE `error` chunk, then close | never a half-open stream |
| `ping` | nothing | |

`pause_turn`: Anthropic can pause a long server-tool turn; the adapter re-sends the
conversation plus the partial assistant content and keeps streaming into the same
client response, bounded by a retry cap.

Request direction (OpenAI → Anthropic): `system` messages hoisted into `system`;
`max_tokens` defaulted (Anthropic requires it); `stream_options` handled locally;
unknown fields dropped.

> **Revised 2026-09-25 after probing the live API (phase 2):**
> - `temperature` and `top_p` are **always dropped**, not clamped. The whole
>   Claude 5 family (`claude-sonnet-5`, `claude-opus-5`, `claude-fable-5-1`)
>   rejects either one with HTTP 400 "deprecated for this model"; only
>   `claude-haiku-4-5` accepts them.
> - Consecutive same-role messages are **not merged**: the API accepts them,
>   as it does a conversation that starts with an assistant turn.
> - Whitespace-only stop sequences are filtered (they 400); an empty user
>   message is rejected with a 400 (it would 400 upstream anyway).
> - Images and other non-text parts are rejected with a 400 rather than
>   dropped, for the same reason tools are.
> - `message_delta`'s usage is cumulative and supersedes `message_start`'s:
>   after two searches, input was 2,822 at start and 29,268 at the end.

## Target layout

Mirrors golinks / soundbyte:

```
cmd/llmux/                     main: config, wiring, HTTP server
internal/domain/               ChatRequest, Message, Event{Text|Citation|Done|Usage|Error},
                               Citation{URL,Title}, Model — stdlib only
internal/ports/inbound/        ChatService, ModelService
internal/ports/outbound/       ChatProvider { Stream(ctx, ChatRequest) (<-chan Event, error);
                                              Models(ctx) ([]Model, error) }
internal/app/                  router: model id → provider; citation de-dupe;
                               request validation
internal/adapters/httpapi/     inbound: OpenAI-compatible /v1/chat/completions + /v1/models;
                               domain Events → OpenAI SSE / JSON
internal/adapters/anthropic/   outbound: native Messages API + web_search (+ web_fetch);
                               SSE parser; the translation table above
internal/adapters/openaicompat/ outbound: the existing vLLM/Ollama path — see decision 1
internal/testdoubles/          fake ChatProvider
.go-arch-lint.yml              copied from golinks, same component map
```

The existing pure transform stages (`applyToolCallTransform`, `<think>` stripping, JSON
repair) move into `openaicompat` unchanged; their invariants and tests carry over.

## Configuration

| Env | Default | Purpose |
|---|---|---|
| `LLMUX_ANTHROPIC_API_KEY` | — | empty disables the Anthropic port (existing "empty disables" convention) |
| `LLMUX_ANTHROPIC_MODELS` | `claude-sonnet-5,claude-opus-5,claude-haiku-4-5` | ids exposed on `/v1/models` and routed to Anthropic |
| `LLMUX_WEB_SEARCH_MAX_USES` | `3` | per turn; the main cost lever (29k input tokens for one searched answer) |
| `LLMUX_WEB_FETCH` | `false` | also enable `web_fetch` |
| `LLMUX_VLLM_URL`, `LLMUX_OLLAMA_URL` | *(empty)* | existing; defaults change to empty — both backends are gone |

## Testing

- **Golden-stream tests**: the captured Anthropic SSE (sanitised) replayed through an
  `httptest` server → assert the exact OpenAI chunks, including one annotation per
  unique URL and **zero** `tool_calls`.
- Unit tests per translation row; `error` mid-stream; `pause_turn`; client disconnect
  cancels the upstream request (no goroutine leak — `-race` + `goleak`).
- `go-arch-lint check` in CI.
- **End to end, in a browser, on staging** — the path neither prior option was tested
  on: a searched question shows source chips; a follow-up in the same chat works; a
  plain question does not search.

## Deployment (homelab, after llmux merges)

- New `llmux` app, `-prod`/`-stage` overlays, date-sha image from GHCR.
- Anthropic key moves from `openwebui-anthropic` to llmux's own SOPS secret;
  Open WebUI's `OPENAI_API_BASE_URL` → `http://llmux.llmux-prod.svc:8080/v1`.
- Netpols: openwebui → llmux :8080; llmux → `world` :443 (+ DNS).
- **Staging first.** Staging currently has no API key by design; it would get its own
  key via llmux-stage, so the "no funded key in preview" rule needs a conscious exception
  or a low-spend key.

## Open decisions

1. **Keep the vLLM/Ollama adapter?** Both backends left with the GPUs. Keeping it costs
   a port of ~400 lines plus tests nobody can run against a real server; deleting it
   loses the transform work. *Recommendation:* keep it as an adapter (cheap once
   hexagonal, and "add a new outbound port" implies the old one stays), defaults empty.
2. **Thinking blocks.** Drop (default), or map to Open WebUI's reasoning display.
   *Recommendation:* drop in v1.
3. **Client-supplied tools** (Open WebUI's own tools / function calling). Translating
   OpenAI `tools` ↔ Anthropic `tool_use` is its own chunk of work. *Recommendation:*
   out of scope for v1; if a request carries `tools`, return a clear 400 rather than
   silently dropping them.
4. **Cost guard.** `LLMUX_WEB_SEARCH_MAX_USES=3` bounds search per turn; there is no
   per-day cap. Worth one before family use scales.

## Phases / PRs

1. **Restructure only** — hexagonal layout, arch-lint, existing behaviour and tests
   unchanged. Pure refactor; reviewable on its own.
2. **Anthropic adapter** — plain chat, streaming + non-streaming, no tools.
3. **Web search + citations** — the translation table, golden-stream tests.
4. **Homelab deploy** — staging, browser test, then production cutover.
