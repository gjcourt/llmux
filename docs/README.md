# llmux Documentation

llmux is an HTTP proxy in front of vLLM and Ollama that repairs malformed tool-call responses. This folder is organized into a fixed six-folder taxonomy. Each folder's `README.md` describes what belongs there.

## Folders

- [`architecture/`](architecture/README.md) — how the system is built today.
- [`design/`](design/README.md) — proposals, RFCs, in-flight or recently shipped designs.
- [`operations/`](operations/README.md) — runbooks, smoke tests, dev setup.
- [`plans/`](plans/README.md) — phased migrations, rollout sequencing.
- [`reference/`](reference/README.md) — backend API quirks, tool-call wire formats, benchmarks.
- [`research/`](research/README.md) — spikes, investigations.

This taxonomy is currently a scaffold — folder READMEs only, with content to be migrated and added as the project grows.

## Conventions

- All non-index docs use frontmatter (`title`, `status`, `created`, `updated`, `updated_by`, `tags`).
- Filenames carry topic and creation date (`<yyyy-mm-dd>-<topic>.md`); state lives in `status:` frontmatter, never in the filename.
- See `AGENTS.md` for the full convention.
