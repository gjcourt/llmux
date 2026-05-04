# operations/

Runbooks, smoke tests, and on-call / day-to-day operating procedures.

**Put here:**
- How to run llmux, point it at a backend, and verify a roundtrip.
- Step-by-step procedures for common failure modes (bad backend response, fallback flapping).

**Do not put here:**
- Backend API specs — `reference/`.
- Architecture explanation — `architecture/`.

**Naming convention:** `<yyyy-mm-dd>-<topic>.md`
Examples: `2026-05-02-running-locally.md`.

**Allowed `status:` values:** `Stable`, `Superseded`.

Stale runbooks are dangerous. When a procedure changes, update the doc in the same PR.
