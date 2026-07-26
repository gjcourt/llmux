# architecture/

How the system is built **today** — the current shape of the code.

**Current docs:**
- [`2026-07-25-overview.md`](2026-07-25-overview.md) — system overview: components, request flow (content-based routing + connection failover), and the tool-call transform pipeline. *(Stable)*

**Put here:**
- System-overview docs that describe how `applyToolCallTransform`, the proxy layer, and the backends fit together.
- Diagrams and prose that explain the present architecture.

**Do not put here:**
- Proposals for future architecture — `design/`.
- Phased migration sequencing — `plans/`.
- Vendor / API quirks — `reference/`.
- Runbooks — `operations/`.

**Naming convention:** `<yyyy-mm-dd>-<topic>.md`
Examples: `2026-05-02-overview.md`.

**Allowed `status:` values:** `Stable`, `Superseded`.

When the architecture changes materially, supersede the existing doc with a new one and set `superseded_by:` on the old one.
