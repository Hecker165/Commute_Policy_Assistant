# Commute Policy Assistant

An AI assistant for the MoveInSync employee-transport platform that answers employee questions **only from the employee's own company's current transport policy** — with checkable citations, an explicit "not covered" reply when the policy is silent, hard tenant isolation, and graceful failure under time pressure.

All data is fictional and local: three made-up companies (`acme`, `bluebird`, `copper`) with deliberately contradictory rules, a mock policy store, and a local LLM. No paid APIs, no keys in the repo.

---

## Run it — one command

**No Docker needed (this is the path used to produce every number in this README):**

```text
Windows:  start.bat
macOS/Linux:  ./start.sh
```

The script bootstraps everything inside the project folder (nothing is written to C:):

1. creates `.venv` and installs pinned dependencies
2. downloads a **portable Ollama** into `.runtime/` and starts it
3. pulls the local model (`llama3.2:3b`, ~2 GB) — one time
4. starts the API on http://127.0.0.1:8000 and seeds the three policies

First-run downloads: model ~2 GB + embedding model ~80 MB (cached under `data/model_cache`). Subsequent starts take seconds. If the LLM backend cannot be prepared, the service still starts in a deterministic `extractive` mode (answers are quote-based; everything else identical).

**With Docker instead** (fully containerized; downloads the same model into a named volume):

```text
docker compose up --build -d
docker compose logs -f api     # wait for "service ready"
```

> The measured results in this README were produced with the local (non-Docker) path on the author's machine. The compose file is provided for container-based environments and validated with `docker compose config`.

### 30-second demo

```bash
# 1. issue a demo token for one company (CLI only - there is no public token API)
python -m app.cli issue-token --company acme --role employee

# 2. ask a question
curl -s -X POST http://127.0.0.1:8000/v1/questions \
  -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" \
  -d '{"question": "Do women employees get a security escort for a 9:30 PM drop?"}'
```

```json
{
  "request_id": "49d70bd849c0",
  "covered": true,
  "answer": "The policy states: 'For women employees, a security escort is mandatory on any drop that begins after 8:00 PM.' Since your drop is at 9:30 PM, an escort is mandatory.",
  "citations": [
    {"document": "Commute and Transport Policy", "version": "2026.1",
     "section": "4. Night Drops and Security Escorts", "chunk_id": "acme:2026.1:s4:0"}
  ]
}
```

The same question with a `bluebird` token answers with *its* 10:00 PM rule and a bluebird citation — cross-company answers are structurally impossible (see "Tenant isolation").

More examples:

```bash
# policy is silent -> plain "not covered" message, no citations
-d '{"question": "Can I take my dog in the office cab?"}'

# policy update (admin token): acme 2026.1 -> 2026.2 (escort threshold moves to 7:30 PM)
python -m app.cli ingest --company acme --version 2026.2 \
  --file data/policies/updates/acme-2026.2.md --effective-at 2026-04-01
# while the server is running, the CLI proxies through its admin API (single writer);
# with no server up it runs the same ingest service directly.
# after this, ONLY the new version answers; the old rule is dead

# admin: check what is active
curl -s http://127.0.0.1:8000/v1/admin/policies/active -H "Authorization: Bearer <ADMIN_TOKEN>"
```

API docs (OpenAPI UI): http://127.0.0.1:8000/docs

## Run the checks

```bash
# full offline test suite (no network, no model, no secrets needed)
python -m pytest tests/

# evaluation: deterministic CI mode (extractive LLM, no model download)
python -m app.eval --mode offline --with-update

# evaluation: live mode (real Ollama model; the submitted results.jsonl comes from this)
python -m app.eval --mode live --with-update

# the measured retrieval comparison (bonus): add --retrieval hybrid
# warm latency benchmark (real LLM included)
python -m app.benchmark
```

`app.eval` writes `eval/results.jsonl` — one line per case with exactly the required fields `id`, `input`, `expected`, `actual`, `pass`, `notes` — plus a metrics summary. It runs against its **own isolated store** (`data/eval-runtime`) because Chroma must not be shared across processes; the API server's store is never touched (the version-update case only flips the eval copy). Exit code is non-zero if any case fails. Committed artifacts: `eval/results.jsonl` (live semantic run), `eval/summary*.json`, `eval/results-offline*.jsonl`, `eval/results-live-hybrid.jsonl`.

---

## Design

### Architecture in one paragraph

FastAPI receives a JWT whose verified claims carry `company_id` + `role`. The question pipeline (plain Python, no LLM) resolves the company's **active policy version** from a SQLite catalog, retrieves chunks from Chroma with a **mandatory** `company_id + version_id` metadata filter, and hands at most **3** evidence chunks to the LLM with a prompt that labels the excerpts as untrusted data. The model's structured output is validated server-side: citations are matched against the retrieved allowlist and recovered by verbatim text match against trusted chunk metadata; fabricated ids, cross-company mentions, and malformed output trigger at most one repair call and otherwise fail safely. The user gets either a cited answer, a fixed "not covered" message pointing to the transport desk, or a 503 "try again" — never a hang, a half answer, or another company's rule.

### Key decisions and why

1. **Tenant isolation is structural, not prompt-based.** There is no code path that queries Chroma without both `company_id` and `version_id` filters (`app/vectordb.py::_where`); the company comes only from the verified JWT, the request body forbids a `company_id` field (422), and a test seeds a highly-matching foreign chunk and proves it can never surface. *Trade-off:* the filter API is inflexible by design — you cannot opt out, which is the point.

2. **SQLite is the authority for "which version is active"; Chroma only stores vectors.** Retrieval always reads the active pointer from SQLite first. *Trade-off:* two stores to keep in sync, bought for a single obvious place where "the old policy is dead" is enforced.

3. **Atomic ingestion: parse → validate → stage (inactive) → verify vector count → activate in one transaction → clean up the old version's vectors.** A partial unique index (`status='active'` per company at most once) makes concurrent double-activation impossible at the database level. Any failure before activation leaves the previous policy answering; the failed version is marked and its orphan vectors are deleted. Re-ingesting identical content is idempotent (stable chunk ids + upserts); the same version id with different content is rejected. *Trade-off:* the verify step adds ~10 ms; caught bugs are cheaper than half-updated policies.

4. **The LLM is untrusted.** Authorization, tenant, active version, citation metadata and the answerability fallback are plain code. The prompt labels excerpts as untrusted data and forbids mentioning other companies; server-side guards enforce what the prompt only requests: (a) claimed evidence ids are checked against the retrieved allowlist; (b) if ids are missing/fabricated, the citation is **recovered by matching the answer's verbatim text (≥8-word span) against the trusted chunk texts** — the model cannot cite what was not retrieved, even by accident; (c) an answer that names a company other than the authenticated one is rejected and triggers a repair; (d) malformed output gets exactly one repair call, then a safe 503. *Trade-off:* occasionally a request fails safely instead of answering — by design.

5. **"Not covered" is a first-class answer.** No relevant evidence → the LLM is never called (0 calls) and the user gets a fixed message pointing to the transport desk. The model can also reach the same outcome by judging the evidence irrelevant. *Trade-off:* the threshold (cosine ≥ 0.30) is calibrated so that silence is decided by the LLM, not by a brittle similarity cut (measured top-1 similarity of not-covered questions ranges 0.28–0.37 vs 0.38–0.71 for covered ones — overlapping, so the threshold only gates garbage).

6. **Local, free, and constrained.** Ollama + `llama3.2:3b` with `format: "json"` (grammar-constrained decoding — the model cannot emit invalid JSON). A tolerant parser also accepts the three-line format and raw JSON for other providers. A deterministic **extractive fallback** (`LLM_PROVIDER=extractive`) answers by picking the best-matching policy sentence — used by CI, tests, and as a no-model safety mode; it also screens injection-looking sentences. *Trade-off:* the fallback is visibly weaker (verbatim quotes, no synthesis) but never fabricates.

7. **Benchmarks are honest.** Warm-up requests are excluded from p95; failures are counted and reported; nothing is hard-coded. Measured warm p95 **1.39 s** (semantic) / **1.41 s** (hybrid) on a 12-core CPU — inside the 3 s budget with ~1.6 s of headroom. A request-level deadline (2.7 s) plus per-call LLM timeout (2.2 s) guarantee the "try again" response instead of a hang; the only allowed retry is one repair call within the remaining budget.

8. **Two embedding modes.** Live: Chroma's bundled ONNX MiniLM-L6-v2 (downloaded once). Tests/CI: a deterministic hashed bag-of-n-grams embedding (no download, no network, stable similarity). *Trade-off:* offline retrieval is lexical rather than semantic, so offline scores approximate but never overstate live behavior.

### The two numbers, explained

| Constraint | How it is met |
|---|---|
| p95 ≤ 3 s with the LLM call included | request deadline 2.7 s enforced with `asyncio.wait_for` around a bounded pipeline; LLM call capped at 2.2 s; measured warm p95 **1.39 s** (`docs/benchmark-semantic.json`) |
| ≤ 2 LLM calls per question | 1 normal call + at most 1 repair, only if the budget affords it (`MAX_LLM_CALLS_PER_REQUEST` is a constant, not a setting); no-evidence questions make 0 calls |
| ≤ 3 cited chunks | retrieval fetches 4 candidates, at most 3 reach the prompt, citations are a subset of those 3, rendered from trusted metadata |

---

## Evaluation

`eval/cases.jsonl` is the immutable gold set: **23 cases** (22 static + 1 version-update scenario) covering all three companies: direct escort questions (the contradictory 8/9/10 PM rules), a paraphrase, Saturday rules (including Bluebird's explicit absence), airport approvers, no-show penalties, cost caps, eligibility, a covered-negative ("guests are not covered" is *stated* in the policies), three not-covered topics (pets / lost laptop / wifi — silent in all policies), two cross-company traps, a direct user-injection case, and the version-update scenario (ingest acme 2026.2; the old 8 PM rule must stop answering).

Scoring is deterministic plain code (no LLM judge): covered/not-covered flag, required and forbidden phrases (with spelling variants), citation company/version/section checks against the *authenticated* company and *active* version, plus these run-level metrics:

- **answer correctness** — share of cases passing all checks
- **retrieval recall@3** — expected section among the top-3 retrieved (covered cases)
- **citation validity** — every covered answer has 1–3 citations, all from the authenticated company + active version
- **tenant leakage** — citations/chunks from another company or inactive version; **must be 0**

### Results (all measured on the author's machine; artifacts committed)

| Run | Mode | Retrieval | Cases | Recall@3 | Citation validity | Leaks | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| **submitted** | live (`llama3.2:3b`) | semantic | **23/23** | 1.00 | 1.00 | **0** | 1.31 s | 1.88 s |
| comparison | live | **hybrid (bonus)** | 21/23 | 1.00 | 1.00 | 0 | 1.21 s | 1.80 s |
| CI mode | offline (extractive) | semantic | 23/23 | 1.00 | 1.00 | 0 | 0.26 s | 0.33 s |
| CI mode | offline | hybrid | 22/23 | 1.00 | 1.00 | 0 | — | 0.45 s |

Warm benchmark (21 timed requests after 5 warm-ups, real LLM call included, zero failures): semantic **p50 1008 ms / p95 1386 ms**, hybrid p50 943 ms / p95 1407 ms. Machine: Windows 10.0.26300, 12 CPUs (Intel Coffee Lake), Python 3.11.9, Ollama, `llama3.2:3b`. Full JSON: `docs/benchmark-*.json`. Three consecutive live runs produced byte-identical pass/answer sets (temperature 0 + deterministic retrieval).

Details, method, failure analysis and limitations: [docs/evaluation-report.md](docs/evaluation-report.md).

### Bonus: hybrid retrieval — measured, kept opt-in

Implemented one retrieval improvement: reciprocal-rank fusion of the semantic ranking with a BM25-lite lexical ranking over the same tenant/version-filtered corpus (`RETRIEVAL_MODE=hybrid`). Result at this corpus scale (3 short policies, ~24 chunks per company): recall@3 unchanged at **1.00** in both modes, latency unchanged (~±3 %), but end-to-end correctness **dropped** 23 → 21 because the re-ordered evidence list changes the small model's behavior on two borderline cases (details in the report). Verdict: **keep `semantic` as the default**; hybrid remains available behind one env var for larger corpora where lexical precision on ids/amounts matters. The measured claim is deliberately "no gain here", not a manufactured win.

---

## Repository layout

```text
app/            FastAPI app, auth, catalog (SQLite), policies (parse/chunk/ingest),
                vectordb (Chroma), retrieval, answering, llm providers, seed, cli, eval, benchmark
data/policies/  the three fictional policies (+ updates/acme-2026.2.md used in demos/tests)
eval/           cases.jsonl (gold, immutable) + committed results/summaries
tests/          65 offline tests (auth, ingestion atomicity, isolation, grounding, budgets, API, eval)
docs/           evaluation report, benchmark JSONs
start.bat / start.sh / Dockerfile / compose.yaml
```

## Security notes

- JWT: HS256 with required claims, strict algorithm allowlist; a random signing secret is generated once at `DATA_DIR/jwt_secret` (git-ignored) or supplied via `JWT_SECRET` for tests. There is no public token-issuing endpoint — the CLI is for local demos.
- Nothing sensitive is logged (request id, company, status, timings only — never question text, tokens, or the secret).
- All policy content, questions, and model output are treated as untrusted input; nothing from them is ever executed.
- First-run artifacts (`data/`, `.runtime/`, `.venv/`) are git-ignored; the committed `eval/results.jsonl` is generated evidence from the submitted run.

## Known limitations

See [docs/evaluation-report.md](docs/evaluation-report.md) for the full list — headline items: the 3B local model occasionally flips borderline answerability judgments (observed and fixed during development; residual risk documented), the first request after a long idle pays a model cold-start (~10–19 s → one 503 "try again", mitigated by a startup warm-up), and Chroma must not be shared across processes (eval therefore uses an isolated store).
