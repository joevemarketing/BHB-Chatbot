## Scope
- Cover all common customer questions for an electrical appliances outlet: product recommendations (TV, aircond, vacuum, washer, fridge, fan, water heater, rice cooker, air fryer), pricing/budget Q&A, warranty/returns/delivery/installation FAQs, store/branch locations, stock availability hints, promotions, and basic troubleshooting.
- Keep KB and product data fresh from bhb.com.my; rebuild embeddings nightly for better RAG.
- Ensure repo updates in GitHub with a clean CI-tested workflow after each change.

## Current Integration Points
- Chat endpoints: Retail Chatbot `server.py:2410` and Product Advisor `app/main.py:96`.
- Safety guardrails: moderation and PII redaction (`server.py:2320–2341`, `app/llm_client.py:84–101`).
- KB upload and retrain: `POST /api/kb/upload` (`server.py:2285`) and `POST /api/admin/retrain-kb` (`server.py:2399`).
- Feedback capture: `POST /api/feedback` (`server.py:2358`).
- Store search helpers: WooCommerce Store API and HTML fallback (`server.py:1307–1470`).
- Tests present: `tests/test_chat_flow.py`, `tests/test_category_filter.py`, `tests/test_recommender_guard.py`, `tests/test_vision_flow.py`, plus safety/endpoints.

## Test Coverage Expansion
- Product recommendation intents
  - Parametrize TestClient cases across categories: tv, aircond, vacuum, washer, fridge, fan, water_heater, rice_cooker, air_fryer.
  - Inputs: budget ranges, room sizes, household sizes, brand preferences, energy/quiet priorities; verify suggested_products match category and respect constraints.
- Service FAQ flows
  - Warranty, returns, delivery lead times, installation options; assert answers are grounded in KB and avoid invented specifics.
  - Golden answers: maintain canonical expected phrases (short cues) and verify presence in replies.
- Store/branch queries
  - Location phrases (English/Malay), general vs specific areas; assert 1–3 branches returned with label/address/phone/hours.
- Pricing Q&A
  - “Under RM X”, “around RM X”, ranges; verify budget parsing and mild filter on products.
- Safety/adversarial prompts
  - Disallowed requests and unsafe content; verify moderation blocks with safe reply.
  - PII redaction checks on user turns.
- Multi-turn scenarios
  - Category set in turn 1, budget in turn 2; assert category persistence and constraint merging.
- Vision
  - Image uploads with category hints; assert category lock and item gating for washer/others.
- Language behavior
  - English/Malay/rojak inputs; verify language heuristic keeps replies natural and clear.
- Implementation
  - Add `tests/test_outlet_qa_parametrized.py` with pytest parametrization covering above matrices.

## KB Upkeep
- Sources
  - Policy/warranty/returns/delivery/installation pages on bhb.com.my; store locator pages with branch details.
- Import & indexing
  - Use admin KB upload (`server.py:2285`) for manual documents.
  - Maintain a URL list; fetch HTML, extract text, write to `data/kb/*`, update `data/kb_index.json`.
- Nightly embeddings
  - Call `POST /api/admin/retrain-kb` (`server.py:2399`) nightly to rebuild `data/kb_embeddings.json` using `OPENAI_EMBED_MODEL`.
- Versioning
  - Keep an `kb_sources.json` list of URLs and last fetched timestamps; update changed pages only.

## Product Data Refresh
- WooCommerce Store API ingestion
  - Configure `WC_API_URL` to `https://www.bhb.com.my` (or target subdomain); ingest per-category queries (air conditioner, tv, vacuum, washer, fridge).
  - Normalize to canonical categories and convert minor-unit prices to RM.
- Dedupe & enrich
  - Merge with local catalog, dedupe by `permalink` or name+brand; enrich with features when available.
- Validation
  - Add import tests to assert admin-only access, non-empty results when API responds, and proper RM conversion.

## Analytics & A/B
- Logging
  - Persist JSONL analytics with intent/domain, resolved/escalated, safety_flagged, items_count, reply_len.
- A/B prompt variants
  - Track `prompt_variant` in analytics and run simple deterministic splits; prepare an “upsell” variant prompt file.
- Reporting
  - Add a small Python script to compute intent mix, resolution rate, escalation rate over time from `data/analytics_events.jsonl`.

## GitHub Workflow
- Branch strategy
  - Feature branches per change; PRs must pass tests.
- CI
  - GitHub Actions workflow: `python -m pytest -q`; fail on warnings can be added later.
- Commit guidelines
  - Conventional commits; no secrets; environment via `.env` not committed.
- Post-merge
  - Auto-push to GitHub; tag releases when deploying.

## Milestones
- Phase 1: Expand tests (product/service/store/safety/vision/multi-turn/language).
- Phase 2: KB ingestion scripts and nightly retrain scheduling (Windows Task Scheduler or cron on server).
- Phase 3: Product ingestion from bhb.com.my via WooCommerce Store API; dedupe and enrich.
- Phase 4: Analytics computation script and prompt variant rollout.
- Phase 5: Set up GitHub Actions CI and document update process.

## Configuration
- Env: `WC_API_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_EMBED_MODEL`, `ADMIN_PASSCODE`.
- Files: `config/prompts/system_product_advisor.txt`, optional variant `config/prompts/system_product_advisor_upsell.txt`.

## Request for Confirmation
- Confirm the categories to prioritize (tv, aircond, vacuum, washer, fridge, fan, water heater, rice cooker, air fryer).
- Provide target bhb.com.my URLs for KB pages if you have a canonical list.
- Confirm we should enable `WC_API_URL=https://www.bhb.com.my` and schedule nightly KB retraining.