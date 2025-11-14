## Prerequisites
- Create venv and install deps: `python -m venv .venv`, `\.venv\Scripts\Activate.ps1`, `pip install fastapi uvicorn openai python-dotenv rapidfuzz requests beautifulsoup4 pytest httpx`
- Optional env vars: `OPENAI_API_KEY`, `WC_API_URL`, `RATE_LIMIT_MAX`, `RATE_LIMIT_WINDOW`, `PROMPT_BASE`, `PROMPT_VARIANT`
- Ensure `data/` has `products.json` (or `bhb_products_real.json`) and `data/faq.json`

## Start Services
- Product Advisor: `python -m uvicorn app.main:app --reload --port 8000`
- Retail Chatbot: `python -m uvicorn server:app --reload --port 8001`

## Smoke Tests (Manual)
- Product Advisor UI: `http://localhost:8000/web/`
- Retail Chatbot UI: `http://localhost:8001/`
- Admin KB UI: `http://localhost:8001/admin`
- API checks (PowerShell):
  - Health: `Invoke-WebRequest http://localhost:8000/api/health`
  - Products: `Invoke-WebRequest "http://localhost:8000/api/products?category=washer&max_price=2000"`
  - Product chat: `Invoke-WebRequest -Uri http://localhost:8000/api/chat -Method POST -ContentType 'application/json' -Body '{"messages":[{"role":"user","content":"Looking for a 9kg washer under RM2000"}]}'`
  - Retail chat: `Invoke-WebRequest -Uri http://localhost:8001/api/chat -Method POST -ContentType 'application/json' -Body '{"session_id":"test1","domain":"smart_support","message":"Need a 55\" TV under RM2000"}'`
  - RAG chat: `Invoke-WebRequest -Uri http://localhost:8001/api/rag-chat -Method POST -ContentType 'application/json' -Body '{"session_id":"test2","message":"What is the delivery timeline?"}'`

## Automated Tests (Add Pytest Suite)
- Create `tests/` with:
  - `test_app_health.py`: GET `/api/health` returns `{"status":"ok"}`
  - `test_products_filter.py`: GET `/api/products` filters by `category`, `max_price`
  - `test_chat_product.py`: POST `/api/chat` (app) with washer query returns `reply` and `suggested_products` matching category
  - `test_chat_service.py`: POST `/api/chat` (app) with warranty query returns FAQ-grounded text when `data/faq.json` present
  - `test_server_chat_smart.py`: POST `/api/chat` (server) smart_support returns `reply` and optional `items`
  - `test_rag_chat.py`: seed simple KB entry, POST `/api/rag-chat` returns grounded reply and `sources`
  - `test_rate_limit.py`: rapid loop to trigger 429 with `{"error":"rate_limited"}`
  - `test_moderation_pii.py`: ensure emails/phones in inbound messages are redacted in echoed context (no raw PII in reply)
- Use `pytest` with `httpx.AsyncClient` and `pytest-asyncio` to run against app instantiated with `TestClient`

## Vision Tests
- Add `samples/washer.jpg` placeholder; test `POST /api/vision-search` expects `details` with `category` «washing machine» and non-empty `results` when catalog/store API is reachable
- If `OPENAI_API_KEY` missing, assert 400 diagnostics response explaining provider configuration

## Postman Collection
- Provide `docs/postman_collection.json` with requests for `/api/health`, `/api/products`, `/api/chat`, `/api/rag-chat`, `/api/vision-search`, `/api/admin/login`, `/api/kb/*`
- Include variables for `host`, `OPENAI_API_KEY`, `WC_API_URL`, `session_id`

## Logging & Analytics Verification
- Observe stdout JSON lines: `{"ts":..., "ip":..., "path":"/api/chat", "status":200, "duration_ms":...}`
- Confirm rate limit behavior: after `RATE_LIMIT_MAX` within `RATE_LIMIT_WINDOW`, subsequent requests get 429

## Success Criteria
- All smoke tests succeed; automated tests passing locally
- Product-only queries stay within category; bundle intent groups by category
- FAQ answers grounded, no fabricated policies
- RAG returns KB sources when available; suppresses policy snippets for shopping intent
- PII redaction and moderation safe response verified

## Next Enhancements
- Add WooCommerce integration tests when `WC_API_URL` is set
- Export test reports and a simple README for testing steps
- Optional: seed minimal `data/products.json` for consistent local tests