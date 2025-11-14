## Local Browser Demo (Fast)
1) Create venv and install deps:
   - `python -m venv .venv`
   - `\.venv\Scripts\Activate.ps1`
   - `pip install fastapi uvicorn openai python-dotenv rapidfuzz requests beautifulsoup4`
2) Set environment vars (PowerShell):
   - `setx OPENAI_API_KEY "<key>"`
   - Optional: `setx WC_API_URL "https://www.bhb.com.my"`, `setx RATE_LIMIT_MAX 60`, `setx RATE_LIMIT_WINDOW 60`
3) Start services:
   - Product Advisor: `python -m uvicorn app.main:app --reload --port 8000`
   - Retail Chatbot: `python -m uvicorn server:app --reload --port 8001`
4) Open demo UIs:
   - Product Advisor UI: `http://localhost:8000/web/`
   - Retail Chatbot UI: `http://localhost:8001/`
5) Quick checks:
   - Product chat: ask “9kg washer under RM2000”
   - Support FAQ: ask “What’s warranty policy?”
   - Store locator: ask “Penang branches”
   - Bundle: “New house bundle: TV, fridge, washer, aircond, fans”

## Shareable URL (No Deploy)
- Use one-click tunnels for client viewing:
  - Ngrok: `ngrok http 8001` → share URL for Retail chatbot
  - Cloudflare Tunnel: `cloudflared tunnel --url http://localhost:8001`
- Optionally expose Product Advisor (port 8000) via a second tunnel

## Cloud Deployment (Quick)
- Render (free tier):
  1) Create two Web Services from repo: `server:app` on port `8001`, `app.main:app` on port `8000`
  2) Set env vars: `OPENAI_API_KEY`, `WC_API_URL`, `RATE_LIMIT_MAX`, `RATE_LIMIT_WINDOW`
  3) Build command: `pip install -r requirements.txt || pip install fastapi uvicorn openai python-dotenv rapidfuzz requests beautifulsoup4`
  4) Start command: `uvicorn server:app --host 0.0.0.0 --port 8001`
- Azure App Service:
  1) Create Python Web App, deploy repo, set Startup Command `python -m uvicorn server:app --host 0.0.0.0 --port 8001`
  2) Configure env vars in App Settings
- Railway/Fly.io are similar (single start command + env)

## Messaging Platform Pilot (Optional)
- Telegram Bot (fastest):
  1) Create bot with BotFather → get token
  2) Small bridge service that forwards Telegram messages to `POST /api/chat` and replies with `reply`
  3) Host bridge on the same Render/Azure app
- WhatsApp via Meta Cloud API or Twilio:
  1) Register phone, set webhook → forward messages to chatbot endpoint → reply returns message content
- Slack App: Socket mode → relay to `/api/chat`

## Demo UX Polish (Browser)
- Product Advisor UI (`web/`):
  - Adjust theme by CSS variables (e.g., `web/styles.css` `--border`, `--primary`) for brand feel
  - Add product card highlights: capacity, energy label, price range
- Retail Chatbot UI (`static/`):
  - Show items list with image, price, CTA link when `WC_API_URL` is configured
  - Add “Escalate to human” button stub (opens mailto/WhatsApp link)

## Pre‑Demo Checklist
- OPENAI configured; health endpoints return ok
- KB content added (admin page) for common support questions
- Catalog present: `data/products.json` and/or `data/bhb_products_real.json`
- Rate limit tuned; test stress without blocking demo
- Tunnels or cloud URL tested from mobile

## Improvements To Prioritize
- Customer workflows:
  - Add appointment/delivery/installation booking stubs + confirmations
  - Human escalation endpoint (`POST /api/escalate`) with ticketing/CRM hook
- Product & retrieval:
  - Normalize catalog schema across app/server; ensure consistent currency handling (RM)
  - Introduce vector DB (Qdrant/FAISS) for KB + product retrieval with metadata filters
- Store locator:
  - Geocoding + distance ranking; store hours formatting; branch maps
- Multilingual & prompts:
  - Robust language detection; configurable prompt templates per store, A/B test variants
- Safety & analytics:
  - Moderation and PII redaction are in place; add structured event logging (intent, conversions, escalations), dashboards for latency and success
- Tests:
  - Pytest suite for chat flows, rate limit, moderation, RAG, store API

## What I’ll Deliver Next
1) A pytest suite in `tests/` to automate health, product filters, chat intents, rate limit, moderation, RAG basics
2) A Postman collection for quick manual testing
3) Optional Telegram bridge for a real messaging demo

Please confirm which hosting option (Render/Azure/Ngrok) you want and whether to include the Telegram pilot. After your confirmation, I’ll implement the tests, collection, and any chosen deployment integration.