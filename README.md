# BHB Retail Chatbot

A FastAPI-based customer-oriented chatbot for electrical appliances franchise stores. Includes:
- Retail chatbot service (`server.py`): chat, RAG, vision search, admin KB
- Product advisor service (`app/main.py`): product recommendations with simple web UI

## Local Development
- Python virtual env:
  - `python -m venv .venv`
  - `./.venv/Scripts/Activate.ps1`
  - `pip install -r requirements.txt`
- Environment variables:
  - `OPENAI_API_KEY` (required for LLM, embeddings, vision)
  - Optional: `WC_API_URL` (e.g., `https://www.bhb.com.my`), `RATE_LIMIT_MAX`, `RATE_LIMIT_WINDOW`
- Run services:
  - Retail: `python -m uvicorn server:app --reload --port 8001`
  - Advisor: `python -m uvicorn app.main:app --reload --port 8000`

## Testing Endpoints
- Retail UI: `http://localhost:8001/` (Admin: `/admin`)
- Advisor UI: `http://localhost:8000/web/`
- Health: `GET /api/health`
- Chat: `POST /api/chat` (retail), `POST /api/chat` (advisor)
- RAG: `POST /api/rag-chat`
- Vision: `POST /api/vision-search`

## Deployment
### Render (Free)
- Uses `render.yaml` to define two services:
  - `retail-chatbot` → `uvicorn server:app`
  - `product-advisor` → `uvicorn app.main:app`
- Add secrets:
  - `openai_api_key` → mapped to `OPENAI_API_KEY`
  - Optional: `wc_api_url`

### Vercel (Frontend)
- Host `web/` on Vercel and proxy API calls to Render using rewrites.

## Security
- Do not commit secrets (`.env` is ignored).
- Use env vars or Render secrets for keys.

## Postman
- `docs/postman_collection.json` includes requests for both services. Update base URLs accordingly.