## Goals & Success Metrics

* Reduce call center workload by 40% while increasing first-contact resolution ≥85%

* Convert ≥20% of consult chats into scheduled jobs; CSAT ≥4.7/5

* Maintain strict safety/compliance messaging; zero unsafe instructions given

## Core User Journeys & Intents

* Emergency power issue triage (partial/total outage, burning smell, tripping breaker)

* Quote/estimate for common services (EV charger, panel upgrade, lighting, rewiring)

* Appointment booking and dispatch (new job, follow-up, warranty)

* Product questions (surge protectors, smart switches) and upsell bundles

* Status updates (technician ETA, job progress, invoicing, warranty claims)

## Conversation Design

* Tone: clear, safety-first, professional, local franchise branding

* Structured intake: address, contact, preferred time, photos/videos, utility provider

* Guardrails: never advise DIY on live circuits; prompt to shut off breaker only if safe; disclaimers on utility-side issues

* Escalation: live agent transfer when high-risk signals (smell of burning, smoke, exposed wires)

* Memory: session-scoped details (property type, panel amperage, utility) with PII scrubbing

## Knowledge & RAG

* Sources: SOPs, service catalog, pricing tiers, warranty policy, local codes/licensing, FAQ

* Indexing: embed documents with OpenAI embeddings; chunking by headings and service

* Store: start with JSON embeddings (existing code), plan for vector DB (e.g., pgvector/Weaviate) when scale requires

* Freshness: nightly rebuild and on-demand admin upload (already supported)

## Service Catalog & Pricing

* Define canonical service taxonomy (diagnostic, panel work, circuits, EV, lighting, smart home)

* Region-specific pricing tiers, travel fees, emergency surcharges

* Risk-based quotes: photo-assisted triage; ranges → firm quotes post-inspection

## Tools/Actions & Integrations

* Booking: integrate calendar/dispatch (ServiceTitan/Housecall Pro/Jobber) for work orders

* CRM: HubSpot/Salesforce/Zoho for contact sync, lead attribution

* Payments: deposit capture and invoicing (Stripe/Square); financing options

* Outage/status: utility API links and local outage maps; detect utility-side vs premise-side

* Ticketing: create/lookup work orders; push notes and attachments (photos)

## Channels

* Web chat widget (existing UI) with photo upload

* WhatsApp and Facebook Messenger for consumer convenience

* SMS fallback for appointment reminders and confirmations

* Phone IVR handoff to human dispatcher for emergencies

## Safety, Compliance & Guardrails

* Safety checklist injected before any troubleshooting beyond visual checks

* Jurisdiction-specific licensing info; display franchise license numbers

* Hard blocks on advising internal panel work to non-licensed users

* Audit trail: store prompts/responses and safety flags for QA

## Data, Privacy & Security

* PII redaction at ingestion; configurable retention (e.g., 30–90 days)

* Role-based admin portal for KB uploads and content moderation

* Secrets via environment; zero logging of keys; encryption at rest for conversations

* Consent capture and privacy notice with opt-out

## Architecture & Implementation Map (fit to current repo)

* Backend FastAPI & RAG: `server.py` (chat, KB upload, embeddings) and `app/*` for product advisor

* LLM providers: OpenAI & Gemini via current clients; keep vision triage enabled

* Frontend: `web/*` and `static/*` chat UIs; extend with booking and CRM flows

* Data: `data/kb_*`, `data/faq.json`, products JSON; continue JSON embeddings; add vector DB later

## End‑to‑End Workflow (orchestration)

1. Greet → detect intent (emergency, quote, booking, product, status)
2. Safety pre-checks if electrical risk; escalate if high-risk
3. RAG grounding: fetch SOP/policy/service pages relevant to intent
4. Optional vision triage: user uploads panel/device photos; extract features
5. Quote range or diagnostic recommendation; capture address/contact
6. Book appointment via dispatch API; create work order and attach notes/photos
7. Payment link or deposit; send confirmations and reminders
8. Post-job follow-up: CSAT survey, warranty info, upsell suggestions
9. Analytics loop: tag outcomes, update KB, refine prompts and tools

## Analytics & Continuous Improvement

* Conversation analytics: intent mix, resolution rate, escalations, safety triggers

* A/B test prompt templates, quote ranges, and upsell scripts

* Feedback loop: frontline staff can flag incorrect answers; retrain KB nightly

## Testing & QA

* Expand FastAPI TestClient suite to cover safety guardrails, booking action, CRM sync, payments

* Golden answer sets for SOPs and pricing Q\&A; adversarial prompts for safety

* Vision tests: panel type classification and risk detection

## Deployment & Operations

* Add Docker + compose for API + vector DB; health checks and autoscaling

* Observability: structured logs, error tracking, conversation traces

* Runbooks for outages, model provider fallbacks (OpenAI↔Gemini)

## Milestones

* Week 1: Catalog taxonomy, SOPs, safety guardrails, web UI polish

* Week 2: Booking + CRM integrations, quote workflows, analytics baseline

* Week 3: Payments, WhatsApp/Messenger, advanced vision triage

* Week 4: Vector DB, CI/CD, full QA, launch and iterate

