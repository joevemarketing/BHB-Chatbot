## Goals
- Make the chatbot answer brand-specific aircon queries intelligently (Daikin), think before answering, and append relevant policy notes.
- Keep all existing tests green and add targeted tests for the new behavior.
- Commit and push changes to `github.com/joevemarketing/BHB-Chatbot`.

## Scope of Changes
1. Shopping and brand intent
- Recognize Daikin as a brand in brand intent detection.
- Filter candidates by requested brand while respecting category intent (e.g., air conditioners) and exclude non-aircon items (e.g., air purifiers).
- Provide a helpful sizing prompt (room size → HP guidance) when the user asks for aircon.

2. Response quality and policy notes
- Append concise return policy and extended warranty information when the user message mentions them.
- Prefer clean product formatting and avoid irrelevant technical fields.

3. Validation
- Ensure all existing tests pass.
- Add new tests that verify:
  - Brand recognition responds correctly to “Do u have Daikin aircon?”
  - Aircon recommendations exclude "air purifier" results.
  - Policy notes are appended when user mentions return/warranty.

## Implementation Outline
### Code Updates (server-side)
- `server.py`
  - Update `extract_brand_intent` to include "daikin" in known brands.
  - In `top_products_for_message`, `handle_shopping_assistant`, and `handle_shopping_assistant_payload`:
    - Apply brand intent filters to candidate products.
    - Exclude non-aircon items (e.g., purifiers/humidifiers) when intent is air conditioner.
    - Add sizing prompt for aircon queries (ask for room size and budget to right-size HP).
  - When user mentions return or warranty, append concise notes to the reply.

### Tests
- Create/extend tests under `tests/` (without changing current assertions):
  - `test_brand_intent_daikin_aircon_ack`: ensures the reply acknowledges Daikin and provides sizing prompt.
  - `test_aircon_excludes_purifier`: ensures air purifier items are excluded from aircon recommendations.
  - `test_policy_append_when_mentioned`: ensures return/warranty notes are appended when the user mentions them.

### Dependencies & Config
- Confirm `rapidfuzz`, `requests`, `beautifulsoup4` are available (used for fuzzy matching and store search).
- Keep external store calls gated by `WC_API_URL` to avoid unintended network hits without config.

## Repository Update Steps
1. Install Git (if missing) and configure:
- `winget install Git.Git`
- `git config --global user.name "Your Name"`
- `git config --global user.email "you@example.com"`

2. Initialize and branch:
- `git init` (if not a repo)
- `git remote add origin https://github.com/joevemarketing/BHB-Chatbot.git` (if not set)
- `git checkout -b feature/smart-assistant-daikin-ac`

3. Stage and commit:
- `git add server.py tests/* requirements.txt`
- `git commit -m "Smart assistant: Daikin brand detection, aircon intent filtering, policy notes, sizing prompt"`

4. Push and open PR:
- `git push -u origin feature/smart-assistant-daikin-ac`
- Open a pull request against `master` with a short summary and test results.

## Rollout
- Restart the app with auto-reload: `uvicorn server:app --reload --host 0.0.0.0 --port 8000`.
- Smoke test the flow:
  - “Do u have Daikin aircon?” → Daikin acknowledgment + sizing prompt + relevant options.
  - Mentions of “return policy” or “extended warranty” → notes appended.

## Backout Plan
- Revert the branch: `git reset --hard <last_good_commit>` or close the PR.
- Keep changes isolated in the feature branch until approved.

## Confirmation
- If this plan looks good, I’ll proceed to implement tests (if absent), run the test suite, and perform the commit/push steps to the GitHub repo you provided.