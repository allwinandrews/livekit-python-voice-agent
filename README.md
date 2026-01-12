````markdown
Live demo: https://livekit-voice-agent-frontend.vercel.app/  
Frontend repository: https://github.com/allwinandrews/livekit-voice-agent-frontend/

<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# LiveKit Voice Agent – Python Backend

This repository contains the **Python backend voice agent** built using **LiveKit Agents** as part of the _Python Developer Test: Build a Voice Agent with LiveKit_.

The project demonstrates:

- A fully working **Stage 1** single-prompt voice agent
- A clear foundation to extend into **Stage 2** structured, state-based conversation flows

The backend is deployed on **LiveKit Cloud** and connected to a live React frontend.

---

## Stage 1 – Basic Voice Agent (Completed)

The agent successfully:

- Joins a LiveKit room
- Listens to user speech via microphone
- Converts speech to text (STT)
- Sends text to an LLM
- Converts the LLM response back to speech (TTS)
- Streams synthesized audio back to the room

### Technology Stack

- **LiveKit Agents (Python SDK)**
- **Speech-to-Text**: AssemblyAI (streaming)
- **LLM**: OpenAI (GPT-4.1-mini)
- **Text-to-Speech**: Cartesia
- **Turn Detection**: LiveKit Multilingual Turn Detector
- **Voice Activity Detection**: Silero VAD
- **Noise Cancellation**: LiveKit Cloud BVC

---

## Live Demo

You can talk to the agent live using the deployed frontend:

**Live Demo:**  
https://livekit-voice-agent-frontend.vercel.app/

**Frontend Repository:**  
https://github.com/allwinandrews/livekit-voice-agent-frontend/

The frontend is based on the official LiveKit React Agent Starter and requires no custom UI changes for Stage 1 or Stage 2.

---

## Architecture Overview

- **Backend (this repo)**  
  Handles all voice processing, inference, and conversation logic using LiveKit Agents.

- **Frontend (separate repo)**  
  Provides microphone access, audio playback, and session control using LiveKit’s React SDK.

All conversation intelligence lives entirely in the backend.

---

## Dev Setup

### Prerequisites

- Python 3.10+
- LiveKit Cloud account
- LiveKit CLI installed
- `uv` package manager

### Install dependencies

```bash
uv sync
```
````

### Environment variables

Copy `.env.example` to `.env.local` and set:

```env
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
```

You can also sync environment variables using the LiveKit CLI:

```bash
lk cloud auth
lk app env -w -d .env.local
```

---

## Running the Agent

### Download required models (one-time)

```bash
uv run python src/agent.py download-files
```

### Run locally in console mode

```bash
uv run python src/agent.py console
```

### Run for frontend or telephony (development)

```bash
uv run python src/agent.py dev
```

### Production mode

```bash
uv run python src/agent.py start
```

---

## Stage 2 – Structured Conversation Flow (Planned / In Progress)

Stage 2 extends the agent to support **state-based, multi-turn conversations** similar to Retell AI’s conversation flow model.

### Planned Use Case

Appointment scheduling (example)

### Key Concepts

- Explicit conversation **states**
- Deterministic **state transitions**
- Shared **conversation context**
- Retry and fallback handling for voice misunderstandings

### Example States

- `GREETING`
- `COLLECT_DETAILS`
- `CONFIRM_DETAILS`
- `RETRY_FALLBACK`
- `TERMINAL_END`

### Why a State-Based Approach

- Voice input is noisy and ambiguous
- Explicit states prevent hallucinated flow
- Improves reliability and user trust
- Makes multi-turn conversations predictable and testable

Stage 2 logic is implemented entirely in the backend.
**No frontend changes are required.**

---

## Tests

Run the evaluation and test suite:

```bash
uv run pytest
```

---

## Deployment

This project includes a production-ready `Dockerfile` and is deployed using **LiveKit Cloud Agents**.

For deployment details, see:
[https://docs.livekit.io/agents/ops/deployment/](https://docs.livekit.io/agents/ops/deployment/)

---

## Self-Hosted LiveKit (Optional)

You may self-host LiveKit if desired. In that case:

- Replace LiveKit Inference models with plugin-based models
- Remove LiveKit Cloud noise cancellation

Docs:
[https://docs.livekit.io/home/self-hosting/](https://docs.livekit.io/home/self-hosting/)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```

```
