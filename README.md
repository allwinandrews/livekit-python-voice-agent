Live demo: https://livekit-voice-agent-frontend.vercel.app/  
Frontend repo: https://github.com/allwinandrews/livekit-voice-agent-frontend/

<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# LiveKit Agents Starter – Python

A complete Python-based voice agent built with [LiveKit Agents for Python](https://github.com/livekit/agents) and deployed on [LiveKit Cloud](https://cloud.livekit.io/).

This project was implemented as part of a **Python Developer Test** and covers:

- **Stage 1**: A fully working voice AI agent (speech → LLM → speech)
- **Stage 2 (design + partial implementation)**: A structured, state-based conversation flow

---

## What this project does

The agent:

- Joins a LiveKit room
- Listens to the user’s voice
- Transcribes speech to text (STT)
- Sends text to an LLM
- Converts the response back to speech (TTS)
- Streams the audio response back into the room in real time

A production-ready React frontend is deployed separately and connects to this agent.

---

## Key features

- Voice AI pipeline using:
  - **STT**: AssemblyAI
  - **LLM**: OpenAI (GPT-4.1 mini)
  - **TTS**: Cartesia
- LiveKit turn detection with multilingual support
- Background noise cancellation
- Preemptive response generation for low-latency voice interaction
- Dockerized and deployed on LiveKit Cloud
- Compatible with web, mobile, or telephony frontends

---

## Stage 2: Structured Conversation Flow (State-Based Design)

This project is designed to support **state/DAG-based conversations**, similar to tools like Retell AI.

### Use case chosen

**Appointment scheduling**

### Example conversation states

- `GREETING` – Welcomes the user and explains the task
- `COLLECT_DETAILS` – Gathers required information (e.g. date, time)
- `CONFIRMATION` – Confirms the collected details with the user
- `RETRY / FALLBACK` – Handles unclear or invalid responses
- `END` – Final confirmation and graceful exit

### Why a state-based approach

- Voice input is noisy and ambiguous
- Explicit states reduce misinterpretation
- Retries and confirmations improve reliability
- Conversation logic becomes predictable, testable, and extensible

The current backend structure supports maintaining:

- `current_state`
- conversation context (slots)
- state-based routing per user turn

Frontend changes are **not required** for Stage 2.

---

## Dev setup

Clone the repository and install dependencies:

```bash
uv sync
```
````

Create `.env.local` from `.env.example` and set:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`

Authenticate with LiveKit Cloud:

```bash
lk cloud auth
lk app env -w -d .env.local
```

---

## Run the agent locally

Download required models:

```bash
uv run python src/agent.py download-files
```

Run in terminal (console mode):

```bash
uv run python src/agent.py console
```

Run for frontend / LiveKit Cloud development:

```bash
uv run python src/agent.py dev
```

Production entrypoint:

```bash
uv run python src/agent.py start
```

---

## Frontend

The frontend is a React + Next.js app based on LiveKit’s official starter.

- **Live demo**: [https://livekit-voice-agent-frontend.vercel.app/](https://livekit-voice-agent-frontend.vercel.app/)
- **Repository**: [https://github.com/allwinandrews/livekit-voice-agent-frontend/](https://github.com/allwinandrews/livekit-voice-agent-frontend/)

The frontend handles:

- microphone capture
- room connection
- audio playback
- transcripts

No frontend changes are required for Stage 2.

---

## Tests

Run backend tests with:

```bash
uv run pytest
```

---

## Deployment

The backend is deployed to **LiveKit Cloud** using the provided Dockerfile.
The frontend is deployed on **Vercel**.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
