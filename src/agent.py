import logging
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    inference,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# =============================================================================
# High-level: This file runs a LiveKit voice agent that schedules appointments.
#
# Flow (very simplified):
# 1) User speaks in a LiveKit room
# 2) STT converts audio -> text
# 3) LLM decides what to say and/or which tool function to call
# 4) TTS converts response -> audio and publishes back to the room
# 5) A simple state machine collects: name, reason, date/time, contact, confirm
# =============================================================================

logger = logging.getLogger("agent")

# Load environment variables (API keys, LiveKit config, etc.)
load_dotenv(".env.local")


# =============================================================================
# Session memory (shared across all states)
# =============================================================================
@dataclass
class AppointmentData:
    # "userdata" for the session: each state reads/writes this object.
    name: Optional[str] = None
    reason: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    contact: Optional[str] = None

    # Confirmation flag after we summarize details
    confirmed: bool = False

    # Counts how many times we couldn't get a valid answer
    retries: int = 0


# =============================================================================
# Prompt style: reused in every state so the voice stays consistent
# =============================================================================
BASE_STYLE = (
    "You are a helpful voice assistant for appointment scheduling. "
    "Keep responses short and natural. "
    "Ask one question at a time. "
    "If the user is unclear, ask a brief follow up. "
    "Do not use emojis or special symbols."
)


# =============================================================================
# Optional generic assistant (not used in the main flow currently)
# =============================================================================
class Assistant(Agent):
    # A generic agent that can respond to a single user turn.
    # You can use this as a fallback/general chat agent if needed later.

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful and friendly assistant. "
                "Keep responses short and natural. "
                "Do not use emojis or special symbols. "
                "If the user asks for personal info you don't have (like their birthplace), say you don't know. "
                "If the user asks for harmful or illegal instructions (like hacking), refuse politely."
            )
        )

    async def on_enter(self):
        # "on_enter" runs when this agent becomes active.
        # generate_reply() produces one assistant response for the current turn.
        await self.session.generate_reply()


# =============================================================================
# Terminal state: end the call
# =============================================================================
class EndAgent(Agent):
    # Final state: says goodbye (or any custom final message) and ends politely.

    def __init__(self, message: str = "Thanks. Goodbye.") -> None:
        super().__init__(
            instructions=f"{BASE_STYLE} End the conversation politely. Say: {message}"
        )

    async def on_enter(self):
        # Disable interruptions so the goodbye isn't cut off.
        await self.session.generate_reply(allow_interruptions=False)


# =============================================================================
# Fallback / retry state: used when user input is missing/unclear
# =============================================================================
class FallbackAgent(Agent):
    # This agent asks a simpler re-try question.

    def __init__(self, prompt: str) -> None:
        super().__init__(
            instructions=(
                f"{BASE_STYLE} "
                "The user response was unclear or incomplete. "
                f"Ask again in a simpler way: {prompt}"
            )
        )

    async def on_enter(self):
        # Ask the retry question.
        await self.session.generate_reply()

    @function_tool
    async def retry_answered(self, context: RunContext[AppointmentData]) -> Agent:
        # If the user answered properly after a retry, reduce penalty.
        context.userdata.retries = max(0, context.userdata.retries - 1)
        # After a retry response, go back to routing logic.
        return RouterAgent()


# =============================================================================
# Router state: decides which step to run next
# =============================================================================
class RouterAgent(Agent):
    # Central "brain" of the state machine.
    # Looks at AppointmentData and activates the next state.

    def __init__(self) -> None:
        super().__init__(
            instructions=f"{BASE_STYLE} Route to the next step based on missing fields."
        )

    async def on_enter(self):
        data = self.session.userdata

        # Safety: stop looping if we keep failing to get valid answers.
        if data.retries >= 3:
            await self.session.switch_agent(
                EndAgent(
                    "Sorry, I am having trouble understanding. Please try again later.")
            )
            return

        # Move through the flow in order (one missing field at a time).
        if not data.name:
            await self.session.switch_agent(CollectNameAgent())
            return

        if not data.reason:
            await self.session.switch_agent(CollectReasonAgent())
            return

        if not data.date or not data.time:
            await self.session.switch_agent(CollectDateTimeAgent())
            return

        if not data.contact:
            await self.session.switch_agent(CollectContactAgent())
            return

        # If we have everything but no confirmation yet, confirm.
        if not data.confirmed:
            await self.session.switch_agent(ConfirmAgent())
            return

        # Everything collected and confirmed: end.
        await self.session.switch_agent(EndAgent("Your appointment is confirmed. Goodbye."))


# =============================================================================
# State 1: collect user's name (combined "Intro + CollectName")
# =============================================================================
class CollectNameAgent(Agent):
    # First state in the flow:
    # - Greets the user
    # - Asks for their name
    # - Stores it using set_name(...)
    #
    # Combining Intro + CollectName avoids an extra state that can't store anything.

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"{BASE_STYLE} "
                "Start the call by greeting the user and saying you can help schedule an appointment. "
                "Then ask for the user's name. "
                "When you have the name, call set_name."
            )
        )

    async def on_enter(self):
        await self.session.generate_reply()

    @function_tool
    async def set_name(self, context: RunContext[AppointmentData], name: str) -> Agent:
        # Normalize and store user input in shared session memory.
        context.userdata.name = name.strip()[:80] if name else None

        # If invalid, count a retry and ask again in a simpler way.
        if not context.userdata.name:
            context.userdata.retries += 1
            return FallbackAgent("What is your name")

        # Otherwise continue the flow.
        return RouterAgent()


# =============================================================================
# State 2: collect appointment reason
# =============================================================================
class CollectReasonAgent(Agent):
    # The LLM should call set_reason(reason=...) when it has a clear reason.

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"{BASE_STYLE} "
                "Ask what the appointment is for in one short question. "
                "When you have a clear reason, call set_reason."
            )
        )

    async def on_enter(self):
        await self.session.generate_reply()

    @function_tool
    async def set_reason(self, context: RunContext[AppointmentData], reason: str) -> Agent:
        # Store a trimmed reason (limit length to avoid huge context).
        cleaned = (reason or "").strip()
        context.userdata.reason = cleaned[:200] if cleaned else None

        if not context.userdata.reason:
            context.userdata.retries += 1
            return FallbackAgent("What is the appointment for")

        return RouterAgent()


# =============================================================================
# State 3: collect date and time (supports partial answers)
# =============================================================================
class CollectDateTimeAgent(Agent):
    # The LLM should call set_date_time(date=..., time=...) when both are known.

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"{BASE_STYLE} "
                "Ask for the preferred date and time. "
                "If the user gives only one, ask for the missing one. "
                "When you have both, call set_date_time with date and time as plain text."
            )
        )

    async def on_enter(self):
        await self.session.generate_reply()

    @function_tool
    async def set_date_time(
        self, context: RunContext[AppointmentData], date: str, time: str
    ) -> Agent:
        # Store date/time as free-text (good enough for a take-home).
        # In production you might parse and validate into a real datetime.
        d = (date or "").strip()
        t = (time or "").strip()
        context.userdata.date = d[:60] if d else None
        context.userdata.time = t[:60] if t else None

        # If date or time is missing, ask only for the missing piece.
        if not context.userdata.date or not context.userdata.time:
            context.userdata.retries += 1

            if not context.userdata.date and not context.userdata.time:
                return FallbackAgent("What date and time would you like")
            if not context.userdata.date:
                return FallbackAgent("What date would you like")
            return FallbackAgent("What time would you like")

        return RouterAgent()


# =============================================================================
# State 4: collect contact info (phone or email)
# =============================================================================
class CollectContactAgent(Agent):
    # The LLM should call set_contact(contact=...) when it extracts a phone/email.

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"{BASE_STYLE} "
                "Ask for a contact method, either phone number or email. "
                "Repeat it back once for clarity. "
                "When you have it, call set_contact."
            )
        )

    async def on_enter(self):
        await self.session.generate_reply()

    @function_tool
    async def set_contact(self, context: RunContext[AppointmentData], contact: str) -> Agent:
        # Store trimmed contact details (phone or email).
        c = (contact or "").strip()
        context.userdata.contact = c[:120] if c else None

        if not context.userdata.contact:
            context.userdata.retries += 1
            return FallbackAgent("What is the best phone number or email to reach you")

        return RouterAgent()


# =============================================================================
# State 5: confirmation step
# =============================================================================
class ConfirmAgent(Agent):
    # Summarizes collected info and asks the user to confirm yes/no.

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"{BASE_STYLE} "
                "Summarize the details from context: name, reason, date, time, contact. "
                "Ask for confirmation with a yes or no. "
                "If yes call confirm_yes. If no call confirm_no."
            )
        )

    async def on_enter(self):
        await self.session.generate_reply()

    @function_tool
    async def confirm_yes(self, context: RunContext[AppointmentData]) -> Agent:
        # User accepted the summary.
        context.userdata.confirmed = True
        return RouterAgent()

    @function_tool
    async def confirm_no(self, context: RunContext[AppointmentData]) -> Agent:
        # User rejected the summary.
        # Reset only the most likely incorrect fields (date/time) and re-ask them.
        context.userdata.confirmed = False
        context.userdata.date = None
        context.userdata.time = None

        # Reset retries to keep UX fair when user changes their mind.
        context.userdata.retries = 0

        return CollectDateTimeAgent()


# =============================================================================
# LiveKit server setup
# =============================================================================

# AgentServer hosts your rtc_session() entrypoint and handles worker lifecycle.
server = AgentServer()


def prewarm(proc: JobProcess):
    # Preload VAD model once per worker process (performance optimization).
    proc.userdata["vad"] = silero.VAD.load()


# Register prewarm hook.
server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    # Add room name to logs for easier debugging/observability.
    ctx.log_context_fields = {"room": ctx.room.name}

    # AgentSession wires the "voice pipeline" (STT -> LLM -> TTS) and runs Agents.
    session: AgentSession[AppointmentData] = AgentSession(
        # STT: speech -> text
        stt=inference.STT(
            model="assemblyai/universal-streaming", language="en"),

        # LLM: decides what to say + which @function_tool to call
        llm=inference.LLM(model="openai/gpt-4.1-mini"),

        # TTS: text -> speech
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),

        # Detect when a user has finished a turn (helps with timing/interruptions).
        turn_detection=MultilingualModel(),

        # Detect voice activity (speech vs silence/noise).
        vad=ctx.proc.userdata["vad"],

        # Reduce perceived latency by starting generation earlier.
        preemptive_generation=True,

        # Shared per-call state used by Router + state agents.
        userdata=AppointmentData(),
    )

    # Start the session in the LiveKit room beginning with CollectNameAgent.
    await session.start(
        agent=CollectNameAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # Use different noise cancellation profiles depending on participant type.
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )

    # Connect to the room (after setup is done).
    await ctx.connect()


if __name__ == "__main__":
    # Entry point: run the LiveKit agent worker.
    cli.run_app(server)
