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

logger = logging.getLogger("agent")

load_dotenv(".env.local")


@dataclass
class AppointmentData:
    name: Optional[str] = None
    reason: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    contact: Optional[str] = None
    confirmed: bool = False
    retries: int = 0


BASE_STYLE = (
    "You are a helpful voice assistant for appointment scheduling. "
    "Keep responses short and natural. "
    "Ask one question at a time. "
    "If the user is unclear, ask a brief follow up. "
    "Do not use emojis or special symbols."
)


class Assistant(Agent):
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
        # Generate a reply for the current user turn.
        # In tests, AgentSession.run(...) provides user_input and expects one assistant message.
        await self.session.generate_reply()


class EndAgent(Agent):
    def __init__(self, message: str = "Thanks. Goodbye.") -> None:
        super().__init__(
            instructions=f"{BASE_STYLE} End the conversation politely. Say: {message}"
        )

    async def on_enter(self):
        await self.session.generate_reply(allow_interruptions=False)


class FallbackAgent(Agent):
    def __init__(self, prompt: str) -> None:
        super().__init__(
            instructions=(
                f"{BASE_STYLE} "
                "The user response was unclear or incomplete. "
                f"Ask again in a simpler way: {prompt}"
            )
        )

    async def on_enter(self):
        await self.session.generate_reply()

    @function_tool
    async def retry_answered(self, context: RunContext[AppointmentData]) -> Agent:
        context.userdata.retries = max(0, context.userdata.retries - 1)
        return RouterAgent()


class RouterAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"{BASE_STYLE} Route to the next step based on missing fields."
        )

    async def on_enter(self):
        data = self.session.userdata
        if data.retries >= 3:
            await self.session.switch_agent(
                EndAgent(
                    "Sorry, I am having trouble understanding. Please try again later."
                )
            )
            return

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

        if not data.confirmed:
            await self.session.switch_agent(ConfirmAgent())
            return

        await self.session.switch_agent(
            EndAgent("Your appointment is confirmed. Goodbye.")
        )


class IntroAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"{BASE_STYLE} "
                "Start the call. Briefly say you can help schedule an appointment. "
                "Then ask for the user's name."
            )
        )

    async def on_enter(self):
        await self.session.generate_reply()


class CollectNameAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"{BASE_STYLE} "
                "Ask for the user's name. "
                "When you have the name, call set_name."
            )
        )

    async def on_enter(self):
        await self.session.generate_reply()

    @function_tool
    async def set_name(self, context: RunContext[AppointmentData], name: str) -> Agent:
        context.userdata.name = name.strip()[:80] if name else None
        if not context.userdata.name:
            context.userdata.retries += 1
            return FallbackAgent("What is your name")
        return RouterAgent()


class CollectReasonAgent(Agent):
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
    async def set_reason(
        self, context: RunContext[AppointmentData], reason: str
    ) -> Agent:
        cleaned = (reason or "").strip()
        context.userdata.reason = cleaned[:200] if cleaned else None
        if not context.userdata.reason:
            context.userdata.retries += 1
            return FallbackAgent("What is the appointment for")
        return RouterAgent()


class CollectDateTimeAgent(Agent):
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
        d = (date or "").strip()
        t = (time or "").strip()
        context.userdata.date = d[:60] if d else None
        context.userdata.time = t[:60] if t else None

        if not context.userdata.date or not context.userdata.time:
            context.userdata.retries += 1
            if not context.userdata.date and not context.userdata.time:
                return FallbackAgent("What date and time would you like")
            if not context.userdata.date:
                return FallbackAgent("What date would you like")
            return FallbackAgent("What time would you like")
        return RouterAgent()


class CollectContactAgent(Agent):
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
    async def set_contact(
        self, context: RunContext[AppointmentData], contact: str
    ) -> Agent:
        c = (contact or "").strip()
        context.userdata.contact = c[:120] if c else None
        if not context.userdata.contact:
            context.userdata.retries += 1
            return FallbackAgent("What is the best phone number or email to reach you")
        return RouterAgent()


class ConfirmAgent(Agent):
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
        context.userdata.confirmed = True
        return RouterAgent()

    @function_tool
    async def confirm_no(self, context: RunContext[AppointmentData]) -> Agent:
        context.userdata.confirmed = False
        context.userdata.date = None
        context.userdata.time = None
        context.userdata.retries = 0
        return CollectDateTimeAgent()


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session: AgentSession[AppointmentData] = AgentSession(
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        userdata=AppointmentData(),
    )

    await session.start(
        agent=IntroAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
