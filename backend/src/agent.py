import logging
import os
import json
import time

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


# -------------------------
#   BARISTA ASSISTANT
# -------------------------

class Assistant(Agent):
    def __init__(self) -> None:

        # State object for Day 2
        self.order_state = {
            "drinkType": "",
            "size": "",
            "milk": "",
            "extras": [],
            "name": ""
        }

        super().__init__(
            instructions="""
            You are a friendly coffee shop barista for a premium café brand.

            Your goal is to take a voice order from the user by filling this JSON structure:

            {
              "drinkType": "string",
              "size": "string",
              "milk": "string",
              "extras": ["string"],
              "name": "string"
            }

            RULES:
            - Ask questions conversationally and one by one.
            - Only ask about fields that are still empty.
            - Do NOT assume anything; always confirm with the user.
            - When the user answers something, call the tool update_order_state.
            - If extras are multiple, they should be comma-separated and stored as a list.
            - When all fields are filled, call the tool finalize_order.
            - After finalizing, verbally confirm the complete order in a short friendly statement.
            """,
        )

    # -------------------------
    #   TOOL 1 — Update order
    # -------------------------

    @function_tool
    async def update_order_state(self, context: RunContext, field: str, value: str):
        """
        Update one field of the user's order.

        Allowed fields:
        - drinkType
        - size
        - milk
        - name
        - extras (comma separated list)
        """

        field = field.strip()

        if field == "extras":
            self.order_state["extras"] = [x.strip() for x in value.split(",") if x.strip()]
        else:
            self.order_state[field] = value.strip()

        return f"Updated {field} to {value}"

    # -------------------------
    #   TOOL 2 — Finalize order
    # -------------------------

    @function_tool
    async def finalize_order(self, context: RunContext):
        """
        Save the completed order to a JSON file.
        """

        orders_dir = "orders"
        os.makedirs(orders_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(orders_dir, f"order_{timestamp}.json")

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.order_state, f, indent=2)

        return {
            "message": "Order saved successfully",
            "file": filename,
            "order": self.order_state
        }


# -------------------------
#   PREWARM VAD
# -------------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# -------------------------
#   ENTRYPOINT
# -------------------------

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),

        llm=google.LLM(
            model="gemini-2.5-flash",
        ),

        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),

        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage summary: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    # Start the pipeline
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Connect to the user
    await ctx.connect()


# -------------------------
#   RUN APP
# -------------------------

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
