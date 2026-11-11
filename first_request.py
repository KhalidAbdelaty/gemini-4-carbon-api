"""First Gemini 4 Carbon request: give it PayGuard's policy and model summary, ask what it needs."""

import json
import time

import config as cfg
from agent import make_client

summary = json.loads(open("data/model_summary.json").read())
prompt = (
    "PayGuard is an e-commerce fraud classifier that gives each transaction a risk score from 0 to 1. "
    f"Model summary: {json.dumps(summary)}\n\n{cfg.POLICY_TEXT}\n\n"
    f"The current threshold is {cfg.CURRENT_THRESHOLD}. Before you judge whether this model can be "
    "deployed, list in at most five short bullets the measurements you would need and why."
)

client = make_client()  # loads only GEMINI_API_KEY from .env
start = time.perf_counter()
interaction = client.interactions.create(
    model=cfg.MODEL_ID,
    input=prompt,
    generation_config={"thinking_level": "medium"},
    timeout=cfg.HTTP_TIMEOUT_SECONDS,
)
print(interaction.output_text)
u = interaction.usage
print(f"\nstatus={interaction.status} | {time.perf_counter() - start:.1f}s | input={u.total_input_tokens} "
      f"cached={u.total_cached_tokens} output={u.total_output_tokens} thinking={u.total_thought_tokens}")
