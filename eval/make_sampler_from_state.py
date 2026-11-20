"""Make a sampler checkpoint from a saved training state.

Usage (after uv sync --extra train):
  uv run python eval/make_sampler_from_state.py \
      <state-tinker-path> <sampler-name>

Resumes a training client from the state (no training steps) and saves
sampler weights under <sampler-name>. Used when a training run saved its
state but the sampler save failed (e.g. storage limit) — the weights are
already good; only the sampler copy is missing.
"""
import asyncio
import os
import sys

from tinker_env import setup_tinker


async def main():
    state_path, name = sys.argv[1:3]
    setup_tinker()

    import tinker

    service_client = tinker.ServiceClient(display_name="inkling-video-cutting-sampler")
    print(f"Resuming from state {state_path} ...", flush=True)
    tc = await service_client.create_training_client_from_state(state_path)
    print("Training client resumed.", flush=True)
    ckpt = await tc.save_weights_for_sampler(name=name)
    print(f"SAMPLER_PATH={ckpt.path}", flush=True)
    print(f"Playground: {ckpt.get_playground_url()}&reasoning_effort=none", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
