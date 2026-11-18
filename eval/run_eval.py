"""Generic eval runner: score a model on an eval JSONL (v1 or v2).

Sampling only — no training. Usage (tinker-venv python):
  /home/hatch/tinker-venv/bin/python eval/run_eval.py \
      --eval-path eval/eval-set-v2.jsonl --model base --out eval/v2-base.json
  /home/hatch/tinker-venv/bin/python eval/run_eval.py \
      --eval-path eval/eval-set-v2.jsonl --model v1 --out eval/v2-v1.json
  /home/hatch/tinker-venv/bin/python eval/run_eval.py \
      --eval-path eval/eval-set-v2.jsonl --model custom \
      --model-path 'tinker://.../sampler_weights/<name>' --out eval/v2-v2.json

Temp 0, 1 sample per item, max_tokens 512 (multi items need room for 3 clips).
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import score_item, summarize  # noqa: E402

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")

BASE_MODEL = "thinkingmachines/Inkling-2-Small"
V1_PATH = "tinker://d26e013b-64d9-5c5f-9b7d-713a2fc54286:train:0/sampler_weights/inkling-video-cutting-v1"


def load_eval(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-path", default=None)
    ap.add_argument("--model", choices=["base", "v1", "custom"], required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    eval_path = args.eval_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "eval-set-v2.jsonl"
    )

    for _k in ("no_proxy", "NO_PROXY"):
        _v = os.environ.get(_k, "")
        if _v:
            os.environ[_k] = ",".join(p for p in _v.split(",") if not p.strip().startswith("["))

    from dynamic_credentials import dynamic_credential_entry

    _entry = dynamic_credential_entry("custom.tinker", "access_token")
    os.environ["TINKER_API_KEY"] = _entry["surrogate"]
    print("Tinker credential loaded (surrogate, prefix hsurr:).", flush=True)

    import tinker
    from tinker.lib import _auth_token_provider as auth_mod

    def _patched_init(self, api_key: str) -> None:
        self._token = api_key
        self._expires_at = auth_mod._credential_expiry(api_key)

    auth_mod.ApiKeyAuthProvider.__init__ = _patched_init

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    service_client = tinker.ServiceClient(display_name="inkling-video-cutting-eval")

    items = load_eval(eval_path)
    print(f"Eval items: {len(items)} from {eval_path}")

    if args.model == "base":
        kwargs = {"base_model": BASE_MODEL}
    elif args.model == "v1":
        kwargs = {"model_path": V1_PATH}
    else:
        if not args.model_path:
            raise SystemExit("--model custom needs --model-path")
        kwargs = {"model_path": args.model_path}

    print(f"=== {args.model}: creating sampling client {kwargs} ===", flush=True)
    client = await service_client.create_sampling_client(**kwargs)

    results = []
    for item in items:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        try:
            result = await client.sample(
                prompt=tinker.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=args.max_tokens, temperature=0.0
                ),
            )
            text = tokenizer.decode(
                result.sequences[0].tokens, skip_special_tokens=True
            ).strip()
        except Exception as e:
            text = ""
            print(f"  {item['id']}: SAMPLE FAILED {type(e).__name__}: {e}")
        rec = {"id": item["id"], "format": item["format"]}
        if "pair" in item:
            rec["pair"] = item["pair"]
        if "style" in item:
            rec["style"] = item["style"]
        rec.update(score_item(item, text))
        results.append(rec)
        print(f"  {item['id']} ({item['format']}): correct={rec['correct']} "
              f"raw={text[:120]!r}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    summarize(args.model, results)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
