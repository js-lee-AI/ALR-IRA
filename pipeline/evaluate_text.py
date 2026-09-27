import argparse
import hashlib
import json
from pathlib import Path
import time

from alr.metrics import summarize


def reference_rows(path, target, max_new, sample_seed):
    if not path:
        return None
    reference = json.loads(Path(path).read_text())
    expected = {"target": target, "max_new": max_new, "sample_seed": sample_seed}
    if any(reference["config"].get(key) != value for key, value in expected.items()):
        raise ValueError("The AR reference uses different evaluation settings")
    return {
        key: {row["id"]: row["ar"] for row in cell["rows"]}
        for key, cell in reference["cells"].items()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a text drafter at T=0 and T=1"
    )
    parser.add_argument("--target", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tasks", default="mt-bench,alpaca,gsm8k,aime24,aime25,humaneval,livecodebench"
    )
    parser.add_argument("--temperatures", default="0,1")
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--ar-reference")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists")
    reference = reference_rows(
        args.ar_reference, args.target, args.max_new, args.sample_seed
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dflash.model import DFlashDraftModel, dflash_generate

    torch.backends.cuda.matmul.allow_tf32 = False
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.target, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    draft = (
        DFlashDraftModel.from_pretrained(
            args.draft, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    result = {"config": vars(args), "cells": {}}
    output.parent.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def run(ids, block, temperature, seed):
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        stats = dflash_generate(
            draft,
            target,
            ids,
            args.max_new,
            [tokenizer.eos_token_id],
            temperature,
            block_size=block,
            return_stats=True,
        )
        torch.cuda.synchronize()
        tokens = stats.output_ids.shape[1] - ids.shape[1]
        lengths = stats.acceptance_lengths
        return {
            "tau": sum(lengths) / len(lengths),
            "rounds": len(lengths),
            "decode_ms_token": 1000 * stats.time_per_output_token,
            "total_ms_token": 1000 * (time.perf_counter() - start) / max(tokens, 1),
            "tokens": tokens,
        }

    for temperature in map(float, args.temperatures.split(",")):
        for task in args.tasks.split(","):
            pool = [
                json.loads(line)
                for line in (Path(args.data_dir) / (task + ".jsonl"))
                .read_text()
                .splitlines()
                if line.strip()
            ]
            if args.limit:
                pool = pool[: args.limit]
            key = f"{task}/T{temperature:g}"
            if reference is not None and (
                key not in reference or set(reference[key]) != {p["id"] for p in pool}
            ):
                raise ValueError("The AR reference covers different prompts")
            rows = []
            for i, prompt in enumerate(pool):
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt["prompt"]}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_tensors="pt",
                ).cuda()
                seed = args.sample_seed * 1000003 + int(
                    hashlib.sha256(prompt["id"].encode()).hexdigest()[:8], 16
                )
                if i == 0:
                    run(ids, 1, temperature, seed)
                    run(ids, draft.block_size, temperature, seed)
                row = {"id": prompt["id"]}
                if reference is not None:
                    row["ar"] = reference[key][prompt["id"]]
                order = [("ar", 1), ("draft", draft.block_size)]
                if i % 2:
                    order.reverse()
                for name, block in order:
                    if name == "ar" and reference is not None:
                        continue
                    row[name] = run(ids, block, temperature, seed)
                rows.append(row)
            result["cells"][f"{task}/T{temperature:g}"] = {
                "summary": summarize(rows),
                "rows": rows,
            }
            output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
