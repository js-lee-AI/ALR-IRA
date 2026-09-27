import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

MATH_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)
SOURCES = {
    "alpaca": "https://raw.githubusercontent.com/SafeAILab/EAGLE/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/data/alpaca/question.jsonl",
    "aime25": "https://huggingface.co/datasets/math-ai/aime25/resolve/563bb8404243c5f09de6ec262f2db674fe5bce9b/test.jsonl",
    "livecodebench": "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test.jsonl",
}


def download_jsonl(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return [
                    json.loads(line)
                    for line in response.read().decode().splitlines()
                    if line.strip()
                ]
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


def prepare():
    from datasets import load_dataset

    tasks = {}
    for task, repo, subset, split in [
        ("mt-bench", "HuggingFaceH4/mt_bench_prompts", None, "train"),
        ("gsm8k", "openai/gsm8k", "main", "test"),
        ("humaneval", "openai/openai_humaneval", None, "test"),
    ]:
        rows = []
        for i, row in enumerate(load_dataset(repo, subset, split=split)):
            if task == "mt-bench":
                prompt = row["prompt"][0]
            elif task == "gsm8k":
                prompt = row["question"] + MATH_SUFFIX
            else:
                prompt = (
                    "Write a solution to the following problem and make sure that it passes the tests:\n"
                    + chr(96) * 3
                    + "python\n"
                    + row["prompt"]
                    + "\n"
                    + chr(96) * 3
                )
            rows.append({"id": f"{task}:{i}", "prompt": prompt})
        tasks[task] = rows
    tasks["aime24"] = [
        {"id": f"aime24:{row['id']}", "prompt": row["problem"] + MATH_SUFFIX}
        for row in load_dataset(
            "math-ai/aime24",
            split="test",
            revision="83a7f387baaa524a8bda0022eac0541582297103",
        )
    ]
    for task, url in SOURCES.items():
        rows = []
        for i, row in enumerate(download_jsonl(url)):
            if task == "alpaca":
                identifier, prompt = row["question_id"], row["turns"][0]
            elif task == "aime25":
                identifier = row.get("unique_id", row.get("id", i))
                prompt = row["problem"] + MATH_SUFFIX
            else:
                identifier = row["question_id"]
                prompt = (
                    "Write a Python solution to the following problem.\n\n"
                    + row["question_content"]
                )
                if row.get("starter_code"):
                    prompt += (
                        "\n\nUse this starter code:\n"
                        + chr(96) * 3
                        + "python\n"
                        + row["starter_code"]
                        + "\n"
                        + chr(96) * 3
                    )
            rows.append({"id": f"{task}:{identifier}", "prompt": prompt})
        tasks[task] = rows
    return tasks


def prompt_hash(rows):
    payload = json.dumps(
        [{"id": row["id"], "prompt": row["prompt"]} for row in rows],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the seven text evaluation sets"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    expected = json.loads(
        (Path(__file__).parent / "data/benchmark_manifest.json").read_text()
    )
    tasks = prepare()
    for task, rows in tasks.items():
        if (
            len(rows) != expected[task]["count"]
            or prompt_hash(rows) != expected[task]["prompt_sha256"]
        ):
            raise ValueError(
                f"The released {task} prompts differ from the fixed evaluation set"
            )
    output.mkdir(parents=True, exist_ok=True)
    for task, rows in tasks.items():
        path = output / (task + ".jsonl")
        if path.exists():
            raise FileExistsError(path)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
    print({task: len(rows) for task, rows in tasks.items()})
