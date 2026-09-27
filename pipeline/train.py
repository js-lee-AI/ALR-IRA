import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

# Objective flags for each training arm of the paper. Every KD arm uses the
# top-8 plus tail loss of Eq. 1 with the slot envelope of the config.
KD = ["--loss-type", "dflash_targetkd"]
METHODS = {
    "dflash": ["--loss-type", "dflash"],
    "kd": KD,
    "erase": KD + ["--dflash-kd-survival-soft"],
    "erase_hard": KD + ["--dflash-kd-survival-mask"],
    "alr": KD + ["--dflash-kd-anchor-rollout"],
    "alr_gate": KD + ["--dflash-kd-anchor-rollout", "--dflash-kd-rollout-survival-soft"],
    "alr_ira": KD + ["--dflash-kd-anchor-rollout", "--dflash-kd-inrollout-anchors"],
    "slot_control": KD + ["--dflash-kd-anchor-rollout", "--dflash-kd-slotmatch-control"],
}


def build_command(args, config, draft_init):
    if args.gpus < 1 or args.microbatch < 1:
        raise ValueError("GPU count and microbatch must be positive")
    denom = args.gpus * args.microbatch
    if config["global_batch_size"] % denom:
        raise ValueError("GPU count times microbatch must divide the global batch")
    rollout = "--dflash-kd-anchor-rollout" in METHODS[args.method]
    if args.rollout_depth is not None and args.method not in ("alr", "alr_gate"):
        raise ValueError("--rollout-depth applies to ALR alone, IRA needs the full rollout")
    if args.rollout_depth is not None and not 1 <= args.rollout_depth <= config["block_size"] - 2:
        raise ValueError(f"--rollout-depth must be in [1, {config['block_size'] - 2}]")
    # The DFlash cross-entropy baseline keeps its own envelope (7 for the
    # vision targets, as in the public DFlash recipe).
    gamma = config["dflash_gamma"] if args.method == "dflash" else config["gamma"]
    if args.gamma is not None:
        gamma = args.gamma
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={args.gpus}",
        "-m",
        "alr.train_backend",
        "--target-model-path",
        args.target or config["target"],
        "--draft-config-path",
        str(Path(draft_init) / "config.json"),
        "--draft-init-path",
        str(draft_init),
        "--train-data-path",
        str(Path(args.data).resolve()),
        "--output-dir",
        str(Path(args.output).resolve()),
        "--cache-dir",
        str(Path(args.cache).resolve()),
        "--num-epochs",
        str(args.epochs or config["epochs"]),
        "--batch-size",
        str(args.microbatch),
        "--accumulation-steps",
        str(config["global_batch_size"] // denom),
        "--learning-rate",
        str(config["learning_rate"]),
        "--max-length",
        str(config["max_length"]),
        "--chat-template",
        config["chat_template"],
        "--attention-backend",
        "sdpa",
        "--target-model-backend",
        "hf",
        "--block-size",
        str(config["block_size"]),
        "--num-anchors",
        str(config["num_anchors"]),
        "--embedding-key",
        config["embedding_key"],
        "--lm-head-key",
        config["lm_head_key"],
        "--loss-decay-gamma",
        str(gamma),
        *METHODS[args.method],
        "--dflash-kd-topk",
        str(config["topk"]),
        "--dflash-kd-lambda-scale",
        str(config["kd_scale"]),
        "--seed",
        str(args.seed),
        "--data-order-seed",
        str(config["data_order_seed"] if args.data_order_seed is None else args.data_order_seed),
        "--dataloader-num-workers",
        "2",
        "--build-dataset-num-proc",
        "2",
        "--save-interval",
        "500",
        "--log-interval",
        "20",
        "--report-to",
        "none",
    ]
    if config["is_vlm"]:
        command += [
            "--is-vlm",
            "--min-pixels",
            str(config["min_pixels"]),
            "--max-pixels",
            str(config["max_pixels"]),
        ]
    else:
        command += ["--is-preformatted"]
    if rollout and args.rollout_depth is not None and args.rollout_depth < config["block_size"] - 2:
        command += ["--dflash-kd-rollout-steps", str(args.rollout_depth)]
    if args.resume:
        command += ["--resume"]
    if args.max_steps:
        command += ["--max-train-steps", str(args.max_steps)]
    return command


def main():
    parser = argparse.ArgumentParser(description="Train a DFlash drafter with one of the paper's objectives")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=sorted(METHODS), default="alr_ira")
    parser.add_argument("--epochs", type=int, help="overrides the config's epoch count")
    parser.add_argument("--rollout-depth", type=int, help="ALR rollout depth R, 14 by default")
    parser.add_argument("--gamma", type=float, help="overrides the slot envelope")
    parser.add_argument("--data-order-seed", type=int, help="overrides the config's data order")
    parser.add_argument("--target")
    parser.add_argument("--draft-init")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=6)
    parser.add_argument("--cache", default=".cache/training")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    draft_init = args.draft_init or config["draft_init"]
    if not args.dry_run:
        if not Path(args.data).is_file():
            parser.error("Training JSONL does not exist")
        if not Path(draft_init).is_dir():
            from huggingface_hub import snapshot_download

            draft_init = snapshot_download(draft_init)
    command = build_command(args, config, draft_init)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1")
    subprocess.run(command, cwd=Path(__file__).resolve().parent, env=env, check=True)


if __name__ == "__main__":
    main()
