# Data

Prompt-level outcomes of every trained drafter in the paper, as measured. The scripts in `experiments/` rebuild the paper's tables from these files and nothing else. No prompt text, image, output token or model weight is included, only counts and timings.

A verification round commits the draft tokens the target accepts plus the one token the target adds, so the mean acceptance length τ (MAT) of a set of prompts is the sum of `committed` over the sum of `rounds`.

## vision/

Qwen3-VL-8B and Qwen3-VL-4B targets with DFlash-family drafters trained on ALLaVA or ShareGPT4V, evaluated on COCO captioning (`caption`), TextVQA (`textvqa`) and DocVQA (`docvqa`). Each domain has 100 fixed prompts, the first is a warm-up, and prompts 1 to 99 are scored.

`runs.csv` has one row per evaluation run.

| column | meaning |
|---|---|
| `run` | run id, used as the key in the other files |
| `set` | `acceptance` for the MAT evaluations, `timing` for the H100 speed runs |
| `session` | timing session. 1 is the main H100 session of Table 16, 2 the later session behind the remaining speed cells of Table 1, 3 the EAGLE-3 comparison of Table 17 |
| `target` | `8b` or `4b` |
| `corpus` | training corpus, `allava` or `sharegpt4v` |
| `method` | `dflash`, `kd`, `erase`, `erase_hard`, `alr`, `alr_ira`, `slot_control`, `regen_truncated`, `regen_window` or `eagle3` |
| `epochs` | training epochs |
| `rollout_depth` | ALR rollout depth R, empty for methods without a rollout |
| `gamma` | slot envelope γ |
| `data_order` | 0 for the main data order, 1 and 2 for the two other orders of Table 15 |
| `seed` | training seed |
| `temperature` | decoding temperature, 0 or 1 |
| `decoder` | `tree` for the main decoder, `chain` for the chain reading of Table 15, `eagle3_tree` for EAGLE-3 |
| `gpu` | `A6000` or `H100` |

`acceptance.csv` has one row per scored prompt of an acceptance run, with the columns `run`, `domain`, `prompt`, `committed` and `rounds`.

`timing.csv` has the same columns for the timing runs, plus `ar_ms` and `draft_ms`, the milliseconds per output token of autoregressive decoding and of the drafter on that prompt in the same run. Both exclude the first target token, and the drafter time excludes the first draft construction.

`rounds.csv` has one row per verification round on the COCO prompts for one checkpoint each of ALR, ALR + IRA and the two regeneration references, with the columns `run`, `prompt`, `start` (the output token offset where the round starts) and `committed`. It backs Table 4b, and its totals per prompt equal the rows of `acceptance.csv`.

## text/

Qwen3-4B with drafters trained on a fixed UltraChat subset, evaluated on MT-Bench, Alpaca, GSM8K, AIME24, AIME25, HumanEval and LiveCodeBench on one H100.

| file | columns | content |
|---|---|---|
| `runs.csv` | `run`, `method`, `seed`, `timed` | one row per training replicate. `timed` runs also carry drafter timings |
| `prompts.csv` | `run`, `task`, `temperature`, `prompt`, `committed`, `rounds`, `draft_ms` | one row per benchmark prompt, in benchmark order from 0 |
| `ar_reference.csv` | `task`, `temperature`, `prompt`, `ar_ms` | the autoregressive timing that every method shares within a task and temperature |
| `a100_timing.csv` | `task`, `prompt`, `seed`, `repeat`, `temperature`, then `ar`, `erase` and `alr_ira` each with `_total_ms` and `_decode_ms` | the repeated A100 timing of Table 11 on 220 fixed prompts, two checkpoint pairs (`seed` 1 and 2) and three timing repeats |

In `a100_timing.csv`, `prompt` counts the fixed prompts of each task from 0 in the order of the timing subset, 32 per benchmark and all 30 of each AIME set. The `_total_ms` columns are total generation milliseconds per token, the measure Table 11 reports, and the `_decode_ms` columns are the decode-only companion.

## What is not here

The evaluation logs behind Figure 4b and 4c, the A100 re-evaluations behind the cross-device shifts of Appendix E, the EAGLE-3 timings that include the first draft construction and the training logs are not included. The README of the repository lists which paper values therefore are not rebuilt.
