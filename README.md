# ToolUseEnhanceProject

TRL-GRPO post-training for improving small-model tool-use ability on BFCL-style
function calling tasks.

This project starts from an existing SFT checkpoint of `Llama-3.2-1B-Instruct`
and applies a lightweight GRPO stage with rule-based rewards. The goal is not to
chase leaderboard-level absolute performance, but to explore whether targeted RL
can reduce tool-use errors such as wrong tool selection, unnecessary tool calls,
format failures, and parameter hallucination.

## Project Overview

The course task focuses on tool use / function calling for small models. Our
pipeline is:

```text
Base Llama-3.2-1B-Instruct
        -> SFT model
        -> SFT + TRL GRPO
        -> BFCL v4 evaluation
```

Official BFCL v4 files are used only for benchmark evaluation. GRPO training uses
self-built BFCL-style data to avoid test leakage.

## Repository Layout

```text
scripts/
  autodl_setup.sh                  AutoDL/A800 environment setup
  generate_synthetic_tooluse_data.py
                                    generate 1,000 self-built training records
  prepare_bfcl_dataset.py           convert BFCL-style JSON/JSONL to GRPO JSONL
  train_grpo.py                     TRL GRPO training entrypoint
  download_bfcl_subset.py           optional BFCL JSON downloader
  evaluate_reward_file.py           offline reward scoring helper

src/trl_grpo_tooluse/
  data_prep.py                      prompt and dataset conversion helpers
  rewards.py                        rule-based GRPO reward functions
  tool_parser.py                    tolerant tool-call parser

data/raw_selfbuilt/
  synthetic_tooluse_1000.jsonl      self-built training data

reports/
  bfcl_13cat_summary.md             SFT vs GRPO BFCL summary table
  bfcl_13cat_summary.csv            same table in CSV format

tests/
  unit tests for data conversion, parser, rewards, and script compatibility
```

Large local artifacts are intentionally ignored by Git:

```text
sft_model/
outputs/
data/grpo_train.jsonl
*.safetensors
```

## Data

The self-built training set contains 1,000 BFCL-style records:

```text
400 normal single-turn tool calls
250 hard tool-selection cases
250 irrelevance / no-tool cases
100 missing-parameter or ambiguous-request cases
```

Regenerate it with:

```bash
python scripts/generate_synthetic_tooluse_data.py \
  --output data/raw_selfbuilt/synthetic_tooluse_1000.jsonl \
  --count 1000 \
  --seed 42
```

Convert it to the JSONL format consumed by `GRPOTrainer`:

```bash
python scripts/prepare_bfcl_dataset.py \
  --input data/raw_selfbuilt \
  --output data/grpo_train.jsonl \
  --max-samples 1000
```

The converter expects each source record to contain a user request, available
tools, a gold answer, and a task type. It accepts common field names such as
`question`, `prompt`, `tools`, `functions`, `ground_truth`, and `answer`.

## Reward Design

The GRPO reward is rule-based and does not require a separate reward model:

```text
R = format_reward
  + tool_selection_reward
  + argument_reward
  + hallucination_penalty
  + brevity_reward
```

- `format_reward`: rewards parseable tool calls, or direct answers for no-tool
  samples.
- `tool_selection_reward`: rewards choosing the correct tool and refusing tool
  calls on irrelevance tasks.
- `argument_reward`: scores parameter-key overlap and exact value matching.
- `hallucination_penalty`: penalizes invented tools and schema-invalid
  parameters.
- `brevity_reward`: lightly discourages verbose malformed templates.

The parser accepts JSON tool calls, OpenAI-style `tool_calls`, and simple
`function(arguments)` strings so that partially valid model outputs can still
receive informative rewards.

## Setup

Recommended instance:

```text
GPU: single A800, preferably 80GB
Image: Ubuntu + PyTorch + CUDA 12.x
Disk: at least 80GB
```

After cloning the repository and uploading `sft_model/`:

```bash
cd /root/autodl-tmp/ToolUseEnhanceProject
bash scripts/autodl_setup.sh
```

Check that the model directory contains:

```text
sft_model/config.json
sft_model/tokenizer.json
sft_model/model.safetensors
```

## GRPO Training

First run a short smoke test:

```bash
mkdir -p logs outputs

accelerate launch scripts/train_grpo.py \
  --model-path /root/autodl-tmp/ToolUseEnhanceProject/sft_model \
  --dataset data/grpo_train.jsonl \
  --output-dir outputs/grpo_smoke \
  --max-steps 30 \
  --num-generations 4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --use-lora \
  --bf16 \
  2>&1 | tee logs/grpo_smoke.log
```

Then run the formal small-scale experiment:

```bash
accelerate launch scripts/train_grpo.py \
  --model-path /root/autodl-tmp/ToolUseEnhanceProject/sft_model \
  --dataset data/grpo_train.jsonl \
  --output-dir outputs/grpo_full \
  --max-steps 500 \
  --num-generations 4 \
  --learning-rate 5e-6 \
  --beta 0.01 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --use-lora \
  --bf16 \
  2>&1 | tee logs/grpo_full.log
```

Useful ablations:

```bash
# Remove hallucination penalty
--disable-hallucination-penalty

# Use only format + tool-selection rewards
--reward-preset selection_only
```

If CUDA memory is tight, reduce prompt/completion length or add `--load-in-4bit`
in a bitsandbytes-compatible Linux CUDA environment.

## BFCL Evaluation

The final benchmark uses official BFCL v4 categories. Do not train on the same
official BFCL files used for reporting.

The 13 categories aligned with the group SFT table are:

```bash
TEST_CATEGORIES="parallel_multiple,simple_python,parallel,simple_java,multiple,simple_javascript,irrelevance,live_irrelevance,live_parallel_multiple,live_multiple,live_parallel,live_simple,live_relevance"
MODEL_ID="meta-llama/Llama-3.2-1B-Instruct-FC"
```

Evaluate the SFT checkpoint:

```bash
bfcl generate \
  --model "$MODEL_ID" \
  --test-category "$TEST_CATEGORIES" \
  --backend vllm \
  --num-gpus 1 \
  --gpu-memory-utilization 0.85 \
  --local-model-path /root/autodl-tmp/ToolUseEnhanceProject/sft_model \
  --result-dir /root/autodl-tmp/ToolUseEnhanceProject/bfcl_runs/result_sft_13cat_fmt

bfcl evaluate \
  --model "$MODEL_ID" \
  --test-category "$TEST_CATEGORIES" \
  --result-dir /root/autodl-tmp/ToolUseEnhanceProject/bfcl_runs/result_sft_13cat_fmt \
  --score-dir /root/autodl-tmp/ToolUseEnhanceProject/bfcl_runs/score_sft_13cat_fmt
```

Evaluate the GRPO LoRA adapter:

```bash
bfcl generate \
  --model "$MODEL_ID" \
  --test-category "$TEST_CATEGORIES" \
  --backend vllm \
  --num-gpus 1 \
  --gpu-memory-utilization 0.85 \
  --local-model-path /root/autodl-tmp/ToolUseEnhanceProject/sft_model \
  --enable-lora \
  --max-lora-rank 128 \
  --lora-modules grpo=/root/autodl-tmp/ToolUseEnhanceProject/outputs/grpo_full \
  --result-dir /root/autodl-tmp/ToolUseEnhanceProject/bfcl_runs/result_grpo_full_13cat_fmt

bfcl evaluate \
  --model "$MODEL_ID" \
  --test-category "$TEST_CATEGORIES" \
  --result-dir /root/autodl-tmp/ToolUseEnhanceProject/bfcl_runs/result_grpo_full_13cat_fmt \
  --score-dir /root/autodl-tmp/ToolUseEnhanceProject/bfcl_runs/score_grpo_full_13cat_fmt
```

`format_sensitivity` can be evaluated separately, but it should not be averaged
with the 13 ordinary accuracy categories.

## Results

The 13-category BFCL summary is:

| Task | SFT | SFT+GRPO | Delta |
|---|---:|---:|---:|
| parallel_multiple | 72.00 | 72.50 | +0.50 |
| simple_python | 83.75 | 83.75 | +0.00 |
| parallel | 74.50 | 76.00 | +1.50 |
| simple_java | 56.00 | 56.00 | +0.00 |
| multiple | 85.00 | 83.00 | -2.00 |
| simple_javascript | 64.00 | 62.00 | -2.00 |
| irrelevance | 90.42 | 90.00 | -0.42 |
| live_irrelevance | 83.37 | 83.37 | +0.00 |
| live_parallel_multiple | 41.67 | 37.50 | -4.17 |
| live_multiple | 48.91 | 49.10 | +0.19 |
| live_parallel | 37.50 | 43.75 | +6.25 |
| live_simple | 49.22 | 49.61 | +0.39 |
| live_relevance | 75.00 | 75.00 | +0.00 |
| **Average** | **66.26** | **66.28** | **+0.02** |

GRPO mostly preserves the SFT baseline, with small gains on `parallel`,
`parallel_multiple`, `live_parallel`, `live_multiple`, and `live_simple`, but
some degradation on `multiple`, `simple_javascript`, and
`live_parallel_multiple`. The average improvement is small, suggesting that the
current 1,000-row synthetic dataset and rule-based reward are useful as a
controlled exploration but are not yet enough to substantially improve the full
BFCL distribution.

## Verification

Run the test suite:

```bash
python -m unittest discover -s tests
```

The tests cover:

- reward behavior for correct calls, wrong tools, missing parameters, and
  no-tool cases
- dataset conversion error handling
- synthetic data generation and task mix
- BFCL subset downloader metadata
- TRL `GRPOConfig` API compatibility filtering

## Notes

- `bfcl_runs/` is tracked so benchmark results can be archived.
- `outputs/` is ignored because it may contain large LoRA checkpoints.
- `sft_model/` is ignored because model weights are large and should be stored
  separately.
- BFCL `Overall Acc` can be misleading when only selected categories are run.
  For this project, report the 13-category average shown above.
