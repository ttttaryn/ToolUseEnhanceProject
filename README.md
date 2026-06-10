# ToolUseEnhanceProject

This mini project starts from an existing SFT model and applies TRL `GRPOTrainer`
with rule-based rewards for BFCL-style function calling.

## Layout

- `scripts/prepare_bfcl_dataset.py`: converts BFCL-like JSON/JSONL files into a
  JSONL dataset accepted by the training script.
- `scripts/train_grpo.py`: runs TRL GRPO from an SFT model or LoRA adapter.
- `scripts/evaluate_reward_file.py`: scores saved generations with the same
  reward logic used in training.
- `src/trl_grpo_tooluse/rewards.py`: format, tool-selection, argument, hallucination,
  and brevity rewards.
- `tests/test_rewards.py`: regression tests for the reward behavior.

## 1. Prepare Data

Put BFCL training or self-constructed samples under `data/raw/`, then convert:

```powershell
python scripts/prepare_bfcl_dataset.py `
  --input data/raw `
  --output data/grpo_train.jsonl `
  --max-samples 3000
```

The converter is intentionally tolerant because BFCL-style files appear in a few
slightly different shapes. For best results, each record should contain:

- user request: `question`, `prompt`, `user`, or `messages`
- tool schema: `function`, `functions`, `tools`, or `tool_list`
- gold answer: `ground_truth`, `answer`, `possible_answer`, `gold`, or `reference`

Irrelevance/no-tool samples are detected from task/category names or from empty
gold tool calls.

## 2. Smoke Run

Use your SFT model directory as `--model-path`. If your SFT artifact is a LoRA
adapter, pass the base model via `--base-model-path`.

```powershell
accelerate launch scripts/train_grpo.py `
  --model-path F:\ToolUseforAgent\sft_model `
  --dataset data/grpo_train.jsonl `
  --output-dir outputs/grpo_smoke `
  --max-steps 30 `
  --num-generations 4 `
  --per-device-train-batch-size 1 `
  --gradient-accumulation-steps 4 `
  --use-lora
```

Add `--load-in-4bit` for QLoRA-style loading when your Linux CUDA environment
has a compatible bitsandbytes install.

For an adapter-only SFT artifact:

```powershell
accelerate launch scripts/train_grpo.py `
  --model-path F:\ToolUseforAgent\sft_adapter `
  --base-model-path F:\ToolUseforAgent\base_model `
  --dataset data/grpo_train.jsonl `
  --output-dir outputs/grpo_smoke `
  --max-steps 30 `
  --use-lora
```

## 3. Small Formal Run

```powershell
accelerate launch scripts/train_grpo.py `
  --model-path F:\ToolUseforAgent\sft_model `
  --dataset data/grpo_train.jsonl `
  --output-dir outputs/grpo_full `
  --max-steps 500 `
  --num-generations 4 `
  --learning-rate 5e-6 `
  --beta 0.01 `
  --use-lora
```

Recommended ablations:

- full reward: default settings
- no hallucination penalty: add `--disable-hallucination-penalty`
- format + tool selection only: add `--reward-preset selection_only`
- vary irrelevance ratio during dataset construction with `--irrelevance-ratio`

## 4. Verification

Run the local reward tests:

```powershell
python -m unittest discover -s tests
```

After training, evaluate Prompt-only, SFT, and GRPO checkpoints with the official
BFCL flow, then report:

- Single Turn accuracy
- Hallucination relevance / irrelevance accuracy
- format error rate
- wrong-tool rate
- parameter hallucination rate

## Notes

The reward parser accepts common tool-call forms, including JSON objects,
OpenAI-style `tool_calls`, and simple `function(arguments)` text. Training still
benefits from a strict system prompt, so the prepared prompts ask the model to
return either a JSON tool call or a direct answer when no tool is needed.

## AutoDL A800 Quickstart

Recommended AutoDL instance:

- GPU: single A800, preferably 80GB
- Image: Ubuntu + PyTorch + CUDA 12.x
- Disk: at least 80GB system/data space
- Keep the instance data disk enabled so checkpoints are not lost when the
  machine stops.

After uploading this project and `sft_model/` to AutoDL:

```bash
cd /root/autodl-tmp/ToolUseforAgent
bash scripts/autodl_setup.sh
```

Prepare BFCL data. If `data/raw` does not exist yet, either create it and copy
your JSON/JSONL files there, or clone BFCL directly on AutoDL:

```bash
mkdir -p external
git clone --depth 1 https://github.com/ShishirPatil/gorilla.git external/gorilla
```

For a quick course-project GRPO dataset, copy only the single-turn and
hallucination-oriented BFCL files you want to use for development:

```bash
mkdir -p data/raw
cp external/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json data/raw/
cp external/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multiple.json data/raw/
cp external/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_irrelevance.json data/raw/
cp external/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_irrelevance.json data/raw/
cp external/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_relevance.json data/raw/
```

Then convert the local files:

```bash
python scripts/prepare_bfcl_dataset.py \
  --input data/raw \
  --output data/grpo_train.jsonl \
  --max-samples 3000
```

For final benchmark reporting, do not train on the official test files you
intend to report. Use the BFCL official `bfcl generate` / `bfcl evaluate` flow
on Prompt-only, SFT, and SFT+GRPO checkpoints.

Smoke run:

```bash
accelerate launch scripts/train_grpo.py \
  --model-path /root/autodl-tmp/ToolUseforAgent/sft_model \
  --dataset data/grpo_train.jsonl \
  --output-dir outputs/grpo_smoke \
  --max-steps 30 \
  --num-generations 4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --use-lora \
  --bf16
```

Formal small run:

```bash
accelerate launch scripts/train_grpo.py \
  --model-path /root/autodl-tmp/ToolUseforAgent/sft_model \
  --dataset data/grpo_train.jsonl \
  --output-dir outputs/grpo_full \
  --max-steps 500 \
  --num-generations 4 \
  --learning-rate 5e-6 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --use-lora \
  --bf16
```

