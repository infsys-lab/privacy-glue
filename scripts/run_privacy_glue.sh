#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
usage: run_privacy_glue.sh [option...]

optional arguments:
  --cuda_visible_devices  <str>
                          comma separated string of integers passed directly to
                          the "CUDA_VISIBLE_DEVICES" environmental variable
                          (default: 0)

  --fp16_all              enable 16-bit mixed precision computation
                          through NVIDIA Apex for both training and evaluation
                          (default: False)

  --model_name_or_path    <str>
                          model to be used for fine-tuning. Currently only the
                          following are supported:
                          "all-mpnet-base-v2",
                          "bert-base-uncased",
                          "nlpaueb/legal-bert-base-uncased",
                          "mukund/privbert"
                          (default: bert-base-uncased)

  --no_cuda               disable CUDA even when available (default: False)

  --num_workers           <int>
                          number of workers to be used for preprocessing
                          (default: 1)

  --overwrite             overwrite cached data and saved checkpoint(s)
                          (default: False)

  --task                  <str>
                          task to be worked on. The following values are
                          accepted: "opp_115", "piextract",
                          "policy_detection", "policy_ie_a", "policy_ie_b",
                          "policy_qa", "privacy_qa", "all"
                          (default: all)

  --wandb                 log metrics and result to wandb
                          (default: False)

  -h, --help              show this help message and exit
EOF
}

parser() {
  while [[ -n "$1" ]]; do
    case "$1" in
    --fp16_all)
      FP16_ALL=("--fp16" "--fp16_full_eval")
      ;;
    --overwrite)
      OVERWRITE=("--overwrite_cache" "--overwrite_output_dir")
      ;;
    --no_cuda)
      NO_CUDA=("--no_cuda")
      ;;
    --wandb)
      WANDB="wandb"
      ;;
    --model_name_or_path)
      if [[ -n "$2" ]]; then
        shift
        MODEL_NAME_OR_PATH="$1"
      else
        {
          printf "%s\n\n" "Missing --model_name_or_path argument"
          usage
        } >&2
        exit 1
      fi
      ;;
    --num_workers)
      if [[ -n "$2" ]]; then
        shift
        NUM_WORKERS="$1"
      else
        {
          printf "%s\n\n" "Missing --num_workers argument"
          usage
        } >&2
        exit 1
      fi
      ;;
    --task)
      if [[ -n "$2" ]]; then
        shift
        TASK="$1"
      else
        {
          printf "%s\n\n" "Missing --task argument"
          usage
        } >&2
        exit 1
      fi
      ;;
    --cuda_visible_devices)
      if [[ -n "$2" ]]; then
        shift
        CUDA_VISIBLE_DEVICES="$1"
        N_GPU=$(($(printf "%s" "$CUDA_VISIBLE_DEVICES" |
          sed 's/^,\+//g; s/,\+$//g; s/,\+ */,/g' |
          tr -cd , |
          wc -c) + 1))
      else
        {
          printf "%s\n\n" "Missing --cuda_visible_devices argument"
          usage
        } >&2
        exit 1
      fi
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      {
        printf "%s\n\n" "Unknown option $1"
        usage
      } >&2
      exit 1
      ;;
    esac
    shift
  done

  # add post-parsing sanity checks
  if [ -n "${NO_CUDA[*]}" ]; then
    CUDA_VISIBLE_DEVICES=""
    N_GPU=0
  elif [ "$N_GPU" -gt "1" ]; then
    PROGRAM_RUNTIME=("torchrun" "--nproc_per_node" "$N_GPU")
  fi
}

main() {
  # execute fine-tuning
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "${PROGRAM_RUNTIME[@]}" \
    src/privacy_glue.py \
    --task "$TASK" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --preprocessing_num_workers "$NUM_WORKERS" \
    --output_dir "$OUTPUT_DIR" \
    --do_train \
    --do_eval \
    --do_pred \
    --do_clean \
    --do_summarize \
    --load_best_model_at_end \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --num_train_epochs 20 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.1 \
    --report_to "$WANDB" \
    --per_device_train_batch_size "$((GLOBAL_BATCH_SIZE / ACCUMULATION_STEPS))" \
    --per_device_eval_batch_size "$((GLOBAL_BATCH_SIZE / ACCUMULATION_STEPS))" \
    --gradient_accumulation_steps "$ACCUMULATION_STEPS" \
    --eval_accumulation_steps "$ACCUMULATION_STEPS" \
    "${FP16_ALL[@]}" \
    "${OVERWRITE[@]}" \
    "${NO_CUDA[@]}"
}

# declare global variable defaults
FP16_ALL=()
OVERWRITE=()
NO_CUDA=()
TASK="all"
OUTPUT_DIR="runs"
WANDB="none"
CUDA_VISIBLE_DEVICES=0
N_GPU=1
GLOBAL_BATCH_SIZE=16
ACCUMULATION_STEPS=1
NUM_WORKERS=1
MODEL_NAME_OR_PATH="bert-base-uncased"
PROGRAM_RUNTIME=("python3")

# overall workflow
parser "$@"
main
