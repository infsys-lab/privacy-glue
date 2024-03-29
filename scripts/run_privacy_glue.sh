#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
usage: run_privacy_glue.sh [option...]

optional arguments:
  --cuda_visible_devices       <str>
                               comma separated string of integers passed
                               directly to the "CUDA_VISIBLE_DEVICES"
                               environment variable
                               (default: 0)

  --fp16                       enable 16-bit mixed precision computation
                               through NVIDIA Apex for training
                               (default: False)

  --model_name_or_path         <str>
                               model to be used for fine-tuning. Currently only
                               the following are supported:
                               "bert-base-uncased",
                               "roberta-base",
                               "nlpaueb/legal-bert-base-uncased",
                               "saibo/legal-roberta-base",
                               "mukund/privbert"
                               (default: bert-base-uncased)

  --no_cuda                    disable CUDA even when available (default: False)

  --overwrite_cache            overwrite caches used in preprocessing
                               (default: False)

  --overwrite_output_dir       overwrite run directories and saved checkpoint(s)
                               (default: False)

  --preprocessing_num_workers  <int>
                               number of workers to be used for preprocessing
                               (default: None)

  --task                       <str>
                               task to be worked on. The following values are
                               accepted: "opp_115", "piextract",
                               "policy_detection", "policy_ie_a", "policy_ie_b",
                               "policy_qa", "privacy_qa", "all"
                               (default: all)

  --wandb                      log metrics and results to wandb
                               (default: False)

  -h, --help                   show this help message and exit
EOF
}

parser() {
  while [[ -n "$1" ]]; do
    case "$1" in
    --fp16)
      FP16=("--fp16")
      ;;
    --overwrite_output_dir)
      OVERWRITE_OUTPUT_DIR=("--overwrite_output_dir")
      ;;
    --overwrite_cache)
      OVERWRITE_CACHE=("--overwrite_cache")
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
    --preprocessing_num_workers)
      if [[ -n "$2" ]]; then
        shift
        PREPROCESSING_NUM_WORKERS=("--preprocessing_num_workers" "$1")
      else
        {
          printf "%s\n\n" "Missing --preprocessing_num_workers argument"
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
    --output_dir "$OUTPUT_DIR" \
    --do_train \
    --do_eval \
    --do_pred \
    --do_clean \
    --do_summarize \
    --load_best_model_at_end \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --logging_steps 100 \
    --save_total_limit 2 \
    --num_train_epochs 20 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.1 \
    --early_stopping_patience 5 \
    --report_to "$WANDB" \
    --full_determinism \
    --per_device_train_batch_size "$DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size "$DEVICE_BATCH_SIZE" \
    "${PREPROCESSING_NUM_WORKERS[@]}" \
    "${FP16[@]}" \
    "${OVERWRITE_OUTPUT_DIR[@]}" \
    "${OVERWRITE_CACHE[@]}" \
    "${NO_CUDA[@]}"
}

# declare global variable defaults
FP16=()
OVERWRITE_OUTPUT_DIR=()
OVERWRITE_CACHE=()
NO_CUDA=()
PREPROCESSING_NUM_WORKERS=()
TASK="all"
OUTPUT_DIR="runs"
WANDB="none"
CUDA_VISIBLE_DEVICES=0
N_GPU=1
DEVICE_BATCH_SIZE=16
MODEL_NAME_OR_PATH="bert-base-uncased"
PROGRAM_RUNTIME=("python3")

# overall workflow
parser "$@"
main
