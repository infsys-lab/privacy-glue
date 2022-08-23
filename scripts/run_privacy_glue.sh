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

  --fp_16                 enable 16-bit mixed precision computation
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

  --overwrite             overwrite cached data and saved checkpoint(s)
                          (default: False)

  --task                  <str>
                          task to be worked on. The following values are
                          accepted: "app_350", "opp_115", "piextract",
                          "policy_detection", "policy_ie", "policy_qa",
                          "privacy_qa", "all"
                          (default: all)

  -h, --help              show this help message and exit
EOF
}

parser() {
  while [[ -n "$1" ]]; do
    case "$1" in
    --fp_16)
      FP_16=("--fp16" "--fp16_full_eval")
      ;;
    --overwrite)
      OVERWRITE=("--overwrite_cache" "--overwrite_output_dir")
      ;;
    --model_name_or_path)
      shift
      MODEL_NAME_OR_PATH="$1"
      ;;
    --task)
      shift
      TASK="$1"
      ;;
    --cuda_visible_devices)
      shift
      CUDA_VISIBLE_DEVICES="$1"
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      printf "%s\n" "Unknown option $1"
      usage
      exit 1
      ;;
    esac
    shift
  done
}

main() {
  # execute fine-tuning
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python3 src/privacy_glue.py \
    --task "$TASK" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --do_train \
    --do_eval \
    --do_pred \
    --do_clean \
    --do_summarize \
    --metric_for_best_model "macro-f1" \
    --greater_is_better "True" \
    --load_best_model_at_end \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_total_limit 5 \
    --num_train_epochs 20 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size "$((GLOBAL_BATCH_SIZE / ACCUMULATION_STEPS))" \
    --per_device_eval_batch_size "$((GLOBAL_BATCH_SIZE / ACCUMULATION_STEPS))" \
    --gradient_accumulation_steps "$ACCUMULATION_STEPS" \
    --eval_accumulation_steps "$ACCUMULATION_STEPS" \
    "${FP_16[@]}" \
    "${OVERWRITE[@]}"
}

# declare global variable defaults
FP_16=()
OVERWRITE=()
TASK="all"
OUTPUT_DIR="runs"
CUDA_VISIBLE_DEVICES=0
GLOBAL_BATCH_SIZE=8
ACCUMULATION_STEPS=1
MODEL_NAME_OR_PATH="bert-base-uncased"

# overall workflow
parser "$@"
main
