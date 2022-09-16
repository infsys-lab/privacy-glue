#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
usage: prepare.sh [-h|--help]

This script clones git submodules, downloads upstream data
and finally extracts data to the relevant locations

optional arguments:
  -h, --help  show this help message and exit
EOF
}

# check for help
parser() {
  for arg; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
      usage
      exit 0
    fi
  done
}

policy_detection() {
  local xz_file target="./data/policy_detection"
  mkdir -p "$target"

  # fetch submodule
  git submodule update --init --recursive "submodules/policy-detection-data"

  # uncompress compressed tar archive
  for xz_file in submodules/policy-detection-data/data/*.xz; do
    printf "%s\n" "Decompressing $xz_file to $target"
    xz -dc "$xz_file" >"$target/$(basename "$xz_file" .xz)"
  done
}

privacy_qa() {
  local tsv_file
  local target="./data/privacy_qa"
  mkdir -p "$target"

  # fetch submodule
  git submodule update --init --recursive "submodules/PrivacyQA_EMNLP"

  # copy relevant data
  for tsv_file in "submodules/PrivacyQA_EMNLP/data/"*.csv; do
    printf "%s\n" "Copying $tsv_file to $target"
    cp "$tsv_file" "$target/$(basename "$tsv_file" _data.csv).tsv"
  done

  # copy relevant data
  for tsv_file in \
    "submodules/PrivacyQA_EMNLP/data/meta-annotations/OPP-115 Annotations/"*.csv; do
    printf "%s\n" "Copying $tsv_file to $target"
    cp "$tsv_file" "$target/$(basename "$tsv_file" .csv).tsv"
  done
}

policy_qa() {
  local json_file target_json_file
  local target="./data/policy_qa"
  mkdir -p "$target"

  # fetch submodule
  git submodule update --init --recursive "submodules/PolicyQA"

  # copy relevant data
  for json_file in "submodules/PolicyQA/data/"*.json; do
    printf "%s\n" "Copying $json_file to $target"
    target_json_file="$target/$(basename "$json_file")"
    cp "$json_file" "$target_json_file"
  done
}

policy_ie() {
  local target_a="./data/policy_ie_a"
  local target_b="./data/policy_ie_b"
  local source="submodules/policy-ie/data/bio_format"
  mkdir -p "$target_a" "$target_b"

  # fetch submodule
  git submodule update --init --recursive "submodules/policy-ie"

  # execute process in subshell
  (
    cd "submodules/policy-ie/data/"
    bash prepare.sh || true
    git clean -f -d
  )

  # copy metadata from source to target
  cp "$source/vocab.txt" "$source/intent_label.txt" "$target_a"
  cp "$source/vocab.txt" "$source/type_I_slot_label.txt" \
    "$source/type_II_slot_label.txt" "$target_b"

  # copy annotation data from source to target
  for inner_source in "$source/"*/; do
    inner_target_a="$target_a/$(basename "$inner_source")"
    inner_target_b="$target_b/$(basename "$inner_source")"
    mkdir -p "$inner_target_a"
    mkdir -p "$inner_target_b"
    cp "$inner_source/seq.in" "$inner_source/label" "$inner_target_a"
    cp "$inner_source/seq.in" "$inner_source/seq_type_I.out" \
      "$inner_source/seq_type_II.out" "$inner_target_b"
  done
}

opp_115() {
  local target="./data/opp_115"
  mkdir -p "$target"

  # fetch submodule
  git submodule update --init --recursive "submodules/Polisis_Benchmark"

  # copy relevant data
  for csv_file in "submodules/Polisis_Benchmark/datasets/Majority/"*.csv; do
    printf "%s\n" "Copying $csv_file to $target"
    target_csv_file="$target/$(basename "$csv_file")"
    cp "$csv_file" "$target_csv_file"
  done
}

piextract() {
  local target="./data/piextract"
  mkdir -p "$target"

  # fetch submodule
  git submodule update --init --recursive "submodules/piextract_dataset"

  # copy relevant data
  printf "%s\n" "Copying CONLL data to $target"
  find "submodules/piextract_dataset/dataset" -mindepth 1 \
    -type d -exec cp -r {} "$target" \;
}

# define main function
main() {
  policy_detection
  privacy_qa
  policy_qa
  policy_ie
  opp_115
  piextract
}

# execute all functions
parser "$@"
main
