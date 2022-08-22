#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
usage: prepare_submodules.sh [-h|--help]

This script instantiates necessary submodules and
extracts relevant data to the necessary locations

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
  local target="./data/policy_ie"
  mkdir -p "$target"

  # fetch submodule
  git submodule update --init --recursive "submodules/policy-ie"

  # execute process in subshell
  (
    cd "submodules/policy-ie/data/"
    bash prepare.sh || true
    git clean -f -d
  )

  # copy and unzip relevant data
  cp -r "submodules/policy-ie/data/bio_format/." "$target"
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
  privacy_qa
  policy_qa
  policy_ie
  opp_115
  piextract
}

# execute all functions
parser "$@"
main
