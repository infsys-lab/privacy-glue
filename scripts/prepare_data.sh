#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
usage: prepare_data.sh [-h|--help]

This script downloads and prepares relevant data

optional arguments:
  -h, --help  <flag>
              show this help message and exit
EOF
}

# check for help
check_help() {
  for arg; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
      usage
      exit 0
    fi
  done
}

# download and prepare privacy policies
policy_detection() {
  local target="./data/policy_detection"
  mkdir -p "$target"
  wget -N -P "$target" "https://privacypolicies.cs.princeton.edu/data-release/data/classifier_data.tar.gz"
  tar -zxvf "$target/classifier_data.tar.gz" -C "$target" --strip-components 1 "dataset/1301_dataset.csv"
}

opp_115() {
  local target="./data/opp_115"
  mkdir -p "$target"
  wget -N -P "$target" "https://usableprivacy.org/static/data/OPP-115_v1_0.zip"
  wget -N -P "$target" "https://usableprivacy.org/static/data/JURIX_2020_OPP-115_GDPR_v1.0.zip"
  unzip -o "$target/OPP-115_v1_0.zip" -d "$target"
  unzip -o "$target/JURIX_2020_OPP-115_GDPR_v1.0.zip" -d "$target"
  rsync -a "$target/OPP-115/" "$target"
  rm -rf "$target/OPP-115"
}

app_350_transform() {
  cat <<EOF
from glob import glob
from tqdm import tqdm
import yaml
import json
import os

for file in tqdm(glob(os.path.join("$target", "annotations", "*.yml"))):
    with open(file, "r") as input_file_stream:
        data = yaml.safe_load(input_file_stream)
    with open("%s%s" % (os.path.splitext(file)[0], ".json"), "w") as output_file_stream:
        json.dump(data, output_file_stream)
    os.remove(file)
EOF
}

app_350() {
  local target="./data/app_350"
  mkdir -p "$target"
  wget -N -P "$target" "https://usableprivacy.org/static/data/APP-350_v1.1.zip"
  unzip -o "$target/APP-350_v1.1.zip" -d "$target"
  rsync -a "$target/APP-350_v1.1/" "$target"
  rm -rf "$target/APP-350_v1.1"
  python3 -c "$(app_350_transform)"
}

opt_out() {
  local target="./data/opt_out" query
  query=$(
    cat <<EOF
SELECT url, full_sentence_text, hyperlink_text, label
FROM hyperlinks
WHERE label IN ("Positive", "Negative");
EOF
  )
  mkdir -p "$target"
  wget -N -P "$target" "https://usableprivacy.org/static/data/OptOutChoice-2020_v1.0.zip"
  unzip -o "$target/OptOutChoice-2020_v1.0.zip" -d "$target"
  rsync -a "$target/OptOutChoice-2020_v1.0/" "$target"
  rm -rf "$target/OptOutChoice-2020_v1.0/"
  sqlite3 -json "$target/binary_data/policies.db" "$query" \
    >"$target/binary_data/binary_data.json"
  cat <<<"$(jq -c '.[]' "$target/binary_data/binary_data.json")" \
  >"$target/binary_data/binary_data.json"
  cat "$target/category_data/train_set1.jsonl" "$target/category_data/train_set2.jsonl" \
    "$target/category_data/train_set3.jsonl" >"$target/category_data/train_set.jsonl"
  rm -f "$target/category_data/train_set1.jsonl" "$target/category_data/train_set2.jsonl" \
    "$target/category_data/train_set3.jsonl"
}

clean() {
  find data -name ".DS_Store" -or -name ".idea" -or -name "__MACOSX" |
    xargs -n 1 rm -rf
}

# define main function
main() {
  policy_detection
  opp_115
  app_350
  opt_out
  clean
}

# execute all functions
check_help "$@"
main
