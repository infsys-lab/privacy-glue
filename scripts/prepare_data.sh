#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
usage: prepare_data.sh [-h|--help]

This script downloads and prepares relevant data

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

# download and prepare privacy policies
policy_detection() {
  local target="./data/policy_detection"
  mkdir -p "$target"
  wget -N -P "$target" "https://privacypolicies.cs.princeton.edu/data-release/data/classifier_data.tar.gz"
  tar -zxvf "$target/classifier_data.tar.gz" -C "$target" --strip-components 1 "dataset/1301_dataset.csv"
}

app_350() {
  local target="./data/app_350"
  mkdir -p "$target"
  wget -N -P "$target" "https://usableprivacy.org/static/data/APP-350_v1.1.zip"
  unzip -o "$target/APP-350_v1.1.zip" -d "$target"
  rsync -a "$target/APP-350_v1.1/" "$target"
  rm -rf "$target/APP-350_v1.1"
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
  app_350
  opt_out
  clean
}

# execute all functions
parser "$@"
main
