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
  wget -N -P "$target" \
    "https://privacypolicies.cs.princeton.edu/data-release/data/classifier_data.tar.gz"
  tar -zxvf "$target/classifier_data.tar.gz" -C "$target" \
    --strip-components 1 "dataset/1301_dataset.csv"
}

# define main function
main() {
  policy_detection
}

# execute all functions
parser "$@"
main
