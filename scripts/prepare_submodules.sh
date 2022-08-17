#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
usage: prepare_submodules.sh [-h|--help]

This script instantiates necessary submodules and
extracts relevant data to the necessary locations

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

privacy_qa() {
  local tsv_file
  local target="./data/privacy_qa"
  mkdir -p "$target"

  # fetch all submodules
  git submodule update --init --recursive

  # copy relevant data
  for tsv_file in submodules/PrivacyQA_EMNLP/data/*.csv; do
    printf "%s\n" "Copying $tsv_file to $target"
    cp "$tsv_file" "$target/$(basename "$tsv_file" _data.csv).tsv"
  done

  # copy relevant data
  for tsv_file in \
    submodules/PrivacyQA_EMNLP/data/meta-annotations/"OPP-115 Annotations"/*.csv; do
    printf "%s\n" "Copying $tsv_file to $target"
    cp "$tsv_file" "$target/$(basename "$tsv_file" .csv).tsv"
  done
}

flatten_policy_qa() {
  local input_json_file="$1"
  cat <<EOF
import json

input_filename = "$input_json_file"
output_filename = "%sl" % input_filename

with open(input_filename) as input_file_stream:
    dataset = json.load(input_file_stream)

with open(output_filename, "w") as output_file_stream:
    for article in dataset["data"]:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            answers = {}
            for qa in paragraph["qas"]:
                question = qa["question"]
                question_type = qa["type"]
                idx = qa["id"]
                answers["text"] = [answer["text"] for answer in qa["answers"]]
                answers["answer_start"] = [
                    answer["answer_start"] for answer in qa["answers"]
                ]
                output_file_stream.write("%s\n" % json.dumps({
                    "id": idx,
                    "title": title,
                    "context": context,
                    "question": question,
                    "question_type": question_type,
                    "answers": answers
                }))
EOF
}

policy_qa() {
  local json_file target_json_file
  local target="./data/policy_qa"
  mkdir -p "$target"

  # fetch all submodules
  git submodule update --init --recursive

  # copy relevant data
  for json_file in submodules/PolicyQA/data/*.json; do
    printf "%s\n" "Copying $json_file to $target"
    target_json_file="$target/$(basename "$json_file")"
    cp "$json_file" "$target_json_file"
    python3 -c "$(flatten_policy_qa "$target_json_file")"
    rm -f "$target_json_file"
  done
}

policy_ie() {
  local target="./data/policy_ie"
  mkdir -p "$target"

  # fetch all submodules
  git submodule update --init --recursive

  # execute process in subshell
  (
    cd "submodules/PolicyIE/data/"
    bash prepare.sh || true
    git clean -f -d
  )

  # copy and unzip relevant data
  cp -r submodules/PolicyIE/data/bio_format/. "$target"
}

# define main function
main() {
  privacy_qa
  policy_qa
  policy_ie
}

# execute all functions
check_help "$@"
main
