#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
usage: prepare_git_hooks.sh [-h|--help]

Copy git hooks to git repository config

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

# define main function
main() {
  cp "./hooks/pre-commit" "./.git/hooks/"
}

# execute function
parser "$@"
main
