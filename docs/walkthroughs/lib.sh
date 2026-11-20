# Shared helpers for the scripted terminal walkthroughs.
set -e
cd "$(git rev-parse --show-toplevel)"
say() { printf '\n\033[1;36m# %s\033[0m\n' "$*"; sleep 1.2; }
run() {
  printf '\033[1;32m$\033[0m '
  local text="$*" i
  for ((i = 0; i < ${#text}; i++)); do printf '%s' "${text:i:1}"; sleep 0.02; done
  printf '\n'; sleep 0.4
  eval "$@" || true
  sleep 1.6
}
