#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEPLOY_DIR="${ROOT}/deploy/huggingface"
STAGING="${ROOT}/.deploy/hf-space"
HF_USER="${HF_USER:-killvung}"
HF_SPACE="${HF_SPACE:-orbit-chase}"
HF_BRANCH="${HF_BRANCH:-main}"

if [[ -z "${HF_TOKEN:-}" ]] && command -v hf >/dev/null 2>&1; then
  HF_TOKEN="$(hf auth token 2>/dev/null || true)"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_REMOTE="https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${HF_USER}/${HF_SPACE}"
else
  HF_REMOTE="https://huggingface.co/spaces/${HF_USER}/${HF_SPACE}"
fi

log() {
  printf '==> %s\n' "$*"
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

require_cmd git
require_cmd npm
require_cmd rsync

cd "$ROOT"

log "Installing dependencies (npm ci)"
npm ci

log "Running production build"
NODE_ENV=production npm run build

[[ -f "${ROOT}/dist/index.html" ]] || fail "build did not produce dist/index.html"
[[ -f "${ROOT}/public/models/linear-sarsa-20260815-030554-ep8000.json" ]] \
  || fail "trained player model missing from public/models/"

mkdir -p "${ROOT}/.deploy"

if [[ -d "${STAGING}/.git" ]]; then
  log "Using Hugging Face staging clone"
  git -C "$STAGING" remote set-url origin "$HF_REMOTE"
else
  log "Cloning Hugging Face Space into ${STAGING}"
  rm -rf "$STAGING"
  if git clone --branch "$HF_BRANCH" "$HF_REMOTE" "$STAGING" 2>/dev/null; then
    :
  elif git clone "$HF_REMOTE" "$STAGING" 2>/dev/null; then
    git -C "$STAGING" checkout -b "$HF_BRANCH" 2>/dev/null \
      || git -C "$STAGING" checkout "$HF_BRANCH"
  else
    log "Space repo not found locally; initializing new staging repository"
    mkdir -p "$STAGING"
    git -C "$STAGING" init -b "$HF_BRANCH"
    git -C "$STAGING" remote add origin "$HF_REMOTE"
  fi
fi

log "Syncing production build (dist/)"
find "$STAGING" -mindepth 1 -maxdepth 1 \
  ! -name '.git' \
  ! -name '.gitattributes' \
  -exec rm -rf {} +

rsync -a --delete \
  "${ROOT}/dist/" \
  "${STAGING}/"

cp "${DEPLOY_DIR}/README.md" "${STAGING}/README.md"

cd "$STAGING"

git fetch origin "$HF_BRANCH" --quiet 2>/dev/null || true

HAS_CHANGES=false
if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git status --porcelain)" ]]; then
  HAS_CHANGES=true
  COMMIT_MSG="Deploy Orbit Chase $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  git add -A
  git commit -m "$COMMIT_MSG"
fi

AHEAD="$(git rev-list --count "origin/${HF_BRANCH}..HEAD" 2>/dev/null || echo 0)"
if [[ "$HAS_CHANGES" == false && "$AHEAD" == 0 ]]; then
  log "No changes to deploy"
  exit 0
fi

if [[ "$HAS_CHANGES" == false ]]; then
  log "No new file changes; pushing ${AHEAD} pending commit(s)"
fi

log "Pushing to https://huggingface.co/spaces/${HF_USER}/${HF_SPACE}"
if ! git push -u origin "$HF_BRANCH"; then
  cat >&2 <<EOF
error: git push failed.

Create the Space first at:
  https://huggingface.co/new-space

Use SDK: Static, name: ${HF_SPACE}
(no build step — this deploy pushes pre-built dist/ files)

Then authenticate with either:
  export HF_TOKEN=hf_...
  npm run deploy:hf

or:
  huggingface-cli login
  npm run deploy:hf
EOF
  exit 1
fi

log "Deploy complete"
log "Space URL: https://huggingface.co/spaces/${HF_USER}/${HF_SPACE}"
