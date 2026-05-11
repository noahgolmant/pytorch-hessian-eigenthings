#!/usr/bin/env bash
# Spin up a single A100-80GB GCP VM, rsync the current branch, run the
# GPU-marked GGN matvec validation tests, capture results, teardown.
#
# Usage:
#   ./scripts/gpu_validate.sh
#
# Requires: gcloud configured, project arcane-rigging-422722-h8 (or set PROJECT).

set -Eeuo pipefail

PROJECT="${PROJECT:-arcane-rigging-422722-h8}"
ZONE_CANDIDATES=("us-central1-a" "us-central1-b" "us-central1-c" "us-central1-f")
ZONE=""
INSTANCE_NAME="ggn-validate-$(date +%s)"

MACHINE_TYPE="a2-ultragpu-1g"
ACCELERATOR="type=nvidia-a100-80gb,count=1"
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2404-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_SIZE="100GB"
BOOT_DISK_TYPE="pd-balanced"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REMOTE_DIR="pytorch-hessian-eigenthings"

cleanup() {
  local exit_code=$?
  echo
  if [ -n "$ZONE" ]; then
    echo "=== cleanup: deleting instance $INSTANCE_NAME (zone $ZONE) ==="
    gcloud compute instances delete "$INSTANCE_NAME" \
      --project="$PROJECT" --zone="$ZONE" --quiet || true
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo "=== creating $INSTANCE_NAME ($MACHINE_TYPE, A100-80GB, project=$PROJECT) ==="
for try_zone in "${ZONE_CANDIDATES[@]}"; do
  echo "  trying $try_zone..."
  if gcloud compute instances create "$INSTANCE_NAME" \
      --project="$PROJECT" --zone="$try_zone" \
      --machine-type="$MACHINE_TYPE" \
      --accelerator="$ACCELERATOR" \
      --image-family="$IMAGE_FAMILY" \
      --image-project="$IMAGE_PROJECT" \
      --boot-disk-size="$BOOT_DISK_SIZE" \
      --boot-disk-type="$BOOT_DISK_TYPE" \
      --maintenance-policy=TERMINATE \
      --provisioning-model=STANDARD \
      --metadata=install-nvidia-driver=True \
      --scopes=https://www.googleapis.com/auth/cloud-platform 2>&1; then
    ZONE="$try_zone"
    echo "  succeeded in $ZONE"
    break
  else
    echo "  $try_zone unavailable, trying next..."
  fi
done
if [ -z "$ZONE" ]; then
  echo "ERROR: A100-80GB unavailable in all candidate zones" >&2
  exit 1
fi

echo "=== waiting for SSH ==="
until gcloud compute ssh "$INSTANCE_NAME" --project="$PROJECT" --zone="$ZONE" \
  --command="echo ssh-ready" --quiet 2>/dev/null; do
  sleep 5
done

echo "=== prepare remote dir ==="
gcloud compute ssh "$INSTANCE_NAME" --project="$PROJECT" --zone="$ZONE" \
  --command="mkdir -p ~/$REMOTE_DIR"

echo "=== rsync repo (excluding .venv, caches) ==="
gcloud compute scp --project="$PROJECT" --zone="$ZONE" --recurse \
  "$REPO_DIR/pyproject.toml" \
  "$REPO_DIR/uv.lock" \
  "$REPO_DIR/LICENSE" \
  "$REPO_DIR/README.md" \
  "$REPO_DIR/hessian_eigenthings" \
  "$REPO_DIR/tests" \
  "$REPO_DIR/scripts" \
  "$INSTANCE_NAME:~/$REMOTE_DIR/"

echo "=== bootstrap + run GPU validation tests ==="
gcloud compute ssh "$INSTANCE_NAME" --project="$PROJECT" --zone="$ZONE" --command="
  set -euo pipefail
  cd ~/$REMOTE_DIR
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=\"\$HOME/.local/bin:\$PATH\"
  fi
  export PATH=\"\$HOME/.local/bin:\$PATH\"
  export PYTHONUNBUFFERED=1
  uv sync --group dev --extra transformers
  uv run python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
  echo
  echo '=== running GPU validation tests ==='
  uv run pytest tests/test_ggn_matvec_a100.py -v -m gpu --tb=short 2>&1 | tee gpu_validate.log
  echo
  echo '=== exit code ==='
  echo \$?
"

echo "=== fetching log ==="
mkdir -p "$REPO_DIR/scripts/_validate_logs"
gcloud compute scp --project="$PROJECT" --zone="$ZONE" \
  "$INSTANCE_NAME:~/$REMOTE_DIR/gpu_validate.log" \
  "$REPO_DIR/scripts/_validate_logs/gpu_validate-$(date +%Y%m%d-%H%M%S).log" || true

echo "=== done; cleanup will tear down $INSTANCE_NAME ==="
