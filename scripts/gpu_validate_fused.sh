#!/usr/bin/env bash
# Spin up an A100-80GB GCP VM, rsync the feat/fused-ce-hvp branch, run the
# fused CE HVP tests (including the Triton path that skips without CUDA),
# run the microbenchmark for memory + wall-time numbers, capture results,
# teardown.

set -Eeuo pipefail

PROJECT="${PROJECT:?Set PROJECT to your GCP project ID, e.g. PROJECT=my-project bash scripts/gpu_validate_fused.sh}"
ZONE_CANDIDATES=(
  "us-central1-a" "us-central1-c"
  "us-east4-c" "us-east5-b"
  "us-west1-b" "us-west3-b" "us-west4-b"
  "europe-west4-a" "europe-west4-b"
  "asia-southeast1-c"
)
ZONE=""
INSTANCE_NAME="fused-validate-$(date +%s)"

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
    break
  fi
done
[ -z "$ZONE" ] && { echo "ERROR: A100-80GB unavailable in all zones" >&2; exit 1; }

echo "=== waiting for SSH ==="
until gcloud compute ssh "$INSTANCE_NAME" --project="$PROJECT" --zone="$ZONE" \
  --command="echo ssh-ready" --quiet 2>/dev/null; do
  sleep 5
done

echo "=== rsync repo ==="
gcloud compute ssh "$INSTANCE_NAME" --project="$PROJECT" --zone="$ZONE" \
  --command="mkdir -p ~/$REMOTE_DIR"
gcloud compute scp --project="$PROJECT" --zone="$ZONE" --recurse \
  "$REPO_DIR/pyproject.toml" \
  "$REPO_DIR/uv.lock" \
  "$REPO_DIR/LICENSE" \
  "$REPO_DIR/README.md" \
  "$REPO_DIR/hessian_eigenthings" \
  "$REPO_DIR/tests" \
  "$REPO_DIR/scripts" \
  "$INSTANCE_NAME:~/$REMOTE_DIR/"

echo "=== bootstrap + run fused CE HVP validation ==="
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
  uv pip install triton

  uv run python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
  uv run python -c 'import triton; print(\"triton\", triton.__version__)'
  echo

  echo '=== test_fused_ce_hvp (full suite incl. analytical-H, shape coverage, edge cases, fp16/bf16, GGN integration) ==='
  # Capture pytest exit code but don't abort: we want the benchmark numbers
  # even when a test fails so we can debug both signals in one run.
  set +e
  uv run pytest tests/test_fused_ce_hvp.py -v --tb=short 2>&1 | tee gpu_fused_test.log
  TEST_EXIT=\${PIPESTATUS[0]}
  set -e
  echo \"pytest exit code: \$TEST_EXIT\"
  echo

  echo '=== microbenchmark: headline shape (B=64,T=256,V=50304, fp32) ==='
  uv run python scripts/bench_fused_ce_hvp.py 2>&1 | tee gpu_fused_bench.log
  echo

  echo '=== microbenchmark: full sweep (V scaling, bf16, N scaling) ==='
  uv run python scripts/bench_fused_ce_hvp.py --full 2>&1 | tee gpu_fused_bench_full.log
  echo
  echo '=== exit code ==='
  echo \$?
"

echo "=== fetching logs ==="
mkdir -p "$REPO_DIR/scripts/_validate_logs"
TS="$(date +%Y%m%d-%H%M%S)"
gcloud compute scp --project="$PROJECT" --zone="$ZONE" \
  "$INSTANCE_NAME:~/$REMOTE_DIR/gpu_fused_test.log" \
  "$REPO_DIR/scripts/_validate_logs/gpu_fused_test-$TS.log" || true
gcloud compute scp --project="$PROJECT" --zone="$ZONE" \
  "$INSTANCE_NAME:~/$REMOTE_DIR/gpu_fused_bench.log" \
  "$REPO_DIR/scripts/_validate_logs/gpu_fused_bench-$TS.log" || true
gcloud compute scp --project="$PROJECT" --zone="$ZONE" \
  "$INSTANCE_NAME:~/$REMOTE_DIR/gpu_fused_bench_full.log" \
  "$REPO_DIR/scripts/_validate_logs/gpu_fused_bench_full-$TS.log" || true

echo "=== done; cleanup will tear down $INSTANCE_NAME ==="
