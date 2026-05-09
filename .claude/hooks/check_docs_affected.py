"""PostToolUse hook for Claude Code.

When a source file under hessian_eigenthings/ is edited, print a reminder to
stderr listing the related docs page(s). The reminder appears in Claude's
context, nudging it to keep the docs in sync. Soft-only — does not block tools.

Reads the tool input as JSON on stdin (Claude Code's hook protocol).
"""

from __future__ import annotations

import json
import os
import sys

# Map source files (paths relative to repo root) to canonical docs pages.
DOCS_MAP: dict[str, list[str]] = {
    "hessian_eigenthings/operators/hessian.py": [
        "docs/concepts/what-is-the-hessian.md",
        "docs/concepts/why-hvp-not-full-h.md",
    ],
    "hessian_eigenthings/operators/ggn.py": [
        "docs/concepts/ggn-vs-fisher-vs-hessian.md",
    ],
    "hessian_eigenthings/operators/fisher.py": [
        "docs/concepts/ggn-vs-fisher-vs-hessian.md",
    ],
    "hessian_eigenthings/operators/distributed/ddp.py": [
        "docs/how-to/distributed-ddp.md",
    ],
    "hessian_eigenthings/algorithms/lanczos.py": [
        "docs/concepts/top-k-eigenvalues.md",
    ],
    "hessian_eigenthings/algorithms/power_iteration.py": [
        "docs/concepts/top-k-eigenvalues.md",
    ],
    "hessian_eigenthings/algorithms/trace.py": [
        "docs/concepts/trace-estimation.md",
    ],
    "hessian_eigenthings/algorithms/spectral_density.py": [
        "docs/concepts/spectral-density.md",
    ],
    "hessian_eigenthings/loss_fns/standard.py": [
        "docs/how-to/custom-loss-functions.md",
    ],
    "hessian_eigenthings/loss_fns/huggingface.py": [
        "docs/how-to/analyze-a-huggingface-model.md",
    ],
    "hessian_eigenthings/loss_fns/transformer_lens.py": [
        "docs/how-to/analyze-with-transformer-lens.md",
    ],
    "hessian_eigenthings/param_utils.py": [
        "docs/how-to/per-layer-hessian.md",
    ],
}


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    file_path = (payload.get("tool_input") or {}).get("file_path", "")
    if not file_path:
        return 0

    cwd = os.getcwd()
    rel_path = file_path[len(cwd) + 1 :] if file_path.startswith(cwd) else file_path

    related = DOCS_MAP.get(rel_path)
    if not related:
        return 0

    print(f"[doc-drift hook] Edited {rel_path}. Related docs:", file=sys.stderr)
    for doc in related:
        print(f"  - {doc}", file=sys.stderr)
    print(
        "[doc-drift hook] If the public API or behavior changed, update those pages.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
