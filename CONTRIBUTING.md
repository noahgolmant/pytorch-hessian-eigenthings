# Contributing

Thanks for your interest in contributing.

## Setup

```bash
git clone https://github.com/noahgolmant/pytorch-hessian-eigenthings
cd pytorch-hessian-eigenthings
uv sync --group dev --group docs --extra transformers --extra transformer-lens --extra curvlinops
```

## Before opening a PR

```bash
uv run ruff check .
uv run black --check .
uv run mypy
uv run pytest
uv run mkdocs build --strict
```

CI runs the same checks plus the `examples/` scripts and the docs codeblock tests. If lint or types fail, the lint job is the cheapest to debug locally.

## Adding a new operator or algorithm

- Operators subclass `CurvatureOperator` and call into `LinAlgBackend` for vector arithmetic. See `hessian_eigenthings/operators/hessian.py` and `docs/how-to/custom-curvature-operators.md`.
- Algorithms take any `CurvatureOperator` and use the backend exclusively. See `hessian_eigenthings/algorithms/lanczos.py`.
- Tests should validate against either a closed-form ground truth (e.g. the full Hessian on a tiny MLP via `torch.autograd.functional.hessian`) or against curvlinops as an external oracle. See `tests/test_hessian_operator.py` and `tests/cross_library/test_against_curvlinops.py`.

## Docs

- Concept pages explain *what* an algorithm computes and *when* to use it.
- How-to pages are recipes for specific user workflows.
- Reference pages are auto-generated from docstrings via mkdocstrings.

If you change a public API, update the matching concept or how-to page. The Claude Code hook in `.claude/hooks/` will remind you which page if you're using Claude Code.

## Reporting bugs

Open an issue with: PyTorch version, the operator and algorithm you used, a minimal reproducer if possible, and the full traceback. Closed-form expectations are appreciated when applicable.
