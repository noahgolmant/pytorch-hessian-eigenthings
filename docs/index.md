# hessian-eigenthings

Iterative eigendecomposition of curvature operators (Hessian, GGN, empirical Fisher) for arbitrary PyTorch models, including HuggingFace and TransformerLens transformers. Top eigenvalues via Lanczos or power iteration, trace via Hutch++, and the spectral density via Stochastic Lanczos Quadrature, all matrix-free.

The full Hessian of a model with $n$ parameters costs $O(n^2)$ memory, infeasible past toy scale. Iterative methods only need Hessian-vector products, which cost $O(n)$. That's what this library is built around.

## Where to start

- **[Quickstart](getting-started/quickstart.md)**: 30-second example on a small MLP.
- **[Transformers quickstart](getting-started/transformers-quickstart.md)**: same shape, on a HuggingFace causal LM.
- **[Concepts](concepts/what-is-the-hessian.md)**: what each algorithm computes and when to pick which. The [GGN vs Fisher vs Hessian](concepts/ggn-vs-fisher-vs-hessian.md) page is worth reading early since they're easy to conflate.
- **[How-to recipes](how-to/analyze-a-huggingface-model.md)**: task-oriented walkthroughs.
- **[API reference](reference/api.md)**: every public symbol.

## Install

```bash
pip install hessian-eigenthings
pip install "hessian-eigenthings[transformers,transformer-lens]"   # with helpers
```

Source on GitHub: <https://github.com/noahgolmant/pytorch-hessian-eigenthings>.
