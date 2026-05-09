# Examples

Self-contained scripts demonstrating common usage patterns.

| File                                     | What it shows                                                                       | Extras needed         |
|------------------------------------------|-------------------------------------------------------------------------------------|-----------------------|
| `supervised_mlp.py`                      | Top-k eigenvalues, Hutch++ trace, SLQ density on a small MLP with synthetic data   | none                  |
| `huggingface_tiny_gpt2.py`               | Hessian of a HuggingFace causal LM; full model and attention-only restriction      | `transformers`        |
| `transformer_lens_attention_only.py`     | TransformerLens model with per-block (attention vs MLP) Hessian analysis           | `transformer-lens`    |

Run any of them with uv:

```bash
uv run python examples/supervised_mlp.py
uv run --extra transformers python examples/huggingface_tiny_gpt2.py
uv run --extra transformer-lens python examples/transformer_lens_attention_only.py
```
