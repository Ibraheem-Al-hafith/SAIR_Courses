"""
UTILS — Module 5 importable utilities
======================================
This package provides the production-ready building blocks used by the
training and fine-tuning notebooks (4.TRAIN, A1.SFT_*, A2.SFT_*).

  from UTILS.model        import GPTModel
  from UTILS.generate     import generate, generate_text_simple
  from UTILS.load_weights import download_and_load_gpt2, load_weights_into_gpt
  from UTILS.finetune_utils import trainerV1

Note on config keys
-------------------
Notebooks that import from this package use the following config schema:
    "drop_rate"  — dropout probability (NOT "dropout")
    "n_heads"    — number of attention heads
    "emb_dim"    — embedding dimension
    "n_layers"   — number of transformer blocks
    "qkv_bias"   — whether QKV projections have a bias term

This is intentionally distinct from the standalone model.py at the module
root, which uses "dropout" for its self-contained teaching script.
"""
