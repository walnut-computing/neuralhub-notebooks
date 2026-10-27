# NeuralHub inference SDK

Downloads public neuron recipes and runs explicit local steering experiments.
Install from this repository checkout with `python -m pip install "./sdk[transformers]"`.
Python 3.10+, an unquantized supported Transformers model, and enough memory for the
full checkpoint are required. Model weights are not included.

Supported text architectures: Qwen3.5, Qwen3.8 Flash (routed/shared experts), OLMo3,
Gemma4, Qwen3 and Llama. Interventions target the input to `mlp.down_proj`, after
activation and gating, on prompt and generated tokens. `add`, `scale` and `set`
are experimental operations; confirm the original method, checkpoint and dose.
Approximate coordinates require explicit opt-in. Quantized and compiled models
are not supported. Calls sharing a model must be serialized.

Recipe downloads are anonymous and cached locally for up to five minutes by default.
Restricted Hugging Face checkpoints may require separate authentication. Model files
are cached independently by Hugging Face. Hooks are removed after each intervention.
Full pretrained behavioral effects have not been independently reproduced.
