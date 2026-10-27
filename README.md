# NeuralHub notebooks

Public Colab notebooks and the minimal inference SDK for
[NeuralHub](https://mywalnuts.com/research/neuralhub/).

The application, authentication configuration and deployment code are not part of
this repository. Notebooks fetch public neuron recipes from the Hub at a pinned
recipe revision. Model weights are downloaded only when you run the experiment.

[Open the Unauthorized Frame example in Colab](https://colab.research.google.com/github/walnut-computing/neuralhub-notebooks/blob/main/notebooks/atlas/qwen35-unofficial/r1.ipynb)

## Run

Open a notebook under `notebooks/<owner>/<neuron>/r<revision>.ipynb` in Colab.
Choose a GPU runtime and run the cells in order. Settings are prefilled and editable.
`MODEL_REVISION="main"` resolves to an exact Hugging Face commit once per session and
prints it. T4 uses FP16 automatically; supported GPUs use native BF16. The default
operation is experimental raw `add`. Saved doses are displayed and preserved; mixed
doses use an editable per-coordinate list, and only missing doses default to experimental
+1.0. Approximate coordinates are clearly flagged and used as listed.

Source methods may normalize their doses (e.g. neuron-explorer residual-norm units);
the SDK uses absolute activation operands. Review this difference before interpreting
the result as a reproduction.
The last cell opens a Gradio chat playground. Every reply uses steering and retains conversation history. Adjust the strength multiplier, temperature, and output length in the UI. Thinking is disabled by default for templates that support it. The model is loaded once per chat launch. Use Clear to start a fresh conversation.

The full unquantized model must fit in memory. Free Colab hardware may be insufficient,
especially for MoE models. Restricted checkpoints require Hugging Face access.
Reported effects are experimental and are not a guarantee of reproduction.

## SDK

The notebook installs SDK 0.4.0 from the Hub's `/api/sdk/source.zip` endpoint (deploy SDK 0.4.0 before publishing these notebooks). Colab uses a temporary Gradio share link, accessible to anyone with the link while the runtime is active.
It does not require access to a private application repository. To install locally:

```sh
python -m pip install './sdk[transformers]'
```

Tests (run with the inference extra and Gradio installed):

```sh
python -m pip install 'gradio>=6,<7'
PYTHONPATH=sdk/src python -m unittest discover -s sdk/tests -v
```
