# NeuralHub notebooks

Public Colab notebooks and the minimal inference SDK for
[NeuralHub](https://mywalnuts.com/research/neuralhub/).

The application, authentication configuration and deployment code are not part of
this repository. Notebooks fetch public neuron recipes from the Hub at a pinned
recipe revision. Model weights are downloaded only when you run the experiment.

[Open the Unauthorized Frame example in Colab](https://colab.research.google.com/github/walnut-computing/neuralhub-notebooks/blob/main/notebooks/atlas/qwen35-unofficial/r1.ipynb)

## Run

Open a notebook under `notebooks/<owner>/<neuron>/r<revision>.ipynb` in Colab.
Choose a GPU runtime and run **install → select neuron → launch playground**.
The SDK loads the model once, resolves the source operation and calibrates normalized
doses before opening chat. Adjust strength, steering on/off, temperature and response
length in the UI. Expand diagnostics to see the checkpoint, source doses and calibration.

All 299 notebooks use this flow. 119 recipes have source-method ports; 180 explain
missing source settings/backend support and stop before downloading model weights.
There is no guessed +1 strength or blanket raw-add fallback. Source MLX/GGUF engines
and quantization differ from this Transformers runtime; effects are not independently
reproduced. See [the source-method matrix](sdk/README.md#source-method-chat-playground).

The full unquantized model must fit in memory. Free Colab hardware may be insufficient,
especially for MoE models. Restricted checkpoints require Hugging Face access.
Reported effects are experimental and are not a guarantee of reproduction.

## SDK

The notebook installs SDK 0.5.0 from the Hub's `/api/sdk/source.zip` endpoint (deploy SDK 0.5.0 before publishing these notebooks). Colab uses a temporary Gradio share link, accessible to anyone with the link while the runtime is active.
It does not require access to a private application repository. To install locally:

```sh
python -m pip install './sdk[transformers]'
```

Tests (run with the inference extra and Gradio installed):

```sh
python -m pip install 'gradio>=6,<7'
PYTHONPATH=sdk/src python -m unittest discover -s sdk/tests -v
```
