# NeuralHub notebooks

Public Colab notebooks and the minimal inference SDK for
[NeuralHub](https://mywalnuts.com/research/neuralhub/).

The application, authentication configuration and deployment code are not part of
this repository. Notebooks fetch public neuron recipes from the Hub at a pinned
recipe revision. Model weights are downloaded only when you run the experiment.

## Run

Open a notebook under `notebooks/<owner>/<neuron>/r<revision>.ipynb` in Colab.
Choose a GPU runtime and set `MODEL_REVISION` to an exact Hugging Face model commit,
then choose `OPERATION` and any missing `STRENGTH`. Review the discovery method first.
The notebook compares baseline and steered output for the same prompt.

The full unquantized model must fit in memory. Free Colab hardware may be insufficient,
especially for MoE models. Restricted checkpoints require Hugging Face access.
Reported effects are experimental and are not a guarantee of reproduction.

## SDK

The notebook installs the SDK from an immutable commit in this public repository.
It does not require access to a private application repository. To install locally:

```sh
python -m pip install './sdk[transformers]'
```

Tests (run with the inference extra installed):

```sh
PYTHONPATH=sdk/src python -m unittest discover -s sdk/tests -v
```
