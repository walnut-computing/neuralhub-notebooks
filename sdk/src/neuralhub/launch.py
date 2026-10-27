"""Generate ready-to-run experimental notebooks without loading models on the Hub."""

import math
import shlex

from .hub import HubClient



def defaults(document):
    positions = document["positions"]
    missing = [i for i, p in enumerate(positions) if p.get("strength") is None]
    doses = [1.0 if i in missing else p["strength"] for i, p in enumerate(positions)]
    for dose in doses:
        if type(dose) not in (int, float) or not math.isfinite(dose) or abs(dose) > 100:
            raise ValueError("Recipe strengths must be finite numbers between -100 and 100")
    uniform = bool(doses) and len(set(doses)) == 1
    return {
        "strength": doses[0] if uniform else "per_coordinate",
        "doses": doses,
        "missing": missing,
        "approximate": any(p.get("coordinatePrecision") == "approximate" for p in positions),
    }


def experiment(document, base_url):
    HubClient(base_url)
    HubClient.validate(document, document["id"], document["revision"])
    settings = defaults(document)
    return f'''from neuralhub import Neuron, Model

HUB_URL = {base_url.rstrip('/')!r}
REPOSITORY = {document['id']!r}
REVISION = {document['revision']!r}
MODEL_REPOSITORY = {document['model'].get('huggingface')!r}

# Ready-to-run exploratory settings. Edit these to try another experiment.
MODEL_REVISION = "main"  # Resolved once to an exact HF commit before loading.
OPERATION = "add"        # Experimental raw addition at mlp.down_proj.input.
STRENGTH = {settings['strength']!r}  # Saved dose, or "per_coordinate" for the list below.
COORDINATE_STRENGTHS = {settings['doses']!r}  # Coordinate order matches the pinned recipe.
MISSING_DOSE_COORDINATES = {settings['missing']!r}  # Missing doses default to experimental +1.0.
ALLOW_APPROXIMATE = {settings['approximate']!r}  # Uses the listed approximate coordinate when True.
DEVICE = "auto"  # CUDA GPU; set "cpu" explicitly for slow local CPU execution.
DTYPE = "auto"   # Native BF16 on supported GPUs, FP16 on T4, FP32 on CPU.
PROMPT = "Tell me a short story."
MAX_NEW_TOKENS = 128


def runtime_settings():
    import torch
    device = DEVICE
    if device == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("Choose a GPU in Colab: Runtime > Change runtime type > T4 GPU, then rerun. For local CPU execution set DEVICE = 'cpu'.")
        device = "cuda"
    if DTYPE == "auto":
        if str(device).startswith("cuda"):
            native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if native_bf16 else torch.float16
        else:
            dtype = torch.float32
    else:
        if DTYPE not in ("float16", "bfloat16", "float32"):
            raise ValueError("DTYPE must be auto, float16, bfloat16 or float32")
        dtype = getattr(torch, DTYPE)
    return device, dtype


def compare():
    import math
    import re
    from huggingface_hub import model_info
    global MODEL_REVISION
    if not isinstance(MODEL_REVISION, str) or not MODEL_REVISION:
        raise ValueError("MODEL_REVISION must be a model branch, tag or exact commit")
    if OPERATION not in ("add", "scale", "set"):
        raise ValueError("OPERATION must be add, scale or set")
    doses = COORDINATE_STRENGTHS if STRENGTH == "per_coordinate" else [STRENGTH]
    if not doses or any(type(d) not in (int, float) or not math.isfinite(d) or abs(d) > 100 for d in doses):
        raise ValueError("Strengths must be finite numbers between -100 and 100")
    device, dtype = runtime_settings()
    neuron = Neuron.from_hub(REPOSITORY, base_url=HUB_URL, revision=REVISION)
    if neuron.document["model"].get("huggingface") != MODEL_REPOSITORY:
        raise ValueError("The recipe model differs from the exported notebook")
    if not ALLOW_APPROXIMATE and any(p.get("coordinatePrecision") == "approximate" for p in neuron.document["positions"]):
        raise ValueError("This recipe uses approximate coordinates; set ALLOW_APPROXIMATE = True to use them as listed.")
    if STRENGTH == "per_coordinate":
        if len(COORDINATE_STRENGTHS) != len(neuron.document["positions"]):
            raise ValueError("Provide one dose for every recipe coordinate")
        neuron = Neuron({{**neuron.document, "positions": [
            {{**p, "strength": dose}} for p, dose in zip(neuron.document["positions"], COORDINATE_STRENGTHS)
        ]}})
    if neuron.document.get("format") == "neuralhub/evidence-v1":
        # Explicit exploratory raw operands, not a conversion of source units.
        neuron = neuron.prepare(
            positions=neuron.document["positions"] if STRENGTH == "per_coordinate" else None,
            strength=None if STRENGTH == "per_coordinate" else STRENGTH)
    if not re.fullmatch(r"[0-9a-fA-F]{{40}}", MODEL_REVISION):
        resolved = model_info(MODEL_REPOSITORY, revision=MODEL_REVISION).sha
        if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-fA-F]{{40}}", resolved):
            raise ValueError("Hugging Face did not return an exact model commit")
        MODEL_REVISION = resolved  # Reused for subsequent runs in this session.
    print(f"Model: {{MODEL_REPOSITORY}} @ {{MODEL_REVISION}}\\nDevice: {{device}}; dtype: {{dtype}}")
    print(f"Recipe: {{REPOSITORY}} r{{REVISION}}; operation: {{OPERATION}}; strengths: {{doses}}")
    print("Experimental settings: the current model checkpoint and raw activation doses do not establish reproduction of the source method or effect.")
    if MISSING_DOSE_COORDINATES:
        print(f"Missing source doses at coordinate indices {{MISSING_DOSE_COORDINATES}} were prefilled with experimental +1.0; review before interpreting results.")
    if ALLOW_APPROXIMATE:
        print("Approximate coordinates are enabled and will be used as listed.")
    model = Model.from_neuron(neuron, model_revision=MODEL_REVISION, device=device, dtype=dtype, attn_implementation="sdpa")
    baseline = model.generate(PROMPT, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
    with model.steer(neuron, operation=OPERATION,
                     strength=None if STRENGTH == "per_coordinate" else STRENGTH,
                     allow_approximate=ALLOW_APPROXIMATE):
        steered = model.generate(PROMPT, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
    print("BASELINE\\n" + baseline + "\\n\\nSTEERED\\n" + steered)
    return {{"baseline": baseline, "steered": steered, "model_revision": MODEL_REVISION}}
'''


def notebook(document, base_url):
    def cell(kind, source):
        result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
        if kind == "code":
            result.update(execution_count=None, outputs=[])
        return result

    settings = defaults(document)
    install = "neuralhub-sdk[transformers] @ " + base_url.rstrip("/") + "/api/sdk/source.zip"
    notes = ""
    if settings["missing"]:
        notes += "\n\n**Missing source doses:** prefilled with experimental +1.0; saved doses are preserved."
    if settings["approximate"]:
        notes += "\n\n**Approximate coordinates:** enabled and used as listed; edit ALLOW_APPROXIMATE to disable."
    return {
        "nbformat": 4, "nbformat_minor": 4,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "accelerator": "GPU", "colab": {"name": document["id"].replace("/", "-") + ".ipynb"}},
        "cells": [
            cell("markdown", "# NeuralHub experiment: " + document["id"] + "\n"
                 "Select **Runtime > Change runtime type > T4 GPU**, then run the cells in order. "
                 "Settings are prefilled and editable. T4 automatically uses FP16; GPUs with native BF16 use BF16. "
                 "The full unquantized model must fit in memory; larger or MoE models may not fit T4. Restricted checkpoints need Hugging Face login.\n\n"
                 "**Experimental defaults, not verified reproduction:** the current model branch is resolved to an exact commit once per session and printed. "
                 "The default operation is raw activation addition. Saved doses are shown below, but source methods may use different normalization units "
                 "(including the neuron-explorer residual-norm unit); this SDK uses absolute activation operands. Confirm the original method before interpreting effects."
                 + notes),
            cell("code", "%pip install " + shlex.quote(install)),
            cell("code", experiment(document, base_url)),
            cell("code", "results = compare()"),
        ],
    }
