"""Explicit, reversible interventions on eager Transformers text models."""

import math
from contextlib import contextmanager
from dataclasses import dataclass

from .hub import HubClient


@dataclass
class Neuron:
    document: dict

    @classmethod
    def from_hub(
        cls,
        repo,
        *,
        base_url,
        revision=None,
        offline=False,
        refresh=False,
        cache_dir=None,
    ):
        return cls(
            HubClient(base_url, cache_dir=cache_dir).neuron(
                repo, revision=revision, offline=offline, refresh=refresh
            )
        )


SUPPORTED = {"qwen3", "llama", "olmo3", "qwen3_5_text", "qwen4_exp_text", "gemma4_text"}
CONDITIONAL = {
    "qwen3_5": "Qwen3_5ForConditionalGeneration",
    "qwen4_exp": "Qwen4ExpForConditionalGeneration",
    "gemma4": "Gemma4ForConditionalGeneration",
}


def text_config(config):
    return getattr(config, "text_config", config)


def decoder(model):
    config = text_config(model.config)
    if config.model_type not in SUPPORTED:
        raise ValueError(f"Unsupported text architecture: {config.model_type}")
    base = model.model
    return getattr(base, "language_model", base)


def operand(value, operation, strength):
    if operation == "add":
        return value + strength
    if operation == "scale":
        return value * strength
    return value * 0 + strength


def expert_hook(edits, operation):
    # Packed experts use functional linear operations, so there is no per-expert
    # down_proj module to hook. Add the projected activation delta to its output,
    # preserving the original router selection and weights (including zero routes).
    def hook(module, args, kwargs, output):
        import torch
        import torch.nn.functional as F

        hidden = args[0] if args else kwargs["hidden_states"]
        selected = args[1] if len(args) > 1 else kwargs["top_k_index"]
        weights = args[2] if len(args) > 2 else kwargs["top_k_weights"]
        result = output.clone()
        for expert, coordinates in edits.items():
            tokens, slots = torch.where(selected == expert)
            if not tokens.numel():
                continue
            gate, up = F.linear(hidden[tokens], module.gate_up_proj[expert]).chunk(
                2, dim=-1
            )
            activation = module.act_fn(gate) * up
            indices = list(coordinates)
            delta = torch.stack(
                [
                    operand(activation[:, i], operation, coordinates[i])
                    - activation[:, i]
                    for i in indices
                ],
                dim=-1,
            )
            change = F.linear(delta, module.down_proj[expert][:, indices])
            change = change * weights[tokens, slots, None]
            result.index_add_(0, tokens, change.to(result.dtype))
        return result

    return hook


@contextmanager
def steer(
    model,
    neuron,
    *,
    operation,
    strength=None,
    site="mlp.down_proj.input",
    allow_approximate=False,
):
    """Change the post-gating MLP vector, all tokens in prefill and decode.

    Layers and neurons are zero-based. Strength is an absolute operand, not a
    multiplier on the recipe's dose. Calls sharing a model must be serialized.
    """
    if operation not in ("add", "scale", "set"):
        raise ValueError("operation must be add, scale or set")
    if site != "mlp.down_proj.input":
        raise ValueError("Only mlp.down_proj.input is supported")
    layers = decoder(model).layers
    if hasattr(model, "_orig_mod"):
        raise ValueError("Compiled models are not supported in this version")
    if getattr(model, "is_quantized", False):
        raise ValueError("Quantized models are not supported in this version")
    if getattr(model, "_neuralhub_active", False):
        raise ValueError(
            "Overlapping steering contexts on the same model are not supported"
        )
    positions = neuron.document.get("positions", [])
    if not positions:
        raise ValueError("This recipe has no executable coordinates")
    targets = {}
    experts = {}
    for position in positions:
        if (
            position.get("coordinatePrecision") == "approximate"
            and not allow_approximate
        ):
            raise ValueError("Approximate coordinates require allow_approximate=True")
        layer, index = position.get("layer"), position.get("neuron")
        if type(layer) is not int or type(index) is not int or layer < 0 or index < 0:
            raise ValueError("Coordinates must be non-negative zero-based integers")
        if layer >= len(layers):
            raise ValueError("Layer coordinate exceeds model depth")
        mlp = layers[layer].mlp
        expert, shared = position.get("expert"), position.get("shared", False)
        if type(shared) is not bool or (shared and expert is not None):
            raise ValueError("Specify either a routed expert or shared expert")
        if hasattr(mlp, "experts"):
            if shared:
                module = mlp.shared_expert.down_proj
                edits = targets.setdefault(module, {})
                width = module.in_features
            else:
                if isinstance(expert, str) and expert.isdecimal():
                    expert = int(expert)
                if type(expert) is not int or not 0 <= expert < mlp.experts.num_experts:
                    raise ValueError(
                        "MoE coordinates require a valid expert ID or shared=True"
                    )
                if text_config(model.config).model_type != "qwen4_exp_text":
                    raise ValueError(
                        "Packed expert steering is supported only for Qwen3.8 Flash"
                    )
                module = mlp.experts
                edits = experts.setdefault(module, {}).setdefault(expert, {})
                width = module.intermediate_dim
        else:
            if expert is not None or shared:
                raise ValueError("Dense models do not have expert coordinates")
            module = mlp.down_proj
            edits = targets.setdefault(module, {})
            width = module.in_features
        if index >= width:
            raise ValueError("Neuron coordinate exceeds MLP width")
        value = strength if strength is not None else position.get("strength")
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or abs(value) > 100
        ):
            raise ValueError(
                "Specify a finite strength between -100 and 100; the recipe may lack a dose"
            )
        if index in edits:
            raise ValueError("Duplicate neuron coordinates")
        edits[index] = value
    handles = []
    model._neuralhub_active = True
    try:
        for module, edits in targets.items():

            def hook(_module, args, edits=edits):
                activation = args[0].clone()
                for index, value in edits.items():
                    if operation == "add":
                        activation[..., index] += value
                    elif operation == "scale":
                        activation[..., index] *= value
                    else:
                        activation[..., index] = value
                return (activation, *args[1:])

            handles.append(module.register_forward_pre_hook(hook))
        for module, edits in experts.items():
            handles.append(
                module.register_forward_hook(
                    expert_hook(edits, operation), with_kwargs=True
                )
            )
        yield model
    finally:
        for handle in handles:
            handle.remove()
        model._neuralhub_active = False


class Model:
    def __init__(self, model, tokenizer, repo):
        self.model, self.tokenizer, self.repo = model, tokenizer, repo

    @classmethod
    def from_neuron(cls, neuron, *, model_revision, device="auto", **kwargs):
        """Caller must explicitly choose a checkpoint; legacy recipes do not pin one."""
        try:
            import transformers
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "Install neuralhub-sdk[transformers] to load models"
            ) from error
        repo = neuron.document["model"].get("huggingface")
        if not repo or not model_revision:
            raise ValueError(
                "A model repository and explicit model_revision are required"
            )
        if kwargs.get("trust_remote_code"):
            raise ValueError("Remote model code is not supported")
        if any(
            kwargs.get(key)
            for key in ("quantization_config", "load_in_4bit", "load_in_8bit")
        ):
            raise ValueError("Quantized models are not supported in this version")
        config = AutoConfig.from_pretrained(
            repo, revision=model_revision, trust_remote_code=False
        )
        if (
            text_config(config).model_type not in SUPPORTED
            or getattr(config, "quantization_config", None)
            or getattr(text_config(config), "quantization_config", None)
        ):
            raise ValueError("Unsupported or quantized model configuration")
        kwargs["trust_remote_code"] = False
        loader = (
            getattr(transformers, CONDITIONAL[config.model_type])
            if config.model_type in CONDITIONAL
            else AutoModelForCausalLM
        )
        model = loader.from_pretrained(
            repo, revision=model_revision, device_map=device, **kwargs
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            repo, revision=model_revision, trust_remote_code=False
        )
        return cls(model, tokenizer, repo)

    def steer(self, neuron, **kwargs):
        if neuron.document["model"].get("huggingface") != self.repo:
            raise ValueError("Recipe and loaded model repositories differ")
        return steer(self.model, neuron, **kwargs)

    def generate(self, prompt, *, max_new_tokens=128, **kwargs):
        import torch

        text = (
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if self.tokenizer.chat_template
            else prompt
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(self.model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, **kwargs
            )
        return self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
