"""Real tiny Transformers models: no downloaded weights or GPU required."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from transformers import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Olmo3Config,
    Olmo3ForCausalLM,
    Qwen3_5TextConfig,
    Qwen3_5ForCausalLM,
    Gemma4TextConfig,
    Gemma4ForCausalLM,
    Qwen4ExpTextConfig,
    Qwen4ExpForCausalLM,
)
from neuralhub import Neuron, steer


def configs():
    common = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        eos_token_id=2,
        pad_token_id=0,
    )
    hybrid = dict(
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
    )
    return [
        (Qwen3ForCausalLM, Qwen3Config(**common)),
        (Olmo3ForCausalLM, Olmo3Config(**common)),
        (
            Qwen3_5ForCausalLM,
            Qwen3_5TextConfig(
                **common, **hybrid, layer_types=["linear_attention", "full_attention"]
            ),
        ),
        (
            Gemma4ForCausalLM,
            Gemma4TextConfig(
                **common,
                vocab_size_per_layer_input=32,
                hidden_size_per_layer_input=4,
                layer_types=["sliding_attention", "full_attention"],
            ),
        ),
        (
            Qwen4ExpForCausalLM,
            Qwen4ExpTextConfig(
                **common,
                **hybrid,
                num_experts=3,
                num_experts_per_tok=2,
                moe_intermediate_size=8,
                shared_expert_intermediate_size=8,
                hc_count=2,
                hc_lowrank=4,
                layer_types=["linear_attention", "indexed_attention"],
                ple_layer_ids=[],
                indexer_n_heads=2,
                indexer_kv_heads=1,
                indexer_head_dim=8,
                indexer_budget=8,
                indexer_compress_ratio=2,
            ),
        ),
    ]


class CatalogRuntime(unittest.TestCase):
    def test_all_catalog_architectures_forward_generate_and_restore(self):
        torch.manual_seed(7)
        for cls, config in configs():
            with self.subTest(architecture=config.model_type):
                model = cls(config).eval()
                positions = [dict(layer=0, neuron=1)]
                if config.model_type == "qwen4_exp_text":
                    positions = [
                        dict(layer=0, neuron=1, expert=str(i)) for i in range(3)
                    ]
                    positions.append(dict(layer=0, neuron=2, shared=True))
                neuron = Neuron({"positions": positions})
                x = torch.tensor([[3, 4, 5]])
                with torch.inference_mode():
                    baseline = model(x).logits
                    with steer(model, neuron, operation="set", strength=5):
                        changed = model(x).logits
                        self.assertFalse(torch.allclose(baseline, changed))
                        self.assertEqual(
                            model.generate(x, max_new_tokens=2, do_sample=False).shape[
                                0
                            ],
                            1,
                        )
                    torch.testing.assert_close(model(x).logits, baseline)
                    with self.assertRaises(RuntimeError):
                        with steer(model, neuron, operation="add", strength=2):
                            raise RuntimeError("failed generation")
                    torch.testing.assert_close(model(x).logits, baseline)
                    self.assertFalse(
                        any(
                            m._forward_hooks or m._forward_pre_hooks
                            for m in model.modules()
                        )
                    )

    def test_routed_experts_match_direct_activation_reference(self):
        cls, config = configs()[-1]
        model = cls(config).eval()
        module = model.model.layers[0].mlp.experts
        hidden = torch.randn(3, 16)
        selected = torch.tensor([[0, 1], [1, 2], [2, 0]])
        weights = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.7, 0.3]])
        neuron = Neuron({"positions": [{"layer": 0, "neuron": 1, "expert": "0"}]})
        for operation, value in [("add", 2), ("scale", 3), ("set", -2)]:
            expected = torch.zeros_like(hidden)
            for t in range(3):
                for slot in range(2):
                    e = selected[t, slot]
                    gate, up = torch.nn.functional.linear(
                        hidden[t], module.gate_up_proj[e]
                    ).chunk(2, -1)
                    a = module.act_fn(gate) * up
                    if e == 0:
                        a[1] = (
                            a[1] + value
                            if operation == "add"
                            else a[1] * value
                            if operation == "scale"
                            else value
                        )
                    expected[t] += (
                        torch.nn.functional.linear(a, module.down_proj[e])
                        * weights[t, slot]
                    )
            with steer(model, neuron, operation=operation, strength=value):
                actual = module(hidden, selected, weights)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_catalog_checkpoint_loader_selection(self):
        import json
        from pathlib import Path
        from unittest.mock import MagicMock
        from transformers import AutoConfig
        from neuralhub import Model
        from neuralhub.runtime import CONDITIONAL

        catalog = json.loads(
            (Path(__file__).resolve().parents[2] / "catalog-source.json").read_text()
        )
        for entry in catalog["models"]:
            with self.subTest(model=entry["id"]):
                data = dict(entry["config"])
                kind = data.pop("model_type")
                config = AutoConfig.for_model(kind, **data)
                loader = CONDITIONAL.get(kind, "AutoModelForCausalLM")
                model = MagicMock()
                tokenizer = MagicMock()
                neuron = Neuron({"model": {"huggingface": entry["repo"]}})
                with (
                    patch(
                        "transformers.AutoConfig.from_pretrained", return_value=config
                    ),
                    patch(
                        f"transformers.{loader}.from_pretrained", return_value=model
                    ) as load,
                    patch(
                        "transformers.AutoTokenizer.from_pretrained",
                        return_value=tokenizer,
                    ),
                ):
                    result = Model.from_neuron(
                        neuron, model_revision="exact-commit", device="cpu"
                    )
                self.assertIs(result.model, model)
                load.assert_called_once_with(
                    entry["repo"],
                    revision="exact-commit",
                    device_map="cpu",
                    trust_remote_code=False,
                )

    def test_multimodal_decoder_and_approximate_opt_in(self):
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
            )
        )
        # Multimodal checkpoints wrap the same text decoder under language_model.
        wrapped = SimpleNamespace(
            config=SimpleNamespace(text_config=model.config),
            model=SimpleNamespace(language_model=model.model),
        )
        neuron = Neuron(
            {
                "positions": [
                    {"layer": 0, "neuron": 1, "coordinatePrecision": "approximate"}
                ]
            }
        )
        with self.assertRaisesRegex(ValueError, "allow_approximate"):
            with steer(wrapped, neuron, operation="set", strength=0):
                pass
        with steer(
            wrapped, neuron, operation="set", strength=0, allow_approximate=True
        ):
            self.assertTrue(model.model.layers[0].mlp.down_proj._forward_pre_hooks)
        self.assertFalse(model.model.layers[0].mlp.down_proj._forward_pre_hooks)
