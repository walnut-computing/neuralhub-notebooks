"""Small random models exercise real activation changes without downloading weights."""

import unittest

try:
    import torch
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )
except ImportError:
    torch = None
from neuralhub import Neuron, steer


@unittest.skipIf(
    torch is None, "Install the transformers extra to test local inference"
)
class Interventions(unittest.TestCase):
    def test_real_model_intervention_and_cleanup(self):
        for config_class, model_class in (
            (Qwen3Config, Qwen3ForCausalLM),
            (LlamaConfig, LlamaForCausalLM),
        ):
            config = config_class(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
            model = model_class(config).eval()
            neuron = Neuron({"positions": [{"layer": 0, "neuron": 1, "strength": 10}]})
            inputs = torch.tensor([[1, 2, 3]])
            with torch.inference_mode():
                baseline = model(inputs).logits
                with steer(model, neuron, operation="set"):
                    changed = model(inputs).logits
                self.assertFalse(torch.allclose(baseline, changed))
                torch.testing.assert_close(model(inputs).logits, baseline)
                with self.assertRaisesRegex(RuntimeError, "generation failed"):
                    with steer(model, neuron, operation="add"):
                        raise RuntimeError("generation failed")
                torch.testing.assert_close(model(inputs).logits, baseline)
                self.assertFalse(model.model.layers[0].mlp.down_proj._forward_pre_hooks)

    def test_invalid_recipe_does_not_leave_hooks(self):
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        )
        for position in (
            {"layer": 0, "neuron": 1},
            {"layer": 0, "neuron": 1, "expert": 0},
            {"layer": 0, "neuron": 500, "strength": 1},
        ):
            with self.assertRaises(ValueError):
                with steer(model, Neuron({"positions": [position]}), operation="add"):
                    pass
            self.assertFalse(model.model.layers[0].mlp.down_proj._forward_pre_hooks)
