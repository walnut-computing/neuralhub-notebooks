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


@unittest.skipIf(torch is None, 'Install inference dependencies')
class ChatGeneration(unittest.TestCase):
    def test_history_and_thinking_are_template_options_not_generation_options(self):
        from unittest.mock import Mock
        from neuralhub import Model
        tokenizer = Mock(chat_template='template')
        tokenizer.apply_chat_template.return_value = 'formatted conversation'
        class Inputs(dict):
            def to(self, device):
                return self
        tokenizer.return_value = Inputs(input_ids=torch.tensor([[1, 2, 3]]))
        tokenizer.decode.return_value = 'reply'
        raw = Mock()
        raw.get_input_embeddings.return_value.weight.device = 'cpu'
        raw.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        model = Model(raw, tokenizer, 'example/model')
        messages = [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi'},
                    {'role': 'user', 'content': 'Continue'}]
        self.assertEqual(model.generate(messages, enable_thinking=False, max_input_tokens=3), 'reply')
        tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.assertNotIn('enable_thinking', raw.generate.call_args.kwargs)
        raw.generate.reset_mock()
        with self.assertRaisesRegex(ValueError, 'too long'):
            model.generate(messages, max_input_tokens=2)
        raw.generate.assert_not_called()

    def test_raw_checkpoint_retains_multi_turn_context(self):
        from unittest.mock import Mock
        from neuralhub import Model
        tokenizer = Mock(chat_template=None)
        class Inputs(dict):
            def to(self, device):
                return self
        tokenizer.return_value = Inputs(input_ids=torch.tensor([[1]]))
        raw = Mock()
        raw.generate.return_value = torch.tensor([[1, 2]])
        model = Model(raw, tokenizer, 'example/base')
        model.generate([{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi'},
                        {'role': 'user', 'content': 'Again'}])
        self.assertEqual(tokenizer.call_args.args[0], 'user: Hello\nassistant: Hi\nuser: Again\nassistant:')
