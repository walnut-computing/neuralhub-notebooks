import unittest
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import contextmanager
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3_5TextConfig, Qwen3_5ForCausalLM
from neuralhub import Model, Neuron
from neuralhub.source_runtime import SourceSession, calibrate, median
from neuralhub.methods import source_method, fingerprint, catalog
from neuralhub.playground import runtime_settings, load_session, chat_callback


class SourceRuntimeTest(unittest.TestCase):
    def model(self, qwen35=True):
        common = dict(vocab_size=32, hidden_size=8, intermediate_size=16,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      head_dim=4, max_position_embeddings=64)
        if qwen35:
            config = Qwen3_5TextConfig(**common, linear_key_head_dim=4, linear_value_head_dim=4,
                linear_num_key_heads=2, linear_num_value_heads=2, layer_types=['full_attention'] * 2)
            raw = Qwen3_5ForCausalLM(config).eval()
        else:
            raw = Qwen3ForCausalLM(Qwen3Config(**common)).eval()
        return Model(raw, None, 'test/model')

    def session(self, model, operation='add', profile='studio-river', site='mlp.down_proj.input', tokens='all_tokens', doses=None):
        doc = dict(id='test/card', revision=1, model={'huggingface':'test/model'}, positions=[
            dict(layer=0, neuron=2, strength=1), dict(layer=0, neuron=3, strength=-1)])
        method = dict(status='ported', calibration=profile, operation=operation,
                      doses=doses or [1, -1], site=site, apply_to=tokens)
        with patch('neuralhub.source_runtime.require_method', return_value=method), \
             patch('neuralhub.source_runtime.calibrate', return_value={0: 12.0} if profile else {}):
            return SourceSession(Neuron(doc), model)

    def test_normalized_vector_has_source_residual_magnitude_and_sign(self):
        model = self.model()
        session = self.session(model)
        for column, value, dose in zip(session.columns, session.operands, [1,-1]):
            self.assertAlmostEqual(float((column * value).norm()), 12, places=5)
            self.assertEqual(value > 0, dose > 0)
        down = model.model.model.layers[0].mlp.down_proj
        x = torch.randn(1,3,16)
        baseline = down(x)
        expected = sum(c*v for c,v in zip(session.columns, session.operands))
        with session.steer():
            torch.testing.assert_close(down(x)-baseline, expected.expand_as(baseline), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(down(x), baseline, rtol=0, atol=0)

    def test_audit_vector_is_applied_after_down_projection(self):
        model = self.model()
        session = self.session(model, site='mlp.down_proj.output')
        down = model.model.model.layers[0].mlp.down_proj
        x = torch.randn(1,3,16)
        baseline = down(x)
        with session.steer():
            self.assertFalse(down._forward_pre_hooks)
            torch.testing.assert_close(down(x)-baseline,
                sum(c*v for c,v in zip(session.columns,session.operands)).expand_as(baseline), atol=2e-6,rtol=2e-6)

    def test_multiply_and_set_keep_native_units(self):
        for operation, doses in [('scale',[2,-1]), ('set',[30,-80])]:
            model = self.model(False)
            session = self.session(model, operation=operation, profile=None, doses=doses)
            down = model.model.model.layers[0].mlp.down_proj
            x = torch.randn(1,3,16); changed=x.clone()
            for index, value in zip([2,3],doses):
                changed[...,index] = changed[...,index]*value if operation=='scale' else value
            expected=down(changed)
            with session.steer():torch.testing.assert_close(down(x),expected)
            with session.steer(enabled=False):torch.testing.assert_close(down(x),down(x))
            if operation=='scale':
                expected=down(x)
                with session.steer(0):torch.testing.assert_close(down(x),expected)

    def test_first_prompt_token_is_skipped_but_decode_and_next_turn_are_edited(self):
        model=self.model(False)
        session=self.session(model,profile=None,tokens='except_first_prompt_token')
        down=model.model.model.layers[0].mlp.down_proj
        x=torch.randn(1,3,16); plain=down(x)
        for turn in range(2):
            with session.steer():
                actual=down(x)
                torch.testing.assert_close(actual[:,0],plain[:,0])
                self.assertFalse(torch.equal(actual[:,1:],plain[:,1:]))
                self.assertFalse(torch.equal(down(x[:,:1]),plain[:,:1]))

    def test_exception_cleanup_and_overlapping_context_rejection(self):
        model=self.model();session=self.session(model)
        with self.assertRaisesRegex(RuntimeError,'generation failed'):
            with session.steer():
                with self.assertRaisesRegex(ValueError,'Overlapping'):
                    with session.steer():pass
                raise RuntimeError('generation failed')
        self.assertFalse(model.model._neuralhub_active)
        for layer in model.model.model.layers:
            self.assertFalse(layer.mlp.down_proj._forward_pre_hooks)
            self.assertFalse(layer.mlp.down_proj._forward_hooks)

    def test_real_tiny_model_logits_change_and_restore(self):
        model=self.model();session=self.session(model)
        inputs=torch.tensor([[1,2,3]])
        with torch.inference_mode():
            plain=model.model(inputs).logits
            with session.steer():changed=model.model(inputs).logits
            restored=model.model(inputs).logits
        self.assertFalse(torch.equal(plain,changed))
        torch.testing.assert_close(plain,restored,rtol=0,atol=0)

    def test_median_conventions_are_not_interchangeable(self):
        x=torch.tensor([9.,1.,3.,5.])
        self.assertEqual(median(x,'lower'),3)
        self.assertEqual(median(x,'upper'),5)
        self.assertEqual(median(x,'midpoint'),4)

    def test_calibration_excludes_first_token_and_cleans_up_on_failure(self):
        model=self.model()
        class Inputs(dict):
            def to(self,device):return self
        tokenizer=unittest.mock.Mock(return_value=Inputs(input_ids=torch.tensor([[1,2,3,4]])))
        model.tokenizer=tokenizer
        observed=[]
        handle=model.model.model.layers[0].register_forward_hook(lambda m,a,o:observed.append(o.detach()))
        try:actual=calibrate(model,[0],catalog()['profiles']['studio-river'])
        finally:handle.remove()
        self.assertAlmostEqual(actual[0],median(observed[0][0,1:].float().norm(dim=-1),'midpoint'))
        self.assertFalse(tokenizer.call_args.kwargs['add_special_tokens'])
        tokenizer.side_effect=RuntimeError('tokenizer failed')
        with self.assertRaises(RuntimeError):calibrate(model,[0],catalog()['profiles']['studio-river'])
        self.assertFalse(model.model.model.layers[0]._forward_hooks)

    def test_unknown_or_changed_recipe_is_not_guessed(self):
        doc=dict(id='test/card',model={'huggingface':'test/model'},positions=[])
        self.assertEqual(source_method(doc)['status'],'unavailable')
        row=dict(fingerprint=fingerprint(doc),status='ported')
        with patch('neuralhub.methods.catalog',return_value={'recipes':{'test/card':row}}):
            self.assertEqual(source_method(doc)['status'],'ported')
            doc['model']['huggingface']='different/model'
            self.assertEqual(source_method(doc)['status'],'unavailable')

    def test_unknown_source_stops_before_model_download(self):
        neuron=Neuron(dict(id='unknown/card',model={},positions=[]))
        with patch('neuralhub.playground.Model.from_neuron') as loader:
            with self.assertRaisesRegex(ValueError,'Source steering unavailable'):load_session(neuron)
            loader.assert_not_called()

    def test_t4_uses_fp16(self):
        with patch('torch.cuda.is_available',return_value=True), \
             patch('torch.cuda.get_device_capability',return_value=(7,5)), \
             patch('torch.cuda.is_bf16_supported',return_value=True):
            self.assertEqual(runtime_settings(),('cuda',torch.float16))

    def test_chat_keeps_history_and_applies_ui_settings(self):
        events=[]
        @contextmanager
        def steer(multiplier,enabled):
            events.append((multiplier,enabled));yield
        model=unittest.mock.Mock()
        model.generate.return_value='reply'
        session=SimpleNamespace(steer=steer,model=model,method={'calibration':'audit-six-answers'})
        reply=chat_callback(session)('next',[{'role':'assistant','content':[{'type':'text','text':'hello'}]}],True,.5,0,256)
        self.assertEqual(reply,'reply');self.assertEqual(events,[(.5,True)])
        args,kwargs=model.generate.call_args
        self.assertEqual(args[0],[{'role':'assistant','content':'hello'},{'role':'user','content':'next'}])
        self.assertFalse(kwargs['enable_thinking']);self.assertFalse(kwargs['do_sample'])
        self.assertEqual(kwargs['repetition_penalty'],1.05)

    def test_gradio_build_has_ui_controls_and_diagnostics(self):
        from neuralhub.playground import build_playground
        model=self.model();session=self.session(model)
        session.model_revision='a'*40
        session.method.update(normalization='layer_residual_median_over_down_column_l2',validation='Test model only.')
        demo=build_playground(session)
        config=demo.get_config_file()
        labels={c['props'].get('label') for c in config['components']}
        self.assertTrue({'Steering enabled','Strength (1 = source dose)','Temperature','Max new tokens'} <= labels)
        self.assertTrue(any(c['type']=='json' for c in config['components']))
        demo.close()
