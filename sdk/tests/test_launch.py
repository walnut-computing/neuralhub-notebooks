import ast
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from neuralhub.launch import experiment, notebook

DOCUMENT = dict(format='neuralhub/v1', id='alice/pirate', revision=3,
                model={'huggingface': 'example/model'}, positions=[{'layer': 1, 'neuron': 2, 'strength': 1.5}],
                readme="__import__('os').system('false')", effect='untrusted card')


class ExperimentTest(unittest.TestCase):
    def namespace(self, document=DOCUMENT):
        namespace = {'__name__': 'test'}
        exec(compile(experiment(document, 'https://example.com/research/neuralhub'), '<generated>', 'exec'), namespace)
        return namespace

    def fake_model(self, events):
        class FakeModel:
            def generate(self, prompt, **kwargs):
                events.append(('generate', prompt, kwargs))
                return 'text'
            @contextmanager
            def steer(self, neuron, **kwargs):
                events.append(('enter', neuron, kwargs))
                try: yield
                finally: events.append(('exit',))
        return FakeModel()

    def test_prefills_saved_scalar_mixed_and_missing_doses(self):
        scalar = self.namespace()
        self.assertEqual(scalar['STRENGTH'], 1.5)
        self.assertEqual(scalar['OPERATION'], 'add')
        mixed = self.namespace({**DOCUMENT, 'positions': [
            {'layer': 1, 'neuron': 2, 'strength': -1.5},
            {'layer': 1, 'neuron': 3, 'strength': 0.5},
            {'layer': 1, 'neuron': 4}]})
        self.assertEqual(mixed['STRENGTH'], 'per_coordinate')
        self.assertEqual(mixed['COORDINATE_STRENGTHS'], [-1.5, 0.5, 1.0])
        self.assertEqual(mixed['MISSING_DOSE_COORDINATES'], [2])

    def test_auto_resolves_checkpoint_once_and_uses_same_prompt(self):
        ns = self.namespace()
        ns['DEVICE'] = 'cpu'
        events = []
        with patch.object(ns['Neuron'], 'from_hub', return_value=ns['Neuron'](DOCUMENT)) as fetch, \
             patch.object(ns['Model'], 'from_neuron', return_value=self.fake_model(events)) as load, \
             patch('huggingface_hub.model_info', return_value=SimpleNamespace(sha='a' * 40)) as info:
            result = ns['compare']()
            ns['compare']()
        info.assert_called_once_with('example/model', revision='main')
        self.assertEqual(ns['MODEL_REVISION'], 'a' * 40)
        self.assertEqual(result['model_revision'], 'a' * 40)
        self.assertEqual(load.call_args.kwargs['model_revision'], 'a' * 40)
        self.assertEqual(events[0], events[2])
        self.assertEqual(events[3], ('exit',))
        self.assertEqual(fetch.call_args.kwargs['revision'], 3)

    def test_exact_checkpoint_skips_resolution_and_mixed_doses_reach_runtime(self):
        doc = {**DOCUMENT, 'positions': [
            {'layer': 1, 'neuron': 2, 'strength': -1.5},
            {'layer': 1, 'neuron': 3, 'strength': 0.5}]}
        ns = self.namespace(doc)
        ns.update(DEVICE='cpu', MODEL_REVISION='b' * 40)
        events = []
        with patch.object(ns['Neuron'], 'from_hub', return_value=ns['Neuron'](doc)), \
             patch.object(ns['Model'], 'from_neuron', return_value=self.fake_model(events)), \
             patch('huggingface_hub.model_info') as info:
            ns['compare']()
        info.assert_not_called()
        self.assertEqual([p['strength'] for p in events[1][1].document['positions']], [-1.5, 0.5])
        self.assertIsNone(events[1][2]['strength'])

    def test_t4_uses_fp16_even_when_bf16_emulation_is_reported(self):
        import torch
        ns = self.namespace()
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_capability', return_value=(7, 5)), \
             patch('torch.cuda.is_bf16_supported', return_value=True):
            self.assertEqual(ns['runtime_settings'](), ('cuda', torch.float16))
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_capability', return_value=(8, 0)), \
             patch('torch.cuda.is_bf16_supported', return_value=True):
            self.assertEqual(ns['runtime_settings'](), ('cuda', torch.bfloat16))

    def test_no_gpu_stops_before_recipe_or_model_download(self):
        ns = self.namespace()
        with patch('torch.cuda.is_available', return_value=False), \
             patch.object(ns['Neuron'], 'from_hub') as fetch, \
             patch.object(ns['Model'], 'from_neuron') as load:
            with self.assertRaisesRegex(RuntimeError, 'Change runtime type'):
                ns['compare']()
            fetch.assert_not_called()
            load.assert_not_called()

    def test_invalid_operation_or_dose_stops_before_download(self):
        for setting in ({'OPERATION': 'invalid'}, {'STRENGTH': float('nan')}, {'STRENGTH': None}):
            ns = self.namespace()
            ns.update(setting)
            with patch.object(ns['Neuron'], 'from_hub') as fetch:
                with self.assertRaises(ValueError): ns['compare']()
                fetch.assert_not_called()

    def test_approximate_use_is_visible_and_can_be_disabled(self):
        doc = {**DOCUMENT, 'positions': [{**DOCUMENT['positions'][0], 'coordinatePrecision': 'approximate'}]}
        ns = self.namespace(doc)
        self.assertTrue(ns['ALLOW_APPROXIMATE'])
        nb = notebook(doc, 'https://example.com')
        self.assertIn('Approximate coordinates', ''.join(nb['cells'][0]['source']))
        ns.update(DEVICE='cpu', ALLOW_APPROXIMATE=False)
        with patch.object(ns['Neuron'], 'from_hub', return_value=ns['Neuron'](doc)), patch.object(ns['Model'], 'from_neuron') as load:
            with self.assertRaisesRegex(ValueError, 'approximate'): ns['compare']()
            load.assert_not_called()

    def test_bad_checkpoint_resolution_is_not_silently_used(self):
        ns = self.namespace()
        ns['DEVICE'] = 'cpu'
        with patch.object(ns['Neuron'], 'from_hub', return_value=ns['Neuron'](DOCUMENT)), \
             patch('huggingface_hub.model_info', return_value=SimpleNamespace(sha='bad')), \
             patch.object(ns['Model'], 'from_neuron') as load:
            with self.assertRaisesRegex(ValueError, 'exact model commit'): ns['compare']()
            load.assert_not_called()

    def test_notebook_has_no_untrusted_card_code_or_outputs(self):
        nb = notebook(DOCUMENT, 'https://example.com')
        self.assertEqual(nb['metadata']['accelerator'], 'GPU')
        self.assertIn('different normalization units', ''.join(nb['cells'][0]['source']))
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                self.assertNotIn(DOCUMENT['readme'], source)
                self.assertEqual(cell['outputs'], [])
                if not source.startswith('%'): ast.parse(source)

class EvidenceExperimentTest(unittest.TestCase):
    def test_evidence_requires_calibrated_strength_and_keeps_original(self):
        from neuralhub import Neuron, HubClient
        from copy import deepcopy
        evidence = {**deepcopy(DOCUMENT), 'format': 'neuralhub/evidence-v1', 'reviewRequired': True}
        original = deepcopy(evidence)
        HubClient.validate(evidence, evidence['id'], evidence['revision'])
        neuron = Neuron(evidence)
        with self.assertRaisesRegex(ValueError, 'absolute activation'):
            neuron.prepare()
        ready = neuron.prepare(strength=0.25)
        self.assertEqual(ready.document['positions'][0]['strength'], 0.25)
        self.assertEqual(ready.document['sourceEvidence'], original)
        self.assertEqual(evidence, original)
        self.assertEqual(ready.document['format'], 'neuralhub/v1')

    def test_unresolved_evidence_can_be_completed_without_guessing(self):
        from neuralhub import Neuron
        evidence = {**DOCUMENT, 'format': 'neuralhub/evidence-v1', 'model': {}, 'positions': []}
        neuron = Neuron(evidence)
        with self.assertRaisesRegex(ValueError, 'MODEL_REPO'):
            neuron.prepare(strength=1)
        with self.assertRaisesRegex(ValueError, 'POSITIONS'):
            neuron.prepare(model_repo='example/verified', strength=1)
        ready = neuron.prepare(model_repo='example/verified', positions=[{'layer': 0, 'neuron': 1, 'strength': 0.1}])
        self.assertEqual(ready.document['model']['huggingface'], 'example/verified')
        self.assertEqual(evidence['positions'], [])

    def test_evidence_notebook_prepares_explicit_experimental_doses(self):
        evidence = {**DOCUMENT, 'format': 'neuralhub/evidence-v1'}
        nb = notebook(evidence, 'https://example.com/research/neuralhub')
        self.assertIn('https://example.com/research/neuralhub/api/sdk/source.zip', ''.join(nb['cells'][1]['source']))
        ns = ExperimentTest().namespace(evidence)
        ns.update(MODEL_REVISION='a' * 40, DEVICE='cpu')
        events = []
        with patch.object(ns['Neuron'], 'from_hub', return_value=ns['Neuron'](evidence)), patch.object(ns['Model'], 'from_neuron', return_value=ExperimentTest().fake_model(events)) as load:
            ns['compare']()
            prepared = load.call_args.args[0].document
            self.assertEqual(prepared['format'], 'neuralhub/v1')
            self.assertEqual(prepared['sourceEvidence'], evidence)
            self.assertEqual(prepared['positions'][0]['strength'], 1.5)
