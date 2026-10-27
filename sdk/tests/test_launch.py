import ast
import unittest
from neuralhub.launch import notebook

class NotebookTest(unittest.TestCase):
    def test_three_small_steps_without_duplicated_runtime(self):
        doc=dict(format='neuralhub/v1',id='atlas/example',revision=2,model={},positions=[])
        nb=notebook(doc,'https://example.com/research/neuralhub')
        code=[c for c in nb['cells'] if c['cell_type']=='code']
        self.assertEqual(len(code),3)
        self.assertIn('pip install --upgrade --no-cache-dir',''.join(code[0]['source']))
        source=''.join(code[1]['source']+code[2]['source'])
        ast.parse(source)
        self.assertIn("REPOSITORY = 'atlas/example'",source)
        self.assertIn('REVISION = 2',source)
        self.assertIn('playground(REPOSITORY',source)
        self.assertLess(len(source.splitlines()),10)
        for name in ('compare','PROMPT','MODEL_REVISION','COORDINATE_STRENGTHS','load_experiment','runtime_settings','def '):
            self.assertNotIn(name,source)
        self.assertTrue(all(not c['outputs'] and c['execution_count'] is None for c in code))
