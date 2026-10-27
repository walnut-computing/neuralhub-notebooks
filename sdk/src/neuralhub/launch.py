"""Generate three-step notebooks; all runtime behavior lives in the SDK."""
import shlex
from .hub import HubClient


def notebook(document, base_url):
    HubClient(base_url)
    HubClient.validate(document, document['id'], document['revision'])
    def cell(kind, source):
        result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(True)}
        if kind == 'code':
            result.update(execution_count=None, outputs=[])
        return result
    install = 'neuralhub-sdk[transformers] @ ' + base_url.rstrip('/') + '/api/sdk/source.zip'
    return {
        'nbformat': 4, 'nbformat_minor': 4,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'accelerator': 'GPU', 'colab': {'name': document['id'].replace('/', '-') + '.ipynb'}},
        'cells': [
            cell('markdown', '# NeuralHub chat playground\n'
                 'Choose **Runtime > Change runtime type > GPU**, then run the three cells: '
                 '**install → choose neuron → launch playground**. Adjust strength in the chat UI.\n\n'
                 'The SDK applies the reviewed source operation and calibration for the selected recipe. '
                 'Recipes with incomplete source settings stop with an explanation. '
                 'Model weights must fit in GPU memory; large models do not fit a T4. '
                 'Restricted checkpoints require Hugging Face login. '
                 'Method details, checkpoint and calibration values are available inside the playground.'),
            cell('code', '%pip install --upgrade --no-cache-dir ' + shlex.quote(install) + " 'gradio>=6,<7'"),
            cell('code', f"REPOSITORY = {document['id']!r}\nREVISION = {document['revision']!r}\nHUB_URL = {base_url.rstrip('/')!r}\n"),
            cell('code', 'from neuralhub import playground\n\nplayground(REPOSITORY, revision=REVISION, base_url=HUB_URL)\n'),
        ],
    }
