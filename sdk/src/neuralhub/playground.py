"""Local chat playground; notebooks contain no model or intervention plumbing."""
import re
from threading import Lock

from .methods import source_method, require_method
from .runtime import Model, Neuron
from .source_runtime import SourceSession


def runtime_settings(device='auto'):
    import torch
    if device == 'auto':
        if not torch.cuda.is_available():
            raise ValueError('Choose Runtime > Change runtime type > GPU in Colab. Local CPU users can pass device="cpu".')
        device = 'cuda'
    if str(device).startswith('cuda'):
        bf16 = torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def load_session(neuron, device='auto'):
    from huggingface_hub import model_info
    method = require_method(neuron.document)  # Fail before downloading model weights.
    device, dtype = runtime_settings(device)
    repository = neuron.document['model']['huggingface']
    revision = method.get('model_revision') or model_info(repository, revision='main').sha
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        raise ValueError('Hugging Face did not return an exact model commit')
    # The reviewed source method, not an arbitrary raw dose, authorizes execution.
    prepared = Neuron({**neuron.document, 'format': 'neuralhub/v1'})
    print(f'Loading {repository} @ {revision} ({device}, {dtype}).')
    model = Model.from_neuron(prepared, model_revision=revision, device=device,
                              dtype=dtype, attn_implementation='sdpa')
    print('Calibrating the source method on the unsteered model…')
    session = SourceSession(neuron, model)
    session.model_revision = revision
    return session


def chat_callback(session):
    import gradio as gr
    lock = Lock()
    def chat(message, history, enabled, multiplier, temperature, max_tokens):
        messages = []
        for turn in history:
            content = turn['content']
            if isinstance(content, list):
                content = ''.join(block.get('text', '') for block in content
                                  if isinstance(block, dict) and block.get('type') == 'text')
            if turn['role'] in ('user', 'assistant') and isinstance(content, str):
                messages.append({'role': turn['role'], 'content': content})
        messages.append({'role': 'user', 'content': message})
        options = dict(max_new_tokens=int(max_tokens), enable_thinking=False,
                       max_input_tokens=4096, do_sample=temperature > 0, use_cache=True)
        if session.method['calibration'] == 'audit-six-answers':
            options['repetition_penalty'] = 1.05
        if temperature > 0:
            options.update(temperature=temperature, top_p=0.9)
        try:
            with lock, session.steer(multiplier, enabled):
                return session.model.generate(messages, **options)
        except ValueError as error:
            raise gr.Error(str(error)) from error
    return chat


def build_playground(session):
    import gradio as gr
    method = session.method
    with gr.Blocks() as demo:
        gr.Markdown('# NeuralHub · Steered chat')
        gr.Markdown(f"Recipe: `{session.document['id']}` · r{session.document['revision']}\n\n"
                    f"Source operation: **{method['operation']}** · units: `{method['normalization']}`. "
                    'Strength 1 uses the source dose. Changes affect future replies; start a new chat for a fresh conversation.')
        if method['operation'] == 'scale':
            gr.Markdown('Strength adjusts the multiplier around identity: 0 = ×1, 1 = the source multiplier.')
        elif method['operation'] == 'set':
            gr.Markdown('Strength multiplies the fixed clamp value; use the steering toggle to disable the clamp.')
        gr.Markdown((method.get('note', '') + '\n\n' + method['validation']).strip())
        if session.document.get('reviewRequired'):
            gr.Markdown('**Source review required:** this record may report weak or failed effects. See the neuron card.')
        with gr.Row():
            enabled = gr.Checkbox(True, label='Steering enabled')
            strength = gr.Slider(-2, 2, value=1, step=0.05, label='Strength (1 = source dose)')
            temperature = gr.Slider(0, 1.5, value=0, step=0.05, label='Temperature')
            length = gr.Slider(64, 1024, value=256, step=64, label='Max new tokens')
        with gr.Accordion('Source method and calibration at strength 1', open=False):
            gr.JSON(value=dict(model=session.model.repo, model_revision=session.model_revision,
                               method=method, coordinates=session.diagnostics))
        gr.ChatInterface(fn=chat_callback(session), additional_inputs=[enabled, strength, temperature, length])
        gr.Markdown('The model runs in this runtime. Anyone with the temporary Gradio share link can use it while it is running.')
    return demo


def playground(repository, *, revision=None, base_url='https://mywalnuts.com/research/neuralhub',
               device='auto', share=True, inline=True):
    """Fetch a pinned recipe, resolve its source method, calibrate once and chat."""
    import gradio as gr
    neuron = Neuron.from_hub(repository, base_url=base_url, revision=revision, refresh=True)
    method = source_method(neuron.document)
    if method['status'] != 'ported':
        with gr.Blocks() as demo:
            gr.Markdown('# Source steering unavailable')
            gr.Markdown(method['reason'])
            gr.Markdown('This recipe has no verified executable source configuration in this SDK. '
                        'No model was loaded and no substitute raw-add intervention was applied.')
            gr.JSON(value=method, label='Source method record')
    else:
        demo = build_playground(load_session(neuron, device))
    return demo.queue(default_concurrency_limit=1).launch(share=share, inline=inline, show_error=True)
