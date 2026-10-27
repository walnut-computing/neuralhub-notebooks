"""Source-specific dose calibration and reversible inference hooks."""
import math
from contextlib import contextmanager
from copy import deepcopy

from .methods import catalog, require_method
from .runtime import decoder, operand, text_config


def median(values, convention):
    ordered = values.sort().values
    n = ordered.numel()
    if not n:
        raise ValueError('Calibration produced no residual samples')
    if convention == 'lower':
        return ordered[(n - 1) // 2].item()
    if convention == 'upper':
        return ordered[n // 2].item()
    if convention == 'midpoint':
        return ((ordered[(n - 1) // 2] + ordered[n // 2]) / 2).item()
    raise ValueError('Unknown source median convention')


def calibrate(model, layer_ids, profile):
    """Measure clean layer-output residuals; never calibrate through active edits."""
    import torch
    raw = model.model
    if getattr(raw, '_neuralhub_active', False):
        raise ValueError('Calibration requires an unsteered model')
    tokenizer = model.tokenizer
    def chat(question):
        return tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if profile['kind'] == 'text':
        texts = [profile['text']]
    elif profile['kind'] == 'answers':
        # Original audit: six unsteered greedy answers, then a fresh full pass.
        texts = [chat(q) + model.generate(q, max_new_tokens=100,
                 enable_thinking=False, do_sample=False, repetition_penalty=1.05)
                 for q in profile['questions']]
    elif profile['kind'] == 'chat':
        texts = [chat(q) for q in profile['questions']]
    else:
        raise ValueError('Unknown source calibration procedure')
    values = {layer: [] for layer in layer_ids}
    handles = []
    try:
        for layer in layer_ids:
            def capture(_module, _args, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                # All supported source profiles exclude the first token.
                values[layer].append(hidden[0, 1:].float().norm(dim=-1).cpu())
            handles.append(decoder(raw).layers[layer].register_forward_hook(capture))
        device = raw.get_input_embeddings().weight.device
        with torch.inference_mode():
            for text in texts:
                inputs = tokenizer(text, return_tensors='pt',
                                   add_special_tokens=profile['add_special_tokens']).to(device)
                # No LM head is needed for residual measurement.
                decoder(raw)(**inputs, use_cache=False)
        result = {layer: median(torch.cat(samples), profile['median'])
                  for layer, samples in values.items()}
        if any(not math.isfinite(v) or v <= 0 for v in result.values()):
            raise ValueError('Calibration residual norms must be positive and finite')
        return result
    finally:
        for handle in handles:
            handle.remove()


class SourceSession:
    """One loaded checkpoint, one immutable recipe, one source calibration."""
    def __init__(self, neuron, model):
        import torch
        self.method = require_method(neuron.document)
        self.document = deepcopy(neuron.document)
        self.model = model
        self.positions = self.document['positions']
        if self.document['model'].get('huggingface') != model.repo:
            raise ValueError('Recipe and loaded model repositories differ')
        raw = model.model
        if getattr(raw, 'is_quantized', False) or hasattr(raw, '_orig_mod'):
            raise ValueError('Source ports require an eager, unquantized model')
        if getattr(raw, '_neuralhub_active', False):
            raise ValueError('Model already has an active steering context')
        self.modules = []
        self.columns = []
        seen = set()
        layers = decoder(raw).layers
        for p in self.positions:
            layer, index = p['layer'], p['neuron']
            if type(layer) is not int or type(index) is not int or not 0 <= layer < len(layers) or index < 0:
                raise ValueError('Invalid source coordinate')
            if (layer, index) in seen:
                raise ValueError('Duplicate source coordinate')
            seen.add((layer, index))
            if p.get('coordinatePrecision') == 'approximate':
                raise ValueError('Source coordinates must be exact')
            mlp = layers[layer].mlp
            if hasattr(mlp, 'experts') or p.get('expert') is not None or p.get('shared'):
                raise ValueError('This source port does not implement routed or shared expert interventions')
            down = mlp.down_proj
            if index >= down.in_features:
                raise ValueError('Source neuron exceeds model MLP width')
            self.modules.append(down)
            self.columns.append(down.weight[:, index].detach().float())
        profile = self.method['calibration']
        if profile and text_config(raw.config).model_type != 'qwen3_5_text':
            raise ValueError('Reviewed residual-normalized profiles require Qwen3.5')
        self.residuals = calibrate(model, sorted({p['layer'] for p in self.positions}),
                                   catalog()['profiles'][profile]) if profile else {}
        self.operands = []
        self.diagnostics = []
        for p, dose, column in zip(self.positions, self.method['doses'], self.columns):
            norm = float(column.norm())
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError('Down-projection column norm must be positive and finite')
            unit = self.residuals[p['layer']] / norm if profile else 1.0
            value = dose * unit
            if not math.isfinite(value):
                raise ValueError('Source dose must resolve to a finite operand')
            self.operands.append(value)
            self.diagnostics.append(dict(layer=p['layer'], neuron=p['neuron'], source_dose=dose,
                residual_median=self.residuals.get(p['layer']), down_column_l2=norm,
                resolved_operand_at_strength_1=value))

    @contextmanager
    def steer(self, multiplier=1.0, enabled=True):
        """Scale the source edit; for multiply, 0→identity and 1→source factor."""
        import torch
        if type(multiplier) not in (int, float) or not math.isfinite(multiplier) or abs(multiplier) > 2:
            raise ValueError('Strength multiplier must be between -2 and 2')
        raw = self.model.model
        if getattr(raw, '_neuralhub_active', False):
            raise ValueError('Overlapping steering contexts are not supported')
        if not enabled:
            yield
            return
        handles = []
        raw._neuralhub_active = True
        try:
            grouped = {}
            for p, module, column, value in zip(self.positions, self.modules, self.columns, self.operands):
                value = 1 + (value - 1) * multiplier if self.method['operation'] == 'scale' else value * multiplier
                grouped.setdefault(module, []).append((p['neuron'], value, column))
            for module, edits in grouped.items():
                if self.method['site'] == 'mlp.down_proj.output':
                    # Audit adds a fixed vector AFTER projection, preserving source rounding/site.
                    delta = sum(column * value for _, value, column in edits)
                    def output_hook(_module, _args, output, delta=delta):
                        change = delta.to(output)
                        if not torch.isfinite(change).all():
                            raise ValueError('Steering vector overflows the loaded dtype')
                        return output + change
                    handles.append(module.register_forward_hook(output_hook))
                else:
                    state = {'first': True}
                    def input_hook(_module, args, edits=edits, state=state):
                        activation = args[0].clone()
                        if activation.shape[0] != 1:
                            raise ValueError('Source playground expects one unpadded conversation')
                        start = 1 if state['first'] and self.method['apply_to'] == 'except_first_prompt_token' else 0
                        state['first'] = False
                        for index, value, _ in edits:
                            selected = activation[:, start:, index]
                            changed = operand(selected, self.method['operation'], value)
                            if not torch.isfinite(changed).all():
                                raise ValueError('Steering operand overflows the loaded dtype')
                            activation[:, start:, index] = changed
                        return (activation, *args[1:])
                    handles.append(module.register_forward_pre_hook(input_hook))
            yield
        finally:
            for handle in handles:
                handle.remove()
            raw._neuralhub_active = False
