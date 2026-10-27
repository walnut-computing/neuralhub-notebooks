# NeuralHub SDK (0.5.0)

Anonymous neuron downloads and local inference. This is an initial source release;
no PyPI package has been published. Install from this checkout:

```sh
pip install './sdk[transformers]'
```

The Hub URL is explicit: use the URL of your running NeuralHub deployment, including
its base path. No API key is sent. Public recipes are data, never executable code.

```python
from neuralhub import Neuron, Model

neuron = Neuron.from_hub(
    "OWNER/NEURON",  # replace with an existing Hub repository
    base_url="http://localhost:8000",
    revision=1,
)
model = Model.from_neuron(neuron, model_revision="EXACT_MODEL_COMMIT")
print(model.generate("Tell me a short story.", do_sample=False))
with model.steer(neuron, operation="add", strength=1.5):
    print(model.generate("Tell me a short story.", do_sample=False))
```

This example specifies an experimental intervention; it does not establish that a
catalog entry was discovered using this operation, site, or strength. Existing v1
recipes lack those semantics and checkpoint revisions. Confirm the source method
before interpreting results. Strength overrides every coordinate with an absolute
operand. Omit it to use per-coordinate recipe strengths (missing doses raise).

## Supported execution

Eager, unquantized Transformers 5.21.x execution supports all five catalog model families:

| Catalog model | Text architecture | Intervention |
| --- | --- | --- |
| Qwen3.5-4B | `qwen3_5_text` | dense MLP |
| Qwen3.8-Flash-Next | `qwen4_exp_text` | routed expert and shared expert MLP |
| Olmo-3-7B-Instruct | `olmo3` | dense MLP |
| Gemma-4-E4B-IT | `gemma4_text` | dense MLP |
| Qwen3-8B | `qwen3` | dense MLP |

Llama remains supported. Multimodal checkpoint wrappers are loaded with their native
conditional-generation classes; this SDK exposes text generation, not image/audio inputs.
The Qwen3 base checkpoint has no chat template, so generation falls back to a raw prompt.

Zero-based coordinates target the input to `mlp.down_proj`, after activation and gating,
on all prompt and generated tokens. `add`, `scale`, and `set` are supported. For Flash,
`expert` selects an expert ID (integer or decimal string); `shared: true` selects its
shared expert. Routed changes apply only to tokens already routed to that expert and
retain the original routing weights; steering never forces an expert to activate.

Approximate coordinates require explicit `allow_approximate=True` in `steer()` and
use the listed coordinate as-is. Quantized models, compiled execution, vLLM and other
intervention sites remain unsupported. Hooks are removed even when generation fails.
Serialize all calls sharing a model; do not use it concurrently while steering.
Local hardware must fit the model, including the full MoE weights for Flash.

Verification uses small randomly initialized real models for all five architectures,
including prefill/decode, restoration, and a numerical routed-expert reference test.
Full pretrained checkpoints and catalog behavioral effects have not been reproduced.

Attach to an existing supported Transformers model:

```python
from neuralhub import steer

# You are responsible for matching the model and checkpoint to the recipe.
with steer(existing_model, neuron, operation="set", strength=0):
    output = existing_model.generate(**inputs)
```

## Download cache and rate limits

`HubClient(base_url, cache_dir=..., ttl=300, timeout=20, retries=2)` downloads v1
recipes. `client.neuron("owner/slug", revision=1)` returns the JSON dictionary.
The cache keys include the full Hub URL and revision. Downloads are bounded to
256 KiB and cached atomically in `~/.cache/neuralhub`. Fresh entries avoid network
requests; expired entries use ETags when available. Pinned entries also expire:
a deleted repository must not stay publicly accessible through permanent caching.
`offline=True` explicitly uses a previously downloaded entry without contacting
the server; `refresh=True` revalidates now. Deleting the cache removes local copies.

429/502/503/504 responses get at most two retries by default, respecting numeric or
HTTP-date Retry-After values. Delays above 30 seconds raise HubError instead of
retrying early. Network failures raise HubError; no silent stale fallback occurs.
Model files are cached separately by Hugging Face. Restricted models may require
Hugging Face authentication; NeuralHub itself does not require an API key.

## Server setup

The accompanying NeuralHub server limits GET /api requests per peer IP with a token
bucket: default burst 120, refill 120/minute. Configure
`PUBLIC_API_REQUESTS_PER_MINUTE`. Successful anonymous neuron JSON downloads have a
60-second server/CDN cache and ETag; authenticated or cookie-bearing requests bypass
that cache. Other API responses remain no-store. The process stores at most 128
responses of at most 256 KiB and 10,000 IP budgets. Edits/deletion may remain visible
in public caches for 60 seconds (SDK caches for up to its TTL).

These are **per-process** limits, not distributed DDoS protection. Multiple workers
have separate budgets. Put an edge CDN/WAF and shared rate limit in front of a
production deployment, restrict origin access, and configure Uvicorn's trusted
proxy addresses deliberately. This middleware does not trust arbitrary forwarded
IP headers. Behind an unconfigured proxy, users share the proxy IP budget. API
clients should receive 429, not browser CAPTCHA challenges. No infrastructure
settings are changed by installing the SDK.

## Development and distribution

```sh
pip install './sdk[transformers]'
PYTHONPATH=sdk/src python -m unittest discover -s sdk/tests -v
python -m pip install build
python -m build sdk
```

Choose an open-source license and verify package-name availability before public
release. Publish the resulting wheel/sdist to TestPyPI, then PyPI via Trusted
Publishing. Model weights are not included. The `transformers` extra installs the
local inference dependencies; the download-only client uses Python's standard library.


## Colab notebooks

Neuron detail pages offer **Open in Colab** for published catalog snapshots and a
revision-pinned **Download .ipynb** for every public recipe revision. Colab links use
public GitHub notebooks under `walnut-computing/neuralhub-notebooks/notebooks/`; those files must be published to the
referenced Git branch before the links work. New or unexported revisions remain
available through the download API and can be uploaded to Colab manually.

The notebook delegates model loading, source calibration and validation to
`neuralhub.playground`. It installs the SDK from the public Hub source archive and
opens a chat UI; no private repository access is needed. Full unquantized weights
must fit in memory. See the source-method support matrix below.

## Collected evidence and notebooks

`HubClient.neuron()` and `Neuron.from_hub()` accept both `neuralhub/v1`
recipes and `neuralhub/evidence-v1` records. Downloading evidence does not make
its source dose an absolute activation value. For a manual absolute-activation experiment (the source playground handles its own calibration):

```python
evidence = Neuron.from_hub("atlas/collected-example", base_url=HUB_URL, revision=2)
print(evidence.document)  # inspect sourceRecord, model and coordinates
neuron = evidence.prepare(strength=0.1)  # explicitly calibrated absolute operand
model = Model.from_neuron(neuron, model_revision=MODEL_COMMIT)
with model.steer(neuron, operation="add"):
    print(model.generate("Tell me a story."))
```

`prepare` returns a copy, retains the original record under `sourceEvidence`,
and requires an explicit strength or replacement positions with explicit doses.
Use `model_repo=` and `positions=` only after verifying missing information.
Normalized source units (including residual-median/down-column normalization),
quantized MLX experiments, SAE features and arbitrary vector interventions are
not silently converted into Transformers MLP interventions. This is an explicit
local experiment, not a reproduction guarantee. Model architecture, bounds,
expert IDs, finite doses and approximation checks still apply at execution.

### Source-method chat playground

```python
from neuralhub import playground
playground("atlas/collected-c0094-evil-qwen-qwen35-4b", revision=2)
```

Every notebook has three code cells: install, select the pinned neuron, launch.
The SDK resolves the model commit, loads once, calibrates on the clean model and
applies the source method on each chat turn. Strength, steering on/off, temperature
and response length live in the chat UI. The expandable diagnostics show the model
commit, original doses, calibration medians, column norms and resolved operands.
There is no comparison helper or duplicated coordinate/configuration code.

`source_catalog.json` is the reviewed method registry, bound to the repository,
model and exact coordinates/doses. Changing a card invalidates that binding.
The Hub includes its method/status under `steering` in `neuron.json`; the SDK uses
its bundled reviewed registry, never executable instructions from card prose.
All 299 current cards have a decision: 119 source ports and 180 unavailable.
Unavailable cards explain the missing source information before loading weights.
They never receive a guessed +1 dose or default raw-add operation.

Implemented profiles:

| Source | Operation and calibration |
| --- | --- |
| Neuron Studio / Evil Qwen | Add at down-projection input. Fixed river passage without special tokens; midpoint median of layer-output residual norms, excluding the first token. |
| Lyra / metrovoc Qwen3.5 | Add at down-projection input. Documented reference-text fallback, lower median, excluding the first token. Original saved corpus statistics are not bundled. |
| Martin Bowling audit / Hedwig | Add a vector after down projection. Six unsteered greedy answers, 100 tokens, repetition penalty 1.05; upper median of residual norms on prompt + answer, excluding the first token. |
| Neuron Lab | Add at down-projection input. Source calibration questions as chat prompts; lower median, excluding the first token. |
| Alien multiplier | Native activation ×2; no residual normalization. |
| Qwen3 mechanistic experiments | Raw source-selected offsets (+32 or +64); skip the first prompt token, edit subsequent prompt/decode tokens. Source model commit is pinned. |
| Qwen3 refusal-neuron constant mode | Set the activation to the source constant at every prompt/decode token; no residual normalization. Anchor mode is a distinct experiment. |

Normalized additions use `source_dose * median_layer_residual_norm / column_L2`.
Column norms and residuals are computed from the actually loaded checkpoint.
There is no clipping to the old raw-operand ±100 limit. Invalid/zero/nonfinite norms
and dtype overflow fail explicitly. Calibration runs without steering, is reused for
the session, and removes hooks even on failure.

Flash source recipes use unconditional control vectors at the MLP output before
hyper-connection injection, not router-weighted expert activation edits. Their
Strata calibration/backend is unavailable here, so those notebooks explain the
limitation. Poor Things campaign calibration artifacts, unspecified OLMo/Gemma
methods, missing doses and ambiguous social-post methods likewise remain unavailable.
This is intentional: declaring them runnable with unrelated units would be incorrect.

Source ports preserve the reviewed operation, formula, token scope and calibration
procedure where available. They do **not** establish behavioral reproduction or
bitwise agreement with the source MLX/GGUF engine. The UI discloses backend differences
and use of reference fallback statistics. Actual full-checkpoint GPU behavioral
validation has not been performed for this release.

The default strength is 1 (source dose). For multiplication the UI interpolates
around identity: `factor = 1 + (source_factor - 1) * strength`. For add/set it scales
the source operand. Use the separate steering toggle for an unsteered reply.
Generation is greedy by default, thinking is off, and the default output limit is
256 tokens. Conversations over 4096 input tokens ask for a new chat. Calls sharing
the model are serialized; hooks are removed on success and failure.

Notebooks install SDK 0.5.0 from the Hub source archive, including its reviewed JSON
registry. PyPI publication remains deferred. T4 uses FP16; GPUs with native BF16 use
BF16. Full unquantized weights must fit in memory; large models require larger GPUs.
Colab share links are accessible to anyone with the link while the runtime is active.
