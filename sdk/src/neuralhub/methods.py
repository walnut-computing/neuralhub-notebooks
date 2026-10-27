"""Reviewed source methods, bound to model, coordinates and saved doses.

Remote card prose is never interpreted as executable configuration. Unknown or
changed cards fail closed instead of silently becoming raw +1 experiments.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path


def fingerprint(document):
    positions = [{k: p.get(k, False if k == 'shared' else None)
                  for k in ('layer', 'neuron', 'expert', 'shared', 'strength', 'coordinatePrecision')}
                 for p in document['positions']]
    value = [document['model'].get('huggingface'), positions]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=1)
def catalog():
    return json.loads(Path(__file__).with_name('source_catalog.json').read_text())


def source_method(document):
    entry = catalog()['recipes'].get(document['id'])
    if entry is None:
        return {'status': 'unavailable', 'reason': 'No reviewed source method for this recipe.'}
    if fingerprint(document) != entry['fingerprint']:
        return {'status': 'unavailable', 'reason': 'Model, coordinates or doses differ from the reviewed source record.'}
    return deepcopy({k: v for k, v in entry.items() if k != 'fingerprint'})


def require_method(document):
    method = source_method(document)
    if method['status'] != 'ported':
        raise ValueError('Source steering unavailable: ' + method['reason'])
    return method
