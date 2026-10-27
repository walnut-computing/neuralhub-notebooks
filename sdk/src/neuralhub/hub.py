"""Anonymous Hub downloads with bounded retries and a disk cache."""

import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


class HubError(RuntimeError):
    pass


class HubClient:
    def __init__(self, base_url, *, cache_dir=None, ttl=300, timeout=20, retries=2):
        url = urlsplit(base_url)
        if (
            url.scheme not in ("https", "http")
            or not url.netloc
            or url.query
            or url.fragment
            or url.username
        ):
            raise ValueError(
                "base_url must be an HTTP(S) Hub URL without credentials, query or fragment"
            )
        if ttl < 0 or timeout <= 0 or not 0 <= retries <= 5:
            raise ValueError("Invalid cache TTL, timeout or retry count")
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir or Path.home() / ".cache/neuralhub")
        self.ttl, self.timeout, self.retries = ttl, timeout, retries

    def neuron(self, repo, *, revision=None, offline=False, refresh=False):
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", repo):
            raise ValueError("Use owner/neuron as the repository name")
        if revision is not None and (type(revision) is not int or revision < 1):
            raise ValueError("revision must be a positive integer")
        url = f"{self.base_url}/api/neurons/{repo}/neuron.json"
        if revision is not None:
            url += "?" + urlencode({"rev": revision})
        path = self.cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        cached = None
        try:
            cached = json.loads(path.read_text())
            self.validate(cached["document"], repo, revision)
            float(cached["saved_at"])
        except (OSError, ValueError, KeyError, TypeError, HubError):
            cached = None
        if offline:
            if not cached:
                raise HubError(
                    "Recipe is not cached; download it once before using offline=True"
                )
            return cached["document"]
        if (
            cached
            and not refresh
            and time.time() - float(cached["saved_at"]) < self.ttl
        ):
            return cached["document"]
        headers = {"Accept": "application/json", "User-Agent": "neuralhub-sdk/0.5.0"}
        if cached and cached.get("etag"):
            headers["If-None-Match"] = cached["etag"]
        for attempt in range(self.retries + 1):
            try:
                with urlopen(
                    Request(url, headers=headers), timeout=self.timeout
                ) as response:
                    body = response.read(256 * 1024 + 1)
                    if len(body) > 256 * 1024:
                        raise HubError("Recipe exceeds 256 KiB")
                    document = json.loads(body)
                    self.validate(document, repo, revision)
                    entry = {
                        "document": document,
                        "etag": response.headers.get("ETag"),
                        "saved_at": time.time(),
                    }
                break
            except HTTPError as error:
                if error.code == 304 and cached:
                    entry = {**cached, "saved_at": time.time()}
                    break
                if error.code not in (429, 502, 503, 504) or attempt == self.retries:
                    raise HubError(
                        f"Hub returned HTTP {error.code} for {repo}"
                    ) from error
                delay = 2**attempt
                retry_after = error.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(0, float(retry_after))
                    except ValueError:
                        try:
                            delay = max(
                                0,
                                (
                                    parsedate_to_datetime(retry_after)
                                    - datetime.now(timezone.utc)
                                ).total_seconds(),
                            )
                        except (ValueError, TypeError):
                            pass
                if delay > 30:
                    raise HubError(
                        f"Hub asks to retry after {retry_after}; try again later"
                    ) from error
                time.sleep(delay)
            except (OSError, ValueError) as error:
                raise HubError(f"Could not download recipe: {error}") from error
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.cache_dir, delete=False
            ) as output:
                temporary = Path(output.name)
                json.dump(entry, output)
            os.replace(temporary, path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
        return entry["document"]

    @staticmethod
    def validate(document, repo, revision):
        if (
            not isinstance(document, dict)
            or document.get("format") not in ("neuralhub/v1", "neuralhub/evidence-v1")
            or document.get("id") != repo
        ):
            raise HubError("Invalid NeuralHub recipe")
        if type(document.get("revision")) is not int or document["revision"] < 1:
            raise HubError("Invalid recipe revision")
        if revision is not None and document["revision"] != revision:
            raise HubError("Recipe revision does not match request")
        if not isinstance(document.get("model"), dict) or not isinstance(
            document.get("positions"), list
        ):
            raise HubError("Recipe is missing model or positions")
