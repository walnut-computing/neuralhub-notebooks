import json
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from neuralhub import HubClient, HubError

DOC = {
    "format": "neuralhub/v1",
    "id": "alice/test",
    "revision": 1,
    "model": {"huggingface": "test/model"},
    "positions": [],
}


class Reply:
    headers = {"ETag": '"test"'}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size):
        return json.dumps(DOC).encode()


class Downloads(unittest.TestCase):
    def test_cache_offline_and_revalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            client = HubClient(
                "http://localhost:8000/research/neuralhub", cache_dir=directory
            )
            with patch("neuralhub.hub.urlopen", return_value=Reply()) as request:
                self.assertEqual(client.neuron("alice/test", revision=1), DOC)
                self.assertEqual(client.neuron("alice/test", revision=1), DOC)
                self.assertEqual(request.call_count, 1)
                self.assertIn(
                    "/research/neuralhub/api/", request.call_args.args[0].full_url
                )
            with patch(
                "neuralhub.hub.urlopen", side_effect=HTTPError("url", 304, "", {}, None)
            ) as request:
                self.assertEqual(
                    client.neuron("alice/test", revision=1, refresh=True), DOC
                )
                self.assertEqual(
                    request.call_args.args[0].get_header("If-none-match"), '"test"'
                )
            self.assertEqual(client.neuron("alice/test", revision=1, offline=True), DOC)
            with self.assertRaises(HubError):
                client.neuron("alice/missing", offline=True)

    def test_retry_budget_and_retry_after(self):
        with tempfile.TemporaryDirectory() as directory:
            client = HubClient("https://example.com", cache_dir=directory)
            error = HTTPError("url", 429, "", {"Retry-After": "2"}, None)
            with (
                patch("neuralhub.hub.urlopen", side_effect=[error, Reply()]) as request,
                patch("neuralhub.hub.time.sleep") as sleep,
            ):
                self.assertEqual(client.neuron("alice/test"), DOC)
                sleep.assert_called_once_with(2)
                self.assertEqual(request.call_count, 2)
            with (
                patch("neuralhub.hub.urlopen", side_effect=error) as request,
                patch("neuralhub.hub.time.sleep"),
            ):
                with self.assertRaises(HubError):
                    client.neuron("alice/test", refresh=True)
                self.assertEqual(request.call_count, 3)

    def test_invalid_names_and_revision_mismatch(self):
        client = HubClient("https://example.com")
        with self.assertRaises(ValueError):
            client.neuron("../bad")
        with self.assertRaises(HubError):
            client.validate(DOC, "alice/test", 2)
