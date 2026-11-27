import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.gateway import claude_env


class GatewayTests(unittest.TestCase):
    def test_only_active_profile_enters_claude_environment(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
            "XDG_CONFIG_HOME": temporary, "SSH_AUTH_SOCK": "unrelated", "ANTHROPIC_API_KEY": "unrelated"}):
            directory = Path(temporary) / "Claude-3p/configLibrary"
            directory.mkdir(parents=True)
            meta = directory / "_meta.json"
            meta.write_text(json.dumps({"appliedId": "active", "entries": [{"id": "active"}]}))
            meta.chmod(0o600)
            profile = directory / "active.json"
            data = {"inferenceProvider": "gateway", "inferenceCredentialKind": "static",
                    "inferenceGatewayBaseUrl": "https://example.invalid", "inferenceGatewayApiKey": " unit-test-only\n"}
            profile.write_text(json.dumps(data))
            profile.chmod(0o600)
            result = claude_env()
            self.assertEqual(result["ANTHROPIC_AUTH_TOKEN"], "unit-test-only")
            self.assertNotIn("ANTHROPIC_API_KEY", result)
            self.assertNotIn("SSH_AUTH_SOCK", result)
            profile.write_text(json.dumps({**data, "inferenceGatewayAuthScheme": "x-api-key"}))
            self.assertNotIn("ANTHROPIC_AUTH_TOKEN", claude_env())
            profile.chmod(0o644)
            with self.assertRaises(ValueError):
                claude_env()
            profile.chmod(0o600)
            profile.write_text(json.dumps({**data, "inferenceGatewayBaseUrl": "http://example.invalid"}))
            with self.assertRaises(ValueError):
                claude_env()
