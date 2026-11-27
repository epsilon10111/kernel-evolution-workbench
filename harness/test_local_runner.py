import hashlib
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness import local_runner as runner
from harness.local_campaign import DeviceUnavailable


CONFIG = dict(KEW_SSH_HOST="gpu-test", KEW_GPU_UUID="GPU-00000000-0000-0000-0000-000000000000",
              KEW_CUDA_IMAGE="example/cuda@sha256:" + "0" * 64, KEW_SSH_CONFIG="/tmp/ssh-config")


class GroupTransportTests(unittest.TestCase):
    def test_ssh_requires_a_host_gpu_and_pinned_image(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(runner.subprocess, "run") as invoke:
            with self.assertRaisesRegex(ValueError, "KEW_SSH_HOST"):
                runner._run_remote("hinge", [b"source"], [0])
            with patch.dict(os.environ, dict(CONFIG, KEW_CUDA_IMAGE="example/cuda:latest")):
                with self.assertRaisesRegex(ValueError, "KEW_CUDA_IMAGE"):
                    runner._run_remote("hinge", [b"source"], [0])
            invoke.assert_not_called()

    def test_ssh_uses_configured_device_without_agent_forwarding(self):
        packet = runner.subprocess.CompletedProcess([], 0, b"{}", b"")
        with patch.dict(os.environ, CONFIG), patch.object(runner.subprocess, "run", return_value=packet) as invoke:
            self.assertEqual(runner._run_remote("hinge", [b"source"], [0]), {})
        args = invoke.call_args.args[0]
        self.assertEqual(args[:3], ["ssh", "-F", CONFIG["KEW_SSH_CONFIG"]])
        self.assertIn("StrictHostKeyChecking=yes", args)
        self.assertIn("ForwardAgent=no", args)
        self.assertEqual(args[-2], CONFIG["KEW_SSH_HOST"])
        self.assertEqual(shlex.split(args[-1])[4:],
                         [CONFIG["KEW_GPU_UUID"], CONFIG["KEW_CUDA_IMAGE"], "hinge", "batch", "1", "0"])

    def test_rejects_source_change_before_ssh(self):
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "input.cu"
            source.write_text("changed")
            with patch.object(runner, "_run_remote") as invoke, self.assertRaises(ValueError):
                list(runner.evaluate_group([dict(problem="hinge", source_path=source, source_sha256="f"*64)]))
            invoke.assert_not_called()

    def test_each_abba_leg_requires_its_own_measurement(self):
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "s.cu"
            source.write_bytes(b"one")
            sha = hashlib.sha256(b"one").hexdigest()
            req = dict(problem="hinge", source_path=source, source_sha256=sha)
            gpu_uuid = CONFIG["KEW_GPU_UUID"]
            env = dict(gpu_uuid=gpu_uuid, driver_version="test",
                host_attestation=dict(remote_ssh=True, gpu_uuid=gpu_uuid, gpu="NVIDIA H20",
                    image_digest=CONFIG["KEW_CUDA_IMAGE"], image_id="test-image", driver_version="test"))
            row = dict(problem="hinge", kind="local_cuda_evaluation", source_sha256=sha,
                       environment=env, run_id="run", rankable=True)
            packet = dict(problem="hinge", kind="local_cuda_batch", reports=[row, row])
            with patch.dict(os.environ, CONFIG), patch.object(runner, "_run_remote", return_value=packet) as invoke:
                results = runner.evaluate_group([req, req])
                self.assertEqual(next(results), row)
                with self.assertRaises(DeviceUnavailable):
                    next(results)
                invoke.assert_called_once_with("hinge", [b"one"], [0, 0])


if __name__ == "__main__":
    unittest.main()
