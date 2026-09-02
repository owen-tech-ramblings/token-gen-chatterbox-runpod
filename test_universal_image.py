from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parent


class UniversalImageContractTests(unittest.TestCase):
    def test_dockerfile_pins_cuda_pytorch_architectures_and_source(self) -> None:
        dockerfile = (ROOT / "Dockerfile.qwen").read_text(encoding="utf-8")

        self.assertIn(
            "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:"
            "17e2934e1fa96152b14f78078bfbafd0f00f391df995dc6c641a720fce1202bb",
            dockerfile,
        )
        self.assertIn("https://download.pytorch.org/whl/cu128", dockerfile)
        self.assertIn("torch==2.11.0", dockerfile)
        self.assertIn("torchaudio==2.11.0", dockerfile)
        self.assertIn("ARG SOURCE_COMMIT", dockerfile)
        self.assertIn("org.opencontainers.image.revision", dockerfile)
        self.assertIn("_cuda_getArchFlags", dockerfile)
        for architecture in ("sm_80", "sm_86", "sm_90", "sm_100", "sm_120"):
            self.assertIn(architecture, dockerfile)
        self.assertLess(
            dockerfile.index("_cuda_getArchFlags"),
            dockerfile.index("snapshot_download"),
        )

    def test_workflow_is_master_only_rootless_and_commit_addressed(self) -> None:
        workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
        self.assertTrue(workflow_path.is_file())
        workflow = workflow_path.read_text(encoding="utf-8")

        self.assertRegex(
            workflow,
            re.compile(
                r"\Aname: Universal GPU image\s+on:\s+push:\s+branches:\s+- master\s+",
                re.MULTILINE,
            ),
        )
        self.assertNotIn("pull_request:", workflow)
        self.assertNotIn("workflow_dispatch:", workflow)
        self.assertNotIn("ubuntu-latest", workflow)
        self.assertIn("runs-on: [self-hosted, linux, x64, lil-zen-ci]", workflow)
        self.assertIn("contents: read", workflow)
        self.assertIn("packages: write", workflow)
        self.assertIn("rootlesskit --subid-source=static", workflow)
        self.assertIn("--oci-worker-snapshotter=native", workflow)
        self.assertIn("--oci-worker-no-process-sandbox", workflow)
        self.assertIn("--root /runner/buildkit-root", workflow)
        self.assertIn("secrets.GITHUB_TOKEN", workflow)
        self.assertIn("qwen3-base-${GITHUB_SHA}", workflow)
        self.assertIn("build-arg:SOURCE_COMMIT=${GITHUB_SHA}", workflow)
        for legacy in (
            "publish.yml",
            "publish-qwen.yml",
            "publish-audition-models.yml",
        ):
            self.assertFalse((workflow_path.parent / legacy).exists())


if __name__ == "__main__":
    unittest.main()
