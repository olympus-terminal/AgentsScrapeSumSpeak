"""Claude CLI fallback: isolation flags, result validation and routing.

Uses a fake ``claude`` executable; no model is called.
"""

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import claude_backend
import codex_backend
import digest
import judge
from queue_manager import ContentItem, ContentType

FAKE_CLAUDE = """#!{python}
import json, os, sys
record = {{"argv": sys.argv[1:], "stdin": sys.stdin.read(),
          "api_key": os.environ.get("ANTHROPIC_API_KEY")}}
with open(os.environ["FAKE_CLAUDE_RECORD"], "a") as f:
    f.write(json.dumps(record) + "\\n")
if sys.argv[1:3] == ["auth", "status"]:
    print(os.environ.get("FAKE_CLAUDE_AUTH", '{{"loggedIn": true, "authMethod": "claude.ai", "subscriptionType": "max"}}'))
    sys.exit(0)
print(os.environ["FAKE_CLAUDE_REPLY"])
sys.exit(int(os.environ.get("FAKE_CLAUDE_EXIT", "0")))
"""


def _reply(structured=None, *, model="claude-sonnet-5-5", **overrides):
    reply = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": json.dumps(structured) if structured is not None else "plain text",
        "structured_output": structured,
        "permission_denials": [],
        "modelUsage": {model: {"outputTokens": 10}},
        "usage": {"input_tokens": 1, "output_tokens": 10},
        "total_cost_usd": 0.001,
    }
    reply.update(overrides)
    return json.dumps(reply)


class FakeClaudeMixin:
    def start_fake_claude(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        exe = tmp / "claude"
        exe.write_text(FAKE_CLAUDE.format(python=sys.executable))
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        self.record = tmp / "record.jsonl"
        self.schema = tmp / "schema.json"
        self.schema.write_text(
            '{"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}'
        )
        self.env = patch.dict(os.environ, {
            "FAKE_CLAUDE_RECORD": str(self.record),
            "FAKE_CLAUDE_REPLY": _reply({"ok": True}),
            "ANTHROPIC_API_KEY": "sk-should-not-be-passed",
        })
        self.env.start()
        self.patches = [
            patch.object(claude_backend, "CLAUDE_BIN", str(exe)),
            patch.object(claude_backend, "CLAUDE_MODEL", "claude-sonnet-5-5"),
            patch.object(claude_backend, "CLAUDE_EFFORT", ""),
            patch.object(codex_backend, "CLAUDE_MODEL", "claude-sonnet-5-5"),
        ]
        for p in self.patches:
            p.start()

    def stop_fake_claude(self):
        for p in reversed(self.patches):
            p.stop()
        self.env.stop()
        self.tmp.cleanup()

    def calls(self):
        if not self.record.exists():
            return []
        return [json.loads(line) for line in self.record.read_text().splitlines()]


class ClaudeTransformTests(FakeClaudeMixin, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.start_fake_claude()

    def tearDown(self):
        self.stop_fake_claude()

    async def test_structured_call_is_isolated_and_keeps_data_on_stdin(self):
        result = await claude_backend.transform(
            "trusted instructions", "untrusted source text",
            timeout_seconds=30, output_schema=self.schema,
        )
        self.assertEqual(json.loads(result), {"ok": True})
        call = self.calls()[0]
        argv = call["argv"]
        self.assertEqual(call["stdin"], "untrusted source text")
        self.assertNotIn("untrusted source text", " ".join(argv))
        self.assertIsNone(call["api_key"])
        for flag in ("-p", "--safe-mode", "--strict-mcp-config",
                     "--disable-slash-commands", "--no-session-persistence"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertEqual(argv[argv.index("--model") + 1], "claude-sonnet-5-5")
        self.assertEqual(argv[argv.index("--system-prompt") + 1], "trusted instructions")
        self.assertEqual(json.loads(argv[argv.index("--json-schema") + 1]), {"type": "object"})

    async def test_rejects_error_results_wrong_model_and_tool_use(self):
        bad_replies = [
            _reply({"ok": True}, is_error=True, subtype="error_max_turns"),
            _reply({"ok": True}, model="claude-haiku-5-5"),
            _reply({"ok": True}, permission_denials=[{"tool_name": "Bash"}]),
            _reply(None),
            "not json",
        ]
        for reply in bad_replies:
            with self.subTest(reply=reply[:40]):
                os.environ["FAKE_CLAUDE_REPLY"] = reply
                with self.assertRaises(claude_backend.ClaudeCLIError):
                    await claude_backend.transform(
                        "instructions", "data", timeout_seconds=30,
                        output_schema=self.schema,
                    )

    async def test_nonzero_exit_is_an_error(self):
        os.environ["FAKE_CLAUDE_EXIT"] = "1"
        try:
            with self.assertRaises(claude_backend.ClaudeCLIError):
                await claude_backend.transform("instructions", "data", timeout_seconds=30)
        finally:
            del os.environ["FAKE_CLAUDE_EXIT"]

    async def test_preflight_requires_login(self):
        self.assertEqual((await claude_backend.preflight())[0], "claude-sonnet-5-5")
        os.environ["FAKE_CLAUDE_AUTH"] = '{"loggedIn": false}'
        try:
            with self.assertRaises(claude_backend.ClaudeCLIError):
                await claude_backend.preflight()
        finally:
            del os.environ["FAKE_CLAUDE_AUTH"]


class FallbackRoutingTests(FakeClaudeMixin, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.start_fake_claude()
        self.routing = [
            patch.object(codex_backend, "LLM_BACKEND", "openai_compatible"),
            patch.object(codex_backend, "LLM_FALLBACK_BACKEND", "claude_cli"),
            patch.object(codex_backend, "_cycle_backend", None),
        ]
        for p in self.routing:
            p.start()

    def tearDown(self):
        for p in reversed(self.routing):
            p.stop()
        self.stop_fake_claude()

    async def test_failed_primary_preflight_moves_the_cycle_to_claude(self):
        failing = AsyncMock(side_effect=codex_backend.CodexExecError("HTTP 502"))
        with patch("compatible_backend.preflight", failing):
            await codex_backend.preflight_codex()
        self.assertEqual(codex_backend.primary_backend(), "claude_cli")
        self.assertIsNone(codex_backend.fallback_backend())

        result = await codex_backend.run_codex(
            "instructions", "data", timeout_seconds=30, output_schema=self.schema,
        )
        self.assertEqual(json.loads(result), {"ok": True})
        self.assertEqual(codex_backend.backend_identity()["model"], "claude-sonnet-5-5")

        # The next cycle tries the primary again.
        with patch("compatible_backend.preflight", AsyncMock(return_value=("mimo", "proxy"))):
            await codex_backend.preflight_codex()
        self.assertEqual(codex_backend.primary_backend(), "openai_compatible")

    async def test_preflight_fails_when_both_backends_fail(self):
        os.environ["FAKE_CLAUDE_AUTH"] = '{"loggedIn": false}'
        try:
            failing = AsyncMock(side_effect=codex_backend.CodexExecError("HTTP 502"))
            with patch("compatible_backend.preflight", failing):
                with self.assertRaisesRegex(codex_backend.CodexExecError, "fallback claude_cli"):
                    await codex_backend.preflight_codex()
        finally:
            del os.environ["FAKE_CLAUDE_AUTH"]

    async def test_failed_primary_call_is_retried_on_claude(self):
        from compatible_backend import ModelError
        failing = AsyncMock(side_effect=ModelError("Model endpoint request failed"))
        with patch("compatible_backend.transform", failing):
            result = await codex_backend.run_codex(
                "instructions", "data", timeout_seconds=30, output_schema=self.schema,
            )
        self.assertEqual(json.loads(result), {"ok": True})
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(codex_backend.backend_identity()["backend"], "claude_cli")

    async def test_explicit_backend_does_not_fall_back(self):
        from compatible_backend import ModelError
        failing = AsyncMock(side_effect=ModelError("down"))
        with patch("compatible_backend.transform", failing):
            with self.assertRaises(codex_backend.CodexExecError):
                await codex_backend.run_codex(
                    "instructions", "data", timeout_seconds=30,
                    backend="openai_compatible",
                )
        self.assertEqual(self.calls(), [])

    async def test_writer_hands_an_incomplete_segment_to_claude(self):
        prose = " ".join(["word"] * 60)
        incomplete = json.dumps({"covered_item_indices": [1], "segment": prose})
        complete = {"covered_item_indices": [1, 2], "segment": prose}
        os.environ["FAKE_CLAUDE_REPLY"] = _reply(complete)
        used = []
        with (
            patch("compatible_backend.transform", AsyncMock(return_value=incomplete)),
            patch.object(digest, "CODEX_MAX_WRITER_ATTEMPTS", 2),
            patch.object(digest, "CODEX_RETRY_DELAY_SECONDS", 0.01),
        ):
            output = await digest._call_codex_scriptwriter(
                "source material", expected_indices={1, 2}, minimum_words=50,
                used_backends=used,
            )
        self.assertEqual(output, prose)
        self.assertEqual(used, ["claude_cli"])
        self.assertEqual(len(self.calls()), 1)

    async def test_writer_returns_none_when_claude_also_fails(self):
        incomplete = {"covered_item_indices": [1], "segment": "too short"}
        os.environ["FAKE_CLAUDE_REPLY"] = _reply(incomplete)
        with (
            patch("compatible_backend.transform", AsyncMock(return_value=json.dumps(incomplete))),
            patch.object(digest, "CODEX_MAX_WRITER_ATTEMPTS", 1),
            patch.object(digest, "CLAUDE_MAX_FALLBACK_ATTEMPTS", 2),
            patch.object(digest, "CODEX_RETRY_DELAY_SECONDS", 0.01),
        ):
            output = await digest._call_codex_scriptwriter(
                "source material", expected_indices={1, 2},
            )
        self.assertIsNone(output)
        self.assertEqual(len(self.calls()), 2)

    async def test_judge_sends_an_invalid_batch_to_claude(self):
        item = ContentItem(
            url="https://example.org/paper", title="Research item",
            source="arxiv", content_type=ContentType.PAPER, raw_text="Abstract text.",
        )
        verdict = {
            "results": [{
                "item_index": 0, "technical_depth": 8, "novelty": 7, "relevance": 8,
                "verdict": "keep", "reason": "Solid method.",
                "key_contributions": "A new method.", "summary": "A summary.",
            }]
        }
        os.environ["FAKE_CLAUDE_REPLY"] = _reply(verdict)
        with patch("compatible_backend.transform", AsyncMock(return_value='{"results": []}')):
            results = await judge._call_codex_judge([item])
        self.assertIsNotNone(results)
        self.assertEqual(results[0]["verdict"], "keep")
        self.assertEqual(len(self.calls()), 1)


if __name__ == "__main__":
    unittest.main()
