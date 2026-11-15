"""Claude review of TTS screening flags. Claude is always mocked here."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import digest
import tts_review

NARRATION = (
    "AI Research Briefing. The model reaches zero point eight eight eight on the benchmark. "
    "The work comes from Jiaqi Tang, Wei Wei and colleagues at a university lab."
)
TEXT_ISSUES = [
    {"line": 1, "pattern": "WORD_REPEAT", "description": "Repeated word -- TTS may loop",
     "match": "eight eight eight", "context": "reaches zero point eight eight eight. ",
     "chunk_file": "chunk_0001.txt"},
    {"line": 1, "pattern": "WORD_REPEAT", "description": "Repeated word -- TTS may loop",
     "match": "Wei Wei", "context": "Jiaqi Tang, Wei Wei and colleagues", "chunk_file": "chunk_0002.txt"},
]
NOTES = "# Sources\n- Authors: Jiaqi Tang; Wei Wei; Baoxi Li\n- Unrelated line\n"


def verdict(decision="approve", kinds=("false_alarm", "false_alarm")):
    return json.dumps({
        "issues": [{"index": i, "verdict": k, "reason": "r"} for i, k in enumerate(kinds)],
        "decision": decision, "summary": "s",
    })


class PayloadAndGateTests(unittest.TestCase):
    def test_payload_widens_context_and_adds_matching_note_lines(self):
        payload = tts_review.build_payload({"text_issues": TEXT_ISSUES}, NARRATION, NOTES)
        first, second = payload["flagged_issues"]
        self.assertEqual(first["voice_text"], TEXT_ISSUES[0]["context"])
        self.assertIn("Jiaqi Tang", second["script_context"])  # located from the words around the flag
        self.assertEqual(first["source_note_lines"], [])  # "eight" is not in the notes
        self.assertEqual(second["source_note_lines"], ["- Authors: Jiaqi Tang; Wei Wei; Baoxi Li"])

    def test_gate_overrides_claude_on_problems_or_missing_issues(self):
        report = {"text_issues": TEXT_ISSUES}
        ok = json.loads(verdict())
        self.assertEqual(tts_review._decide(report, ok), "approve")
        problem = json.loads(verdict(kinds=("false_alarm", "listener_problem")))
        self.assertEqual(tts_review._decide(report, problem), "keep_review")
        missing = json.loads(verdict())
        missing["issues"].pop()
        self.assertEqual(tts_review._decide(report, missing), "keep_review")


class ReviewScreeningTests(unittest.IsolatedAsyncioTestCase):
    def _files(self, root: Path, report: dict):
        report_path = root / "hallucination_report.json"
        report_path.write_text(json.dumps(report))
        (root / "digest.txt").write_text(NARRATION)
        (root / "notes.md").write_text(NOTES)
        return report_path, root / "digest.txt", root / "notes.md"

    async def test_off_without_model_and_never_for_audio_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._files(Path(tmp), {"text_issues": TEXT_ISSUES,
                                            "audio_issues": [{"type": "INCOMPLETE_AUDIO_CHUNKS"}]})
            call = AsyncMock(return_value=verdict())
            with patch.object(tts_review.claude_backend, "transform", call):
                with patch.object(tts_review, "TTS_REVIEW_MODEL", ""):
                    self.assertIsNone(await tts_review.review_screening(*paths))
                with patch.object(tts_review, "TTS_REVIEW_MODEL", "claude-opus-5-5"):
                    self.assertIsNone(await tts_review.review_screening(*paths))
            call.assert_not_called()

    async def test_records_decision_with_requested_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._files(Path(tmp), {"text_issues": TEXT_ISSUES, "audio_issues": []})
            call = AsyncMock(return_value=verdict())
            with patch.object(tts_review.claude_backend, "transform", call), \
                    patch.object(tts_review, "TTS_REVIEW_MODEL", "claude-opus-5-5"):
                record = await tts_review.review_screening(*paths)
            self.assertEqual(record["decision"], "approve")
            self.assertEqual(call.call_args.kwargs["model"], "claude-opus-5-5")
            self.assertEqual(record["issues"][1]["match"], "Wei Wei")
            saved = json.loads((Path(tmp) / "claude_review.json").read_text())
            self.assertEqual(saved["decision"], "approve")

    def test_sync_wrapper_turns_errors_into_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._files(Path(tmp), {"text_issues": TEXT_ISSUES, "audio_issues": []})
            call = AsyncMock(side_effect=RuntimeError("CLI down"))
            with patch.object(tts_review.claude_backend, "transform", call), \
                    patch.object(tts_review, "TTS_REVIEW_MODEL", "claude-opus-5-5"), \
                    self.assertLogs("tts_review", "WARNING"):
                self.assertIsNone(tts_review.review_screening_sync(*paths))


class DigestIntegrationTests(unittest.TestCase):
    """synthesize_audio with a fake toolkit that flags two text issues."""

    def _run(self, claude_reply):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "audio").mkdir()
        script_path = root / "digest.txt"
        script_path.write_text(NARRATION)
        notes_path = script_path.with_suffix(".md")
        notes_path.write_text(NOTES)
        fake = root / "fake_tts.py"
        report = {"metadata": {"chunks": 1}, "text_issues": TEXT_ISSUES, "audio_issues": [],
                  "summary": {"screened_text_chunks": 1, "screened_audio_chunks": 1,
                              "status": "REVIEW_NEEDED"}}
        fake.write_text("\n".join([
            "import json, sys",
            "from pathlib import Path",
            "args = sys.argv",
            f"Path(args[args.index('--output') + 1]).write_text({NARRATION!r})",
            "out = Path('fake_voice_output'); out.mkdir(exist_ok=True)",
            "(out / 'digest_complete.mp3').write_bytes(b'x' * 2048)",
            f"(out / 'hallucination_report.json').write_text(json.dumps({report!r}))",
        ]))
        call = AsyncMock(side_effect=claude_reply)
        with (
            patch.object(digest, "AUDIO_DIR", root / "audio"),
            patch.object(digest, "DELIVERY_DIR", root),
            patch.object(digest, "TTS_CONDA_PYTHON", sys.executable),
            patch.object(digest, "TTS_SCRIPT", str(fake)),
            patch.object(digest, "TTS_SCRIPT_DIR", str(root)),
            patch.object(digest, "TTS_VOICE", "p246"),
            patch.object(digest, "TTS_FORMAT", "mp3"),
            patch.object(digest, "TTS_ENABLED", True),
            patch.object(digest, "_probe_audio_duration", return_value=12.3),
            patch.object(tts_review, "TTS_REVIEW_MODEL", "claude-opus-5-5"),
            patch.object(tts_review.claude_backend, "transform", call),
        ):
            return digest.synthesize_audio(script_path, notes_path), call

    def test_claude_approval_publishes_without_review_suffix(self):
        output, call = self._run([verdict()])
        call.assert_called_once()
        self.assertTrue(output.approved)
        self.assertEqual(output.qc_status, "claude_reviewed")
        self.assertFalse(output.path.name.endswith("_REVIEW_NEEDED.mp3"))
        manifest = json.loads(output.manifest_path.read_text())
        self.assertEqual(manifest["status"], "claude_reviewed")
        self.assertEqual(manifest["claude_review"]["model"], "claude-opus-5-5")

    def test_claude_keep_review_or_failure_keeps_the_flag(self):
        for reply in ([verdict("keep_review", ("false_alarm", "listener_problem"))],
                      RuntimeError("CLI down")):
            with self.subTest(reply=type(reply).__name__):
                output, _ = self._run(reply)
                self.assertFalse(output.approved)
                self.assertEqual(output.qc_status, "review_needed")
                self.assertTrue(output.path.name.endswith("_REVIEW_NEEDED.mp3"))


if __name__ == "__main__":
    unittest.main()
