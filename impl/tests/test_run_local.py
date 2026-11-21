import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import server
from core import pairs


LAUNCHER = Path(__file__).resolve().parents[1] / "run-local.ps1"


class RunLocalTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt" and LAUNCHER.exists(),
                         "requires Windows PowerShell and the source checkout")
    def test_fresh_copy_and_resync(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "impl"
            env = dict(os.environ, T8_SCRIPT=str(LAUNCHER), T8_RUN=root,
                       T8_PYTHON=sys.executable)

            def launch(args):
                command = (
                    "$ErrorActionPreference = 'Stop'; $tokens = $null; $errors = $null; "
                    "[void][Management.Automation.Language.Parser]::ParseFile("
                    "$env:T8_SCRIPT, [ref]$tokens, [ref]$errors); "
                    "if ($errors.Count) { throw ($errors -join [Environment]::NewLine) }; "
                    "& $env:T8_SCRIPT -RunRoot $env:T8_RUN -Python $env:T8_PYTHON "
                    f"-CliArgs @({args}); exit $LASTEXITCODE"
                )
                return subprocess.check_output(
                    ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "RemoteSigned",
                     "-Command", command], env=env, stderr=subprocess.STDOUT, timeout=60)

            self.assertIn(b"setup", launch("'--help'"))
            self.assertFalse((target / "artifacts").exists())
            markers = [target / sub / "t8-marker" for sub in
                       ("artifacts/results", "stylize/weights", "stylize/samples")]
            for marker in markers:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_bytes(b"local resource")
            self.assertIn(b"--size", launch("'stylize','bench','--help'"))
            for marker in markers:
                self.assertEqual(marker.read_bytes(), b"local resource")
            self.assertEqual({p.stem for p in (target / "pairs").glob("*.json")},
                             set(pairs.list_pairs()))
            subprocess.run([sys.executable, "-B", "-c",
                            "import server, stylize.adain; "
                            "from core.pairs import list_pairs, load_pair; "
                            "[load_pair(p) for p in list_pairs()]"],
                           cwd=target, check=True, timeout=60)

    def test_e4_uses_only_selected_pair(self):
        with tempfile.TemporaryDirectory() as root, patch.object(pairs, "ART", Path(root)), \
                patch.object(server, "ART", Path(root)), \
                patch.object(server, "PAIRS", server._pair_registry()), \
                patch.object(server, "_runtime", None), patch.object(server, "_runtime_pair", None):
            def check(pair_id, expected):
                state = server.build_state(pair_id)
                card = next(e for e in state["experiments"] if e["code"] == "E4")
                self.assertEqual(state["layers"], expected)
                self.assertEqual(card["status"], "done" if expected else "todo")
                if expected:
                    self.assertIn(expected["summary"]["most_drifted_layer"], card["note"])
                else:
                    self.assertIn("未开始", card["note"])

            for pid in pairs.list_pairs():
                check(pid, None)
            measured = {}
            for pid in ("horse2zebra", "zebra2horse"):
                data = {"summary": {"most_drifted_layer": pid, "most_drifted_cosine": 0.96}}
                path = server.PAIRS[pid].paths.results / "layers.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(data), encoding="utf-8")
                measured[pid] = data
            for pid in pairs.list_pairs():
                check(pid, measured.get(pid))
            check(None, measured[pairs.DEFAULT_PAIR])
