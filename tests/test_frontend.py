"""Frontend behaviour, run in Node with a minimal DOM stub (no browser needed).

GitHub's ubuntu runners ship Node; the test is skipped only when no `node` binary is installed."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "frontend" / "script.js"

HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const names = JSON.parse(process.argv[2]);
const el = () => ({ value: '', textContent: '', innerHTML: '', className: '', style: {},
  classList: { remove() {}, add() {} }, addEventListener() {}, appendChild() {} });
const elements = {};
const sent = [];
class FormData {
  constructor() { this.items = []; }
  append(key, value) { this.items.push([key, value]); }
  getAll(key) { return this.items.filter(([k]) => k === key).map(([, v]) => v); }
}
const context = {
  window: { location: { protocol: 'http:', origin: 'http://127.0.0.1:8000' } },
  document: { getElementById: (id) => (elements[id] = elements[id] || el()), createElement: el },
  sessionStorage: { getItem: () => '', setItem() {} },
  console: { warn() {}, error() {}, log() {} },
  FormData,
  fetch: async (url, opts) => {
    sent.push(...opts.body.getAll('files').map((f) => f.name));
    return { ok: true, json: async () => [] };
  },
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[3], 'utf8'), context);
elements.fileInput.files = names.map((name) => ({ name }));
vm.runInContext('uploadFile()', context).then(() => {
  console.log(JSON.stringify({ sent, error: elements.errorDisplay.textContent }));
});
"""


def _upload(names, tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    harness = tmp_path / "harness.js"
    harness.write_text(HARNESS, encoding="utf-8")
    out = subprocess.run(
        [node, str(harness), json.dumps(names), str(SCRIPT)], capture_output=True, text=True, timeout=30, check=True
    )
    return json.loads(out.stdout)


def test_frontend_accepts_md_extension_in_any_case(tmp_path):
    """Regression: the UI only sent names ending in lowercase `.md`, so `NOTES.MD` was dropped before upload."""
    result = _upload(["NOTES.MD", "Readme.Md", "plain.md", "image.png"], tmp_path)
    assert result["sent"] == ["NOTES.MD", "Readme.Md", "plain.md"]


def test_frontend_reports_when_no_md_file_is_selected(tmp_path):
    result = _upload(["image.png", "notes.txt"], tmp_path)
    assert result["sent"] == []
    assert "有效的 .md" in result["error"]
