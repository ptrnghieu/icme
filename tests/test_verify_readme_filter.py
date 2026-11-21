"""Regression test: README lines with archive passwords must never be printed (bug found 2026-11-21)."""
import importlib.util, pathlib

spec = importlib.util.spec_from_file_location("vg", pathlib.Path(__file__).resolve().parents[1] / "datasets/verify/verify_gated.py")
vg = importlib.util.module_from_spec(spec); spec.loader.exec_module(vg)

FAKE = """# Dataset Download
## Files Provided
| `train.zip` | `111222333` |
| `labels.zip` | `abcd1234efgh` |
7z x train.zip -p111222333 -o./out
### How to Extract
#### Password: 999888777
## Package Layout
"""


def test_no_secret_lines():
    out = vg.safe_readme_lines(FAKE)
    joined = "\n".join(out)
    for bad in ["111222333", "abcd1234efgh", "999888777", "-p", "`"]:
        assert bad not in joined
    assert "# Dataset Download" in out and "## Package Layout" in out


if __name__ == "__main__":
    test_no_secret_lines(); print("ok")
