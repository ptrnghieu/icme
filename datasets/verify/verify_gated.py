"""L3 verification for gated datasets the user can access (IEMOCAP, MER2023-2026). Runs on the USER's Kaggle.

Outputs ONLY aggregate statistics (counts, class distributions, column names, folder structure). It never prints
samples, transcripts or per-utterance labels, to respect the licences (no redistribution of data or annotations).

Usage (Kaggle):  python verify_gated.py --iemocap /kaggle/input/<your-iemocap-folder> --mer --out /kaggle/working/verify_report.md
  --mer needs an HF token with access to MERChallenge/* in env HF_TOKEN (Kaggle secret).
"""
import argparse, collections, glob, io, json, os, re, sys, zipfile

L = []


def say(s=""):
    print(s); L.append(s)


# ----------------------------------------------------------------------------------------------------- IEMOCAP
SUMMARY = re.compile(r"^\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(Ses\d\d[FM]_\S+)\s+(\S+)\s+\[([^\]]*)\]")
EVAL = re.compile(r"^([CA])-([A-Za-z]+\d*)\s*:")


def find_iemocap_root(root):
    for dp, dn, fn in os.walk(root):
        if {"Session1", "Session2", "Session3", "Session4", "Session5"} <= set(dn):
            return dp
    return None


def verify_iemocap(root):
    say("## IEMOCAP")
    r = find_iemocap_root(root)
    if r is None:
        say(f"- NOT FOUND: no folder containing Session1..Session5 under {root}")
        return
    say(f"- root: `{r}`")
    summ, ncat, evaluators, sess_count = [], collections.Counter(), collections.Counter(), collections.Counter()
    for s in range(1, 6):
        files = sorted(glob.glob(os.path.join(r, f"Session{s}", "dialog", "EmoEvaluation", "*.txt")))
        cur = None
        for f in files:
            for line in open(f, errors="ignore"):
                m = SUMMARY.match(line.strip())
                if m:
                    cur = m.group(3); dims = [x.strip() for x in m.group(5).split(",")]
                    summ.append(dict(sess=s, utt=cur, label=m.group(4), improv="impro" in cur, ndims=len(dims)))
                    sess_count[s] += 1
                    continue
                e = EVAL.match(line.strip())
                if e and cur:
                    evaluators[(e.group(1), re.sub(r"\d+$", "", e.group(2)))] += 1
                    if e.group(1) == "C":
                        ncat[cur] += 1
        wav = len(glob.glob(os.path.join(r, f"Session{s}", "sentences", "wav", "*", "*.wav")))
        avi = len(glob.glob(os.path.join(r, f"Session{s}", "dialog", "avi", "*", "*.avi")))
        trn = len(glob.glob(os.path.join(r, f"Session{s}", "dialog", "transcriptions", "*.txt")))
        mocap = [d for d in os.listdir(os.path.join(r, f"Session{s}", "sentences")) if d.lower().startswith("mocap")] \
            if os.path.isdir(os.path.join(r, f"Session{s}", "sentences")) else []
        say(f"- Session{s}: EmoEvaluation files {len(files)}, utterance summaries {sess_count[s]}, sentence wav {wav}, "
            f"dialog avi {avi}, transcription files {trn}, MOCAP folders {sorted(mocap)}")
    lab = collections.Counter(x["label"] for x in summ)
    say(f"- total utterances with a summary line: {len(summ)} (improvised {sum(x['improv'] for x in summ)}, scripted {sum(not x['improv'] for x in summ)})")
    say(f"- consensus label counts: {dict(sorted(lab.items(), key=lambda kv: -kv[1]))}")
    four = sum(lab[k] for k in ["neu", "ang", "sad", "hap", "exc"])
    say(f"- 4-class (neu, ang, sad, hap+exc): {four} = neu {lab['neu']}, ang {lab['ang']}, sad {lab['sad']}, hap {lab['hap']}, exc {lab['exc']}")
    say(f"- 4-class without exc (neu, ang, sad, hap): {sum(lab[k] for k in ['neu', 'ang', 'sad', 'hap'])}")
    say(f"- 6-class (hap, sad, neu, ang, exc, fru): {sum(lab[k] for k in ['hap', 'sad', 'neu', 'ang', 'exc', 'fru'])}")
    dist = collections.Counter(ncat.values())
    say(f"- number of categorical (C-) evaluations per utterance: {dict(sorted(dist.items()))}")
    say(f"- evaluator id prefixes (type, prefix) -> lines: {dict(evaluators)}")
    say(f"- summary-line dimensional entries per utterance: {dict(collections.Counter(x['ndims'] for x in summ))}")


# ----------------------------------------------------------------------------------------------------- MER
def mer_fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem(token=os.environ.get("HF_TOKEN"))


def describe_csv(fs, path):
    import pandas as pd
    with fs.open(path, "rb") as f:
        D = pd.read_csv(f)
    say(f"  - `{path.split('/')[-1]}`: {len(D)} rows; columns {list(D.columns)}")
    for c in D.columns:
        u = D[c].nunique()
        if 1 < u <= 12 and (pd.api.types.is_string_dtype(D[c]) or pd.api.types.is_object_dtype(D[c])):
            say(f"    - `{c}` distribution: {D[c].value_counts().to_dict()}")
    return D


def zip_names(fs, path):
    with fs.open(path, "rb", block_size=1 << 20) as f:
        return zipfile.ZipFile(f).namelist()


def verify_mer():
    fs = mer_fs()
    for repo in ["MER2023", "MER2024", "MER2025", "MER2026"]:
        base = f"datasets/MERChallenge/{repo}"
        say(f"\n## {repo}")
        try:
            files = fs.ls(base, detail=True)
        except Exception as e:  # noqa: BLE001
            say(f"- cannot list {base}: {type(e).__name__}: {e}"); continue
        say(f"- top-level entries: {len(files)}")
        try:
            with fs.open(f"{base}/README_AFTER_APPROVAL.md", "r") as f:
                txt = f.read()
            say(f"- README_AFTER_APPROVAL.md readable: {len(txt)} chars")
            for line in txt.splitlines():  # headings and protocol-relevant lines only; NEVER credentials
                if re.search(r"(?i)pass(word)?|pwd|密码|口令|key|token|secret", line):
                    continue
                if line.startswith("#") or re.search(r"(?i)train|test|split|fold|label|metric|f1|accuracy", line):
                    say(f"    > {line.strip()[:200]}")
        except Exception as e:  # noqa: BLE001
            say(f"- README_AFTER_APPROVAL.md not readable: {type(e).__name__}")
        for fdesc in files:
            p = fdesc["name"]
            if p.endswith(".csv"):
                try:
                    describe_csv(fs, p)
                except Exception as e:  # noqa: BLE001
                    say(f"  - {p}: {type(e).__name__}: {e}")
        if repo == "MER2024":
            try:
                n = zip_names(fs, f"{base}/video-labeled.zip")
                vids = [x for x in n if x.lower().endswith((".mp4", ".avi", ".mkv"))]
                say(f"- video-labeled.zip: {len(n)} entries, {len(vids)} video files")
            except Exception as e:  # noqa: BLE001
                say(f"- video-labeled.zip listing failed: {type(e).__name__}: {e}")
        if repo == "MER2023":
            try:
                n = zip_names(fs, f"{base}/test-labels.zip")
                say(f"- test-labels.zip entries: {n}")
            except Exception as e:  # noqa: BLE001
                say(f"- test-labels.zip listing failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iemocap", default="")
    ap.add_argument("--mer", action="store_true")
    ap.add_argument("--out", default="verify_report.md")
    a = ap.parse_args()
    say("# L3 verification report (aggregate statistics only)")
    if a.iemocap:
        verify_iemocap(a.iemocap)
    if a.mer:
        verify_mer()
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"\nreport -> {a.out}")
