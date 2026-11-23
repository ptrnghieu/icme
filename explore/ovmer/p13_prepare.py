"""P13 data preparation (runs on the USER's Kaggle; see p13_mbr.py for the pre-registered design).
Builds items.jsonl = {key, ds, video, sub, gt} for (a) OV-MERD clips present in MER2024 video-labeled.zip and (b) 400 EmoS-MELD clips.
Usage: python p13_prepare.py --affectgpt DIR --emos DIR --mer24zip ZIP --meld_tar URL_OR_PATH --work DIR --out items.jsonl
"""
import argparse, csv, glob, json, os, subprocess, zipfile
import numpy as np


def ovmerd(a):
    R = list(csv.DictReader(open(os.path.join(a.affectgpt, "AffectGPT-RL/dataset/ovmerdplus-emotion.csv"), encoding="utf-8")))
    sub = {r["name"]: r["chinese"] for r in csv.DictReader(open(os.path.join(a.affectgpt, "AffectGPT-RL/dataset/subtitle_chieng.csv"), encoding="utf-8"))}
    z = zipfile.ZipFile(a.mer24zip); byname = {os.path.splitext(os.path.basename(n))[0]: n for n in z.namelist() if n.lower().endswith((".mp4", ".avi", ".mkv"))}
    out_dir = os.path.join(a.work, "ovmerd"); os.makedirs(out_dir, exist_ok=True); items = []
    for r in R:
        n = r["name"]
        if n not in byname:
            continue
        p = os.path.join(out_dir, os.path.basename(byname[n]))
        if not os.path.exists(p):
            with open(p, "wb") as f:
                f.write(z.read(byname[n]))
        items.append(dict(key=f"ovmerd:{n}", ds="OV-MERD", video=p, sub=sub.get(n, ""), gt=[x.strip() for x in r["openset"].split(",") if x.strip()]))
    print("OV-MERD clips with video:", len(items), "/", len(R), flush=True)
    return items


def emos_meld(a):
    import pandas as pd
    frames = [pd.read_excel(os.path.join(a.emos, f"datasets/data-final/English/goemotion/goemotion_soft_labels_{s}.xlsx")) for s in ("first3000", "last2000")]
    d = pd.concat(frames, ignore_index=True); cols = [c for c in d.columns if c.startswith("goemotion_prob_")]
    rows = []
    for _, r in d.iterrows():
        gt = [c.replace("goemotion_prob_", "") for c in cols if float(r[c] or 0) >= 0.5 and c.split("_")[-1] not in ("neutral", "unsure")]
        if gt:
            rows.append(dict(dia=int(r["Dialogue_ID"]), utt=int(r["Utterance_ID"]), text=str(r["Utterance"]), gt=gt))
    print("EmoS-MELD clips with a non-empty GT:", len(rows), "/", len(d), flush=True)
    rng = np.random.default_rng(0); pick = [rows[i] for i in sorted(rng.choice(len(rows), min(400, len(rows)), replace=False))]
    out_dir = os.path.join(a.work, "meld"); os.makedirs(out_dir, exist_ok=True)
    want = [f"train_splits/dia{r['dia']}_utt{r['utt']}.mp4" for r in pick]
    missing = [w for w in want if not os.path.exists(os.path.join(out_dir, w))]
    if missing:
        lst = os.path.join(a.work, "meld_list.txt"); open(lst, "w").write("\n".join(missing) + "\n")
        src = f"curl -sL '{a.meld_tar}'" if a.meld_tar.startswith("http") else f"cat '{a.meld_tar}'"
        cmd = f"{src} | tar -xzO MELD.Raw/train.tar.gz | tar -xz -C '{out_dir}' -T '{lst}'"
        print("extracting", len(missing), "MELD clips (streams the 10.9 GB archive)...", flush=True)
        subprocess.run(cmd, shell=True, check=False)
    items = []
    for r, w in zip(pick, want):
        p = os.path.join(out_dir, w)
        if os.path.exists(p):
            items.append(dict(key=f"meld:{r['dia']}_{r['utt']}", ds="EmoS-MELD", video=p, sub=r["text"], gt=r["gt"]))
    print("EmoS-MELD clips with video:", len(items), "/", len(pick), flush=True)
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("affectgpt", "emos", "mer24zip", "meld_tar", "work", "out"):
        ap.add_argument("--" + k)
    a = ap.parse_args(); os.makedirs(a.work, exist_ok=True)
    items = ovmerd(a) + emos_meld(a)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("items:", len(items))
