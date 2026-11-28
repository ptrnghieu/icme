"""P17 data preparation (runs on the user's rented GPU machine).

Builds items.jsonl = {key, ds, video, sub, gt} for:
  OV-MERD+   (532; GT AffectGPT-RL/dataset/ovmerdplus-emotion.csv)
  MER2025-FG (1,000; GT AffectGPT-RL/dataset/mer2025ov1000-emotion.csv)
  EmoS-MELD  (all MELD-train utterances in EmoS with a non-empty GT: GoEmotions labels with annotator share >= 0.5, excluding
              neutral and unsure; 2,732 clips)
Subtitles: ENGLISH for the MER sets (AffectGPT-RL/dataset/subtitle_chieng.csv, column 'english', as in the official
ovmerdplus_dataset.py); the MELD utterance text for EmoS-MELD.
Media: MER2026 video_7z/video_track2_train_human/video_split.zip (OV-MERD+ and MER2025-FG); MELD.Raw.tar.gz (MELD train) streamed.
Usage: python p17_prepare.py --affectgpt DIR --emos DIR --mer26zip ZIP --meld_tar URL --work DIR --out items.jsonl [--sets A,B]
"""
import argparse, csv, json, os, subprocess, zipfile
import pandas as pd


def mer_sets(a):
    sub = {r["name"]: (r.get("english") or "").strip() for r in csv.DictReader(open(os.path.join(a.affectgpt, "AffectGPT-RL/dataset/subtitle_chieng.csv"), encoding="utf-8"))}
    z = zipfile.ZipFile(a.mer26zip)
    if any(zi.flag_bits & 0x1 for zi in z.infolist()[:20]):
        if not a.zip_pw:
            raise SystemExit("MER2026 video zip is encrypted: export MER2026_ZIP_PW")
        z.setpassword(a.zip_pw.encode())
    byname = {os.path.splitext(os.path.basename(n))[0]: n for n in z.namelist() if n.lower().endswith((".mp4", ".avi", ".mkv"))}
    out_dir = os.path.join(a.work, "mer26human"); os.makedirs(out_dir, exist_ok=True); items = []
    for fn, ds in (("ovmerdplus-emotion.csv", "OV-MERD+"), ("mer2025ov1000-emotion.csv", "MER2025-FG")):
        if ds not in a.sets.split(","):
            continue
        R = list(csv.DictReader(open(os.path.join(a.affectgpt, "AffectGPT-RL/dataset", fn), encoding="utf-8"))); k = 0
        for r in R:
            n = r["name"]
            if n not in byname:
                continue
            p = os.path.join(out_dir, os.path.basename(byname[n]))
            if not os.path.exists(p):
                with open(p, "wb") as f:
                    f.write(z.read(byname[n]))
            items.append(dict(key=f"{ds}:{n}", ds=ds, video=p, sub=sub.get(n, ""), gt=[x.strip() for x in r["openset"].split(",") if x.strip()])); k += 1
        print(ds, "clips with video:", k, "/", len(R), "| with English subtitle:", sum(1 for it in items if it["ds"] == ds and it["sub"]), flush=True)
    return items


def emos_meld(a):
    frames = [pd.read_excel(os.path.join(a.emos, f"datasets/data-final/English/goemotion/goemotion_soft_labels_{s}.xlsx")) for s in ("first3000", "last2000")]
    d = pd.concat(frames, ignore_index=True); cols = [c for c in d.columns if c.startswith("goemotion_prob_")]
    rows = []
    for _, r in d.iterrows():
        gt = [c.replace("goemotion_prob_", "") for c in cols if float(r[c] or 0) >= 0.5 and c.split("_")[-1] not in ("neutral", "unsure")]
        if gt:
            rows.append(dict(dia=int(r["Dialogue_ID"]), utt=int(r["Utterance_ID"]), text=str(r["Utterance"]), gt=gt))
    print("EmoS-MELD clips with a non-empty GT:", len(rows), "/", len(d), flush=True)
    out_dir = os.path.join(a.work, "meld"); os.makedirs(out_dir, exist_ok=True)
    want = [f"train_splits/dia{r['dia']}_utt{r['utt']}.mp4" for r in rows]
    missing = [w for w in want if not os.path.exists(os.path.join(out_dir, w))]
    if missing:
        lst = os.path.join(a.work, "meld_list.txt"); open(lst, "w").write("\n".join(missing) + "\n")
        src = f"curl -sL '{a.meld_tar}'" if a.meld_tar.startswith("http") else f"cat '{a.meld_tar}'"
        print("extracting", len(missing), "MELD clips (streams the 10.9 GB archive once)...", flush=True)
        subprocess.run(f"{src} | tar -xzO MELD.Raw/train.tar.gz | tar -xz -C '{out_dir}' -T '{lst}'", shell=True, check=False)
    items = [dict(key=f"meld:{r['dia']}_{r['utt']}", ds="EmoS-MELD", video=os.path.join(out_dir, w), sub=r["text"], gt=r["gt"])
             for r, w in zip(rows, want) if os.path.exists(os.path.join(out_dir, w))]
    print("EmoS-MELD clips with video:", len(items), "/", len(rows), flush=True)
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("affectgpt", "emos", "mer26zip", "meld_tar", "work", "out"):
        ap.add_argument("--" + k)
    ap.add_argument("--zip_pw", default=os.environ.get("MER2026_ZIP_PW", ""))
    ap.add_argument("--sets", default="OV-MERD+,MER2025-FG,EmoS-MELD",
                    help="tuning on Kaggle uses MER2025-FG only, so OV-MERD+ media and labels are never touched")
    a = ap.parse_args(); os.makedirs(a.work, exist_ok=True)
    items = mer_sets(a) + (emos_meld(a) if "EmoS-MELD" in a.sets.split(",") else [])
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("items:", len(items), {ds: sum(1 for it in items if it["ds"] == ds) for ds in ("OV-MERD+", "MER2025-FG", "EmoS-MELD")})
