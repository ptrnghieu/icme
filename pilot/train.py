"""Train one model of the pilot grid and dump per-sample outputs.

Methods
  joint        concat fusion, single cross-entropy (OGM-GE "Normal")
  ogm_ge       OGM-GE gradient modulation, re-implemented line by line from
               GeWu-Lab/OGM-GE_CVPR2022 main.py (coefficients from the softmax score ratio of the
               decomposed logits; Gaussian enhancement with std of the gradient; applied to 4-D conv
               weights of each encoder between --mod_start and --mod_end epochs)
  audio_only / visual_only   unimodal references

Hyper-parameters default to OGM-GE's CREMA-D setting: SGD(lr 1e-3, momentum 0.9, wd 1e-4), batch 64,
100 epochs, StepLR(step 70, gamma 0.1), 1 frame, modulation epochs 0..50.

Model selection: best validation accuracy (protocol fix). Test accuracy is logged every epoch so the
OGM-GE-style number (best test epoch, i.e. selection on test) can be reported alongside.

Outputs in <out>/<name>/: config.json, log.jsonl, metrics.json, preds_val.csv, preds_test.csv
(optionally best.pt with --save_ckpt).
"""
import argparse
import copy
import csv
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import CLASSES, set_seed
from data import CremadAV
from models import AVNet


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True, help="standard | actor0 .. actor4")
    ap.add_argument("--method", required=True, choices=["joint", "ogm_ge", "audio_only", "visual_only"])
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_step", type=int, default=70)
    ap.add_argument("--lr_gamma", type=float, default=0.1)
    ap.add_argument("--mod_start", type=int, default=0)
    ap.add_argument("--mod_end", type=int, default=50)
    ap.add_argument("--n_frames", type=int, default=1)
    ap.add_argument("--eval_frame", default="center", choices=["center", "first"],
                    help="frame(s) used for validation/test; the other choice is also reported at the end")
    ap.add_argument("--fps", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (T4 supports FP16, not BF16)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default="")
    ap.add_argument("--save_ckpt", action="store_true")
    ap.add_argument("--max_batches", type=int, default=0, help="debug: limit batches per epoch")
    return ap.parse_args()


def run_name(a):
    m = a.method + (f"_a{a.alpha:g}" if a.method == "ogm_ge" else "")
    return a.name or f"{a.split}_{m}_s{a.seed}_f{a.n_frames}"


def ogm_ge_modulate(model, out_a, out_v, label, alpha, enhance=True):
    """Exact transcription of OGM-GE main.py (train_epoch) for concat fusion.

    Mixed-precision guard (not needed in the original fp32 code): when the fp16 GradScaler overflows, some
    gradients are inf, grad.std() is nan and normal_(0, nan) raises. The scaler skips such a step anyway, so
    we skip the modulation for that step and return applied=False. With finite gradients the behaviour is
    unchanged (see test_ogm_equivalence.py)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is not None and p.grad.dim() == 4 and name.startswith(("audio_net", "visual_net")):
                if not torch.isfinite(p.grad).all():
                    return float("nan"), float("nan"), float("nan"), False
        sm_v, sm_a = F.softmax(out_v.float(), 1), F.softmax(out_a.float(), 1)
        idx = torch.arange(label.size(0), device=label.device)
        score_v, score_a = sm_v[idx, label].sum(), sm_a[idx, label].sum()
        ratio_v = score_v / score_a
        ratio_a = 1 / ratio_v
        if ratio_v > 1:
            coeff_v = 1 - torch.tanh(alpha * F.relu(ratio_v))
            coeff_a = torch.tensor(1.0, device=label.device)
        else:
            coeff_a = 1 - torch.tanh(alpha * F.relu(ratio_a))
            coeff_v = torch.tensor(1.0, device=label.device)
        for name, p in model.named_parameters():
            if p.grad is None or p.grad.dim() != 4:
                continue
            if name.startswith("audio_net"):
                c = coeff_a
            elif name.startswith("visual_net"):
                c = coeff_v
            else:
                continue
            if enhance:
                p.grad = p.grad * c + torch.zeros_like(p.grad).normal_(0, p.grad.std().item() + 1e-8)
            else:
                p.grad *= c
    return float(coeff_a), float(coeff_v), float(ratio_v), True


@torch.no_grad()
def evaluate(model, loader, device, amp, dump=False, clips=None):
    model.eval()
    n = c = ca = cv = 0
    rows = []
    for spec, imgs, y, idx in loader:
        spec, imgs, y = spec.to(device, non_blocking=True), imgs.to(device, non_blocking=True), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            a, v, out = model(spec, imgs)
        out = out.float()
        pred = out.argmax(1)
        n += y.numel()
        c += (pred == y).sum().item()
        rec = {"pred": pred, "p_true": F.softmax(out, 1).gather(1, y[:, None])[:, 0]}
        if model.mode == "av":
            a, v = a.float(), v.float()
            out_a, out_v = model.decompose(a, v)
            ca += (out_a.argmax(1) == y).sum().item()
            cv += (out_v.argmax(1) == y).sum().item()
            if dump:
                co = model.coalition_logits(a, v)
                lp = {k: F.log_softmax(t, 1).gather(1, y[:, None])[:, 0] for k, t in co.items()}
                hit = {k: (t.argmax(1) == y).float() for k, t in co.items()}
                rec.update(
                    pred_a=out_a.argmax(1), pred_v=out_v.argmax(1),
                    pa_true=F.softmax(out_a, 1).gather(1, y[:, None])[:, 0],
                    pv_true=F.softmax(out_v, 1).gather(1, y[:, None])[:, 0],
                    # 2-player Shapley values, value = log p(true class)
                    phi_a=0.5 * ((lp["a"] - lp["none"]) + (lp["av"] - lp["v"])),
                    phi_v=0.5 * ((lp["v"] - lp["none"]) + (lp["av"] - lp["a"])),
                    # 2-player Shapley values, value = correctness (0/1)
                    phi_a_acc=0.5 * ((hit["a"] - hit["none"]) + (hit["av"] - hit["v"])),
                    phi_v_acc=0.5 * ((hit["v"] - hit["none"]) + (hit["av"] - hit["a"])),
                )
                # label-free reliance (does not use the true label): how much the full prediction changes when a
                # modality is removed. kl_drop_v = KL(p_av || p_audio-only), kl_drop_a = KL(p_av || p_visual-only).
                # rel_lf = kl_drop_v - kl_drop_a = Shapley(visual) - Shapley(audio) with value v(S) = -KL(p_av || p_S).
                lav = F.log_softmax(co["av"], 1)
                kl = lambda s_: (lav.exp() * (lav - F.log_softmax(co[s_], 1))).sum(1)  # noqa: E731
                rec.update(kl_drop_v=kl("a"), kl_drop_a=kl("v"))
                rec["rel_lf"] = rec["kl_drop_v"] - rec["kl_drop_a"]
        if dump:
            for j in range(y.numel()):
                r = {"clip": clips[idx[j].item()], "label": CLASSES[y[j].item()],
                     "correct": int(rec["pred"][j].item() == y[j].item())}
                for k, t in rec.items():
                    val = t[j].item()
                    r[k] = CLASSES[val] if k.startswith("pred") else round(float(val), 6)
                rows.append(r)
    res = {"acc": c / max(n, 1)}
    if model.mode == "av":
        res.update(acc_a=ca / max(n, 1), acc_v=cv / max(n, 1))
    return res, rows


def write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    a = get_args()
    set_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = a.amp and device.type == "cuda"
    name = run_name(a)
    od = os.path.join(a.out, name)
    os.makedirs(od, exist_ok=True)
    json.dump(vars(a) | {"run": name}, open(os.path.join(od, "config.json"), "w"), indent=1)

    sd = os.path.join(a.root, "splits")
    mk = lambda part, tr, ef=a.eval_frame: CremadAV(a.root, os.path.join(sd, f"{a.split}_{part}.csv"), tr,  # noqa: E731
                                                     a.n_frames, a.fps, eval_frame=ef)
    ds = {"train": mk("train", True), "val": mk("val", False), "test": mk("test", False)}
    g = torch.Generator()
    g.manual_seed(a.seed)
    dl = {k: DataLoader(v, batch_size=a.bs, shuffle=(k == "train"), num_workers=a.workers, pin_memory=True,
                        drop_last=False, generator=g if k == "train" else None, persistent_workers=a.workers > 0)
          for k, v in ds.items()}

    mode = {"joint": "av", "ogm_ge": "av", "audio_only": "audio", "visual_only": "visual"}[a.method]
    model = AVNet(len(CLASSES), mode).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, a.lr_step, a.lr_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ce = nn.CrossEntropyLoss()

    best_val, best_state, best_ep, best_test_any, best_test_ep = -1.0, None, -1, -1.0, -1
    log = open(os.path.join(od, "log.jsonl"), "w")
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        te = time.time()
        tot, nb, coefs = 0.0, 0, []
        for b, (spec, imgs, y, _) in enumerate(dl["train"]):
            if a.max_batches and b >= a.max_batches:
                break
            spec, imgs, y = spec.to(device, non_blocking=True), imgs.to(device, non_blocking=True), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                fa, fv, out = model(spec, imgs)
                loss = ce(out, y)
            scaler.scale(loss).backward()
            if a.method == "ogm_ge":
                scaler.unscale_(opt)
                out_a, out_v = model.decompose(fa.detach().float(), fv.detach().float())
                if a.mod_start <= ep <= a.mod_end:
                    coefs.append(ogm_ge_modulate(model, out_a, out_v, y, a.alpha))
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            nb += 1
        sch.step()
        rv, _ = evaluate(model, dl["val"], device, amp)
        rt, _ = evaluate(model, dl["test"], device, amp)
        if rv["acc"] > best_val:
            best_val, best_ep, best_state = rv["acc"], ep, copy.deepcopy(model.state_dict())
        if rt["acc"] > best_test_any:
            best_test_any, best_test_ep = rt["acc"], ep
        rec = {"epoch": ep, "loss": tot / max(nb, 1), "val": rv, "test": rt, "sec": round(time.time() - te, 1)}
        if coefs:
            ok = [c for c in coefs if c[3]]
            rec["ogm_steps_modulated"] = len(ok)
            rec["ogm_steps_skipped_overflow"] = len(coefs) - len(ok)
            if ok:
                rec["mean_coeff_a"] = sum(c[0] for c in ok) / len(ok)
                rec["mean_coeff_v"] = sum(c[1] for c in ok) / len(ok)
                rec["mean_ratio_v"] = sum(c[2] for c in ok) / len(ok)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[{name}] ep {ep:3d} loss {rec['loss']:.3f} val {rv['acc']:.4f} test {rt['acc']:.4f} ({rec['sec']}s)",
              flush=True)

    model.load_state_dict(best_state)
    rv, rows_v = evaluate(model, dl["val"], device, amp, dump=True, clips=ds["val"].clips())
    rt, rows_t = evaluate(model, dl["test"], device, amp, dump=True, clips=ds["test"].clips())
    write_rows(os.path.join(od, "preds_val.csv"), rows_v)
    write_rows(os.path.join(od, "preds_test.csv"), rows_t)
    other = None
    if mode != "audio":  # same checkpoint, test set evaluated with the other frame choice
        alt = "first" if a.eval_frame == "center" else "center"
        dl_alt = DataLoader(mk("test", False, alt), batch_size=a.bs, shuffle=False, num_workers=a.workers)
        other = {"eval_frame": alt, "test_acc": evaluate(model, dl_alt, device, amp)[0]["acc"]}
    if a.save_ckpt:
        torch.save(best_state, os.path.join(od, "best.pt"))
    metrics = {
        "run": name, "split": a.split, "method": a.method, "alpha": a.alpha, "seed": a.seed, "n_frames": a.n_frames,
        "n_train": len(ds["train"]), "n_val": len(ds["val"]), "n_test": len(ds["test"]),
        "best_val_epoch": best_ep, "val_acc": rv["acc"], "test_acc": rt["acc"],
        "test_acc_a_decomp": rt.get("acc_a"), "test_acc_v_decomp": rt.get("acc_v"),
        "test_acc_oracle_best_epoch": best_test_any, "oracle_epoch": best_test_ep,
        "minutes": round((time.time() - t0) / 60, 2),
        "eval_frame": a.eval_frame, "test_acc_other_eval_frame": other,
        "dump_version": 2,  # 2 = label-free reliance columns + centre eval frame (2026-11-20)
    }
    json.dump(metrics, open(os.path.join(od, "metrics.json"), "w"), indent=1)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
