"""Translate-test text for P0: MOSI (en) -> zh and CH-SIMS (zh) -> en with opus-mt; embed with multilingual-e5-base.
Usage: python -I p0_translate.py mosi_label.csv sims_label_h.csv out.npz
"""
import sys, os
import numpy as np, pandas as pd, torch
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModel

torch.set_num_threads(int(os.environ.get("NT", "1")))
M = pd.read_csv(sys.argv[1], dtype={"clip_id": str})
S = pd.read_csv(sys.argv[2], dtype={"clip_id": str})


def translate(texts, name):
    tok = MarianTokenizer.from_pretrained(name); mt = MarianMTModel.from_pretrained(name).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            b = tok([str(t) for t in texts[i:i + 16]], return_tensors="pt", padding=True, truncation=True, max_length=128)
            out += tok.batch_decode(mt.generate(**b, max_new_tokens=128, num_beams=2), skip_special_tokens=True)
            if i % 400 == 0:
                print(name, i, flush=True)
    return out


m_zh = translate(M.text.str.lower().tolist(), "Helsinki-NLP/opus-mt-en-zh")
s_en = translate(S.text.tolist(), "Helsinki-NLP/opus-mt-zh-en")
tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base"); tm = AutoModel.from_pretrained("intfloat/multilingual-e5-base").eval()


@torch.no_grad()
def emb(texts):
    out = []
    for i in range(0, len(texts), 32):
        b = tok(["query: " + str(t) for t in texts[i:i + 32]], padding=True, truncation=True, max_length=128, return_tensors="pt")
        h = tm(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).numpy())
    return np.concatenate(out)


np.savez_compressed(sys.argv[3], mosi_clip=(M.video_id + "_" + M.clip_id).values, sims_clip=(S.video_id + "_" + S.clip_id).values,
                    mosi_zh=np.array(m_zh, dtype=object), sims_en=np.array(s_en, dtype=object), mosi_zh_emb=emb(m_zh), sims_en_emb=emb(s_en))
print("examples:", list(zip(M.text[:3], m_zh[:3])), list(zip(S.text[:3], s_en[:3])))
