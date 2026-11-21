"""Gap harvesting, stage 1 of protocol v2 (CLAUDE.md §4.6).

Step A (list):  python -I gap_harvest.py list <out.tsv>
  arXiv API: affect-related queries, submitted 2025-01-01..today, keeps papers whose arXiv comment names an accepted venue.
Step B (fetch): python -I gap_harvest.py fetch <in.tsv> <pdf_dir> [max]
Step C (extract): python -I gap_harvest.py extract <in.tsv> <pdf_dir> <out.md>
  pdftotext, then keeps (1) Limitation / Future work / Conclusion / Discussion sections, (2) sentences with gap cue phrases.
Downloaded PDFs are untrusted data: only parsed as text, never executed.
"""
import html, os, re, subprocess, sys, time, urllib.parse, urllib.request

QUERIES = [
    'abs:"multimodal emotion recognition"', 'abs:"multimodal sentiment analysis"', 'abs:"emotion recognition in conversation"',
    'abs:"speech emotion recognition"', 'abs:"facial expression recognition"', 'abs:"audio-visual emotion"',
    'abs:"affective computing"', 'abs:"emotion understanding" AND abs:"large language model"', 'abs:"multimodal emotion"',
    'abs:"emotion recognition" AND abs:multimodal',
]
VENUE = re.compile(r"(?i)(accept|to appear|published|camera|proceedings).{0,80}?(ACM\s*MM|ACM Multimedia|MM\s*'?\s*2[56]|AAAI|ICASSP|Interspeech|"
                   r"\bACL\b|EMNLP|NAACL|COLING|CVPR|ICCV|ECCV|NeurIPS|ICLR|ICML|IJCAI|TAFFC|Transactions on Affective|TMM|"
                   r"Transactions on Multimedia|Information Fusion|ICME|ACII|ICMI|WWW|KDD|TASLP|TPAMI|TIP|IJCV|Pattern Recognition)"
                   r"|(ACM\s*MM|AAAI|ICASSP|Interspeech|EMNLP|NAACL|CVPR|ICCV|NeurIPS|ICLR|ICME|IJCAI|ACL)\s*'?\s*20?2[56]")
CUES = re.compile(r"(?i)\b(limitation|remains? (challenging|open|unclear|unexplored|underexplored)|open (problem|question|challenge)|"
                  r"fail(s|ed)? to|struggle|future work|we leave|left for future|under-?explored|has not been|have not been|neglect|overlook|"
                  r"bottleneck|degrad(e|es|ation)|does not generali[sz]e|poor generali[sz]ation|lack of|insufficient)\b")
HEAD = re.compile(r"(?im)^\s*(?:\d+\.?\s*|[IVX]+\.\s*)?(limitations?(?: and future work)?|future work|conclusions?(?: and future work)?|"
                  r"discussion|broader impacts?|limitation and future work)\s*$")
END = re.compile(r"(?im)^\s*(references|acknowledg(e)?ments?|appendix|bibliography)\s*$")


def api(q, n=200):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": f"({q}) AND submittedDate:[202501010000 TO 202611220000]",
                                                                          "start": 0, "max_results": n, "sortBy": "submittedDate", "sortOrder": "descending"})
    for a in range(6):
        try:
            return urllib.request.urlopen(url, timeout=90).read().decode()
        except Exception:  # noqa: BLE001
            time.sleep(min(120, 15 * 2 ** a))
    return ""


def cmd_list(out):
    seen = {}
    for q in QUERIES:
        x = api(q); ents = x.split("<entry>")[1:]
        print(f"{q}: {len(ents)} entries", flush=True)
        for e in ents:
            i = re.search(r"<id>http://arxiv.org/abs/(.*?)</id>", e).group(1).split("v")[0]
            c = re.search(r"<arxiv:comment[^>]*>(.*?)</arxiv:comment>", e, re.S); c = html.unescape(re.sub(r"\s+", " ", c.group(1))) if c else ""
            t = html.unescape(re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", e, re.S).group(1)))
            d = re.search(r"<published>(.*?)</published>", e).group(1)[:10]
            if VENUE.search(c) and i not in seen:
                seen[i] = (d, t, c)
        time.sleep(4)
    with open(out, "w") as f:
        for i, (d, t, c) in sorted(seen.items(), key=lambda kv: kv[1][0], reverse=True):
            f.write(f"{i}\t{d}\t{t}\t{c}\n")
    print(f"kept {len(seen)} accepted-venue papers -> {out}")


def cmd_fetch(tsv, pdir, mx=10 ** 9):
    os.makedirs(pdir, exist_ok=True); n = 0
    for line in open(tsv):
        i = line.split("\t")[0]; p = os.path.join(pdir, i.replace("/", "_") + ".pdf")
        if os.path.exists(p) and os.path.getsize(p) > 10000:
            continue
        if n >= int(mx):
            break
        try:
            urllib.request.urlretrieve(f"https://arxiv.org/pdf/{i}", p); n += 1
        except Exception as e:  # noqa: BLE001
            print("fail", i, type(e).__name__)
        time.sleep(3)
    print("downloaded", n)


def sections(txt):
    out = []
    for m in HEAD.finditer(txt):
        rest = txt[m.end():]; e = END.search(rest); nxt = re.search(r"(?m)^\s*(?:\d+\.?\s+|[IVX]+\.\s+)[A-Z][A-Za-z ]{3,40}$", rest)
        stop = min([x.start() for x in [e, nxt] if x] + [3500])
        out.append((m.group(1).strip(), re.sub(r"\s+", " ", rest[:stop]).strip()))
    return out


def cmd_extract(tsv, pdir, out):
    meta = {l.split("\t")[0]: l.rstrip("\n").split("\t") for l in open(tsv)}
    with open(out, "w") as f:
        for i, (_, d, t, c) in meta.items():
            p = os.path.join(pdir, i.replace("/", "_") + ".pdf")
            if not os.path.exists(p):
                continue
            txt = subprocess.run(["pdftotext", "-q", p, "-"], capture_output=True, text=True, timeout=120).stdout
            body = END.split(txt)[0] if END.search(txt) else txt
            f.write(f"\n\n### {i} | {d} | {t}\n[venue] {c}\n")
            for h, s in sections(txt):
                f.write(f"[section: {h}] {s[:2500]}\n")
            sents = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", body))
            cues = [s for s in sents if CUES.search(s) and 40 < len(s) < 400][:25]
            for s in cues:
                f.write(f"[cue] {s}\n")


if __name__ == "__main__":
    {"list": lambda: cmd_list(sys.argv[2]), "fetch": lambda: cmd_fetch(*sys.argv[2:]), "extract": lambda: cmd_extract(*sys.argv[2:])}[sys.argv[1]]()
