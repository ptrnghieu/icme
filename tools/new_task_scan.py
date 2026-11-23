"""Systematic scan for NEW PROBLEM FORMULATIONS in affective computing at top venues, 2025-2026 (OpenAlex; DBLP is behind a
bot wall from this container, 2026-11-23). A paper is flagged when its abstract contains task-introduction language
("new task", "novel task", "we introduce ... task/benchmark/dataset/problem/setting", "first ... task/benchmark/dataset", "we formulate",
"we define ... task", "a new paradigm/perspective/setting").
Recall check: Hi-EF (emotion forecasting, AAAI 2026 per the user) must appear.
Usage: python -I new_task_scan.py <out.tsv>
"""
import json, re, sys, time, urllib.parse, urllib.request

OUT = sys.argv[1]
KW = ["emotion", "emotional", "affective", "affect", "sentiment", "empathy", "empathetic", "mood", "feeling", "emotions"]
VENUES = [("AAAI", r"\bAAAI\b|Association for the Advancement of Artificial Intelligence"), ("IJCAI", r"IJCAI|International Joint Conference on Artificial Intelligence"),
          ("ACM MM", r"ACM International Conference on Multimedia|^Proceedings of the \d+(nd|rd|th|st) ACM International Conference on Multimedia|ACM Multimedia"),
          ("CVPR", r"Computer Vision and Pattern Recognition|\bCVPR\b"), ("ICCV", r"International Conference on Computer Vision|\bICCV\b"),
          ("ECCV", r"European Conference on Computer Vision|\bECCV\b|Lecture notes in computer science"),
          ("ACL", r"Annual Meeting of the Association for Computational Linguistics|Findings of the Association for Computational Linguistics|\bACL\b"),
          ("EMNLP", r"Empirical Methods in Natural Language Processing|\bEMNLP\b"), ("NAACL", r"North American Chapter|\bNAACL\b"),
          ("NeurIPS", r"Neural Information Processing Systems|NeurIPS"), ("ICLR", r"Learning Representations|\bICLR\b"), ("ICML", r"International Conference on Machine Learning|\bICML\b"),
          ("Interspeech", r"Interspeech"), ("ICASSP", r"ICASSP|Acoustics, Speech,? and Signal Processing"),
          ("TAFFC", r"Transactions on Affective Computing"), ("TMM", r"Transactions on Multimedia"), ("InfFusion", r"Information Fusion"),
          ("ACII/ICMI", r"Affective Computing and Intelligent Interaction|International Conference on Multimodal Interaction"),
          ("CHI", r"CHI Conference on Human Factors"), ("WWW/KDD", r"Web Conference|Knowledge Discovery and Data Mining")]
NEW = re.compile(r"introduc\w* (the|a) (concept|notion|task|problem|setting)|this (new )?task|reformulat|rethink|re-?frame|formaliz|(new|novel|first)[\w\s\-]{0,40}\b(task|problem|setting|paradigm|benchmark|dataset|formulation)\b|we (formulate|define|introduce|propose)[\w\s\-]{0,60}\b(task|problem|setting|paradigm)\b|shift[\w\s]{0,30}from", re.I)


def get(url, tries=6):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-scan/0.1"}), timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            print("retry", k, type(e).__name__, str(e)[:80], flush=True); time.sleep(5 * 2 ** k)
    raise RuntimeError(url)


def abstract(inv):
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps); return " ".join(w for _, w in pos)


SRC = {'S4210191458': 'AAAI', 'S4306419999': 'IJCAI', 'S4363608757': 'ACM MM', 'S4210176548': 'CVPR', 'S4363607764': 'ICCV', 'S4393915740': 'ACL', 'S4363605144': 'ACL Findings', 'S7407096085': 'EMNLP Findings', 'S4363608991': 'EMNLP', 'S4363608774': 'NAACL', 'S4363605604': 'NAACL Findings', 'S4363606243': 'NeurIPS', 'S4306419637': 'ICLR', 'S4306419644': 'ICML', 'S4393918409': 'Interspeech', 'S4363607879': 'ICASSP', 'S104780363': 'TAFFC', 'S137030581': 'TMM', 'S7560371': 'Information Fusion'}
seen, rows = set(), []
for kw in KW:
  for ids in SRC:
    page = 1; cursor = True
    while cursor:
        q = {"filter": f"title.search:{kw},publication_year:2025-2026,locations.source.id:{ids}", "per-page": "200", "page": str(page),
             "select": "id,title,publication_year,abstract_inverted_index,doi,locations"}
        d = get("https://api.openalex.org/works?" + urllib.parse.urlencode(q)); cursor = page * 200 < d["meta"]["count"]; page += 1
        for w in d["results"]:
            if w["id"] in seen:
                continue
            seen.add(w["id"])
            sids = [(((l or {}).get("source") or {}).get("id") or "").split("/")[-1] for l in (w.get("locations") or [])]
            ven = next((SRC[x] for x in sids if x in SRC), "?")
            ab = abstract(w.get("abstract_inverted_index")); m = NEW.search(ab)
            rows.append((ven, w["publication_year"], w["title"] or "", "NEW" if m else "", (ab[max(0, m.start() - 80):m.end() + 160] if m else "")[:320].replace("\t", " "), w.get("doi") or w["id"], "abstract" if ab else "no-abstract", ab[:2500].replace("\t", " ")))
        if not d["results"]:
            break
        time.sleep(0.3)
  print(kw, "rows so far", len(rows), flush=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("venue\tyear\ttitle\tflag\tevidence\tdoi\tsources\tabstract\n")
    for r in sorted(rows):
        f.write("\t".join(str(x).replace("\n", " ") for x in r) + "\n")
print("done", len(rows), "venue rows;", sum(r[3] == "NEW" for r in rows), "flagged NEW")
hief = [r for r in rows if re.search(r"Hi-EF|emotion forecasting", r[2], re.I)]
print("recall check (Hi-EF / emotion forecasting):", hief[:5])
