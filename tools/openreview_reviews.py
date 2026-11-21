"""Harvest reviewer weaknesses for affect-related submissions on OpenReview (source (c) of protocol v2, CLAUDE.md §4.6).
Usage: python -I openreview_reviews.py <out.md> [group ...]   (default groups: ICLR.cc/2026/Conference ICLR.cc/2025/Conference NeurIPS.cc/2025/Conference)
Writes, per forum: title, venue/decision string, and each official review's weaknesses field (trimmed). Untrusted text: stored only.
"""
import json, sys, time, urllib.parse, urllib.request

TERMS = ["emotion", "sentiment", "affective"]
API = "https://api2.openreview.net"


def get(path, **q):
    url = f"{API}{path}?{urllib.parse.urlencode(q)}"
    for a in range(5):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=60).read().decode())
        except Exception:  # noqa: BLE001
            time.sleep(5 * (a + 1))
    return {}


def val(c, k):
    v = c.get(k, {}); return v.get("value", "") if isinstance(v, dict) else v


def main():
    out = sys.argv[1]; groups = sys.argv[2:] or ["ICLR.cc/2026/Conference", "ICLR.cc/2025/Conference", "NeurIPS.cc/2025/Conference"]
    forums = {}
    for g in groups:
        for t in TERMS:
            r = get("/notes/search", term=t, group=g, limit=100, source="forum")
            for n in r.get("notes", []):
                forums[n["forum"]] = (val(n["content"], "title"), val(n["content"], "venue"))
            time.sleep(1.5)
    print("forums", len(forums), flush=True)
    with open(out, "w", buffering=1) as f:
        f.write(f"# OpenReview weaknesses harvest ({len(forums)} forums; groups {groups})\n")
        for fid, (title, v) in forums.items():
            r = get("/notes", forum=fid, limit=200); time.sleep(1.2)
            revs = [n for n in r.get("notes", []) if any(i.endswith("Official_Review") for i in n.get("invitations", []))]
            f.write(f"\n### {title} | {v} | https://openreview.net/forum?id={fid} | reviews {len(revs)}\n")
            for n in revs:
                c = n.get("content", {}); w = val(c, "weaknesses") or val(c, "strengths_and_weaknesses") or val(c, "review")
                rating = val(c, "rating")
                f.write(f"- [rating {rating}] {' '.join(str(w).split())[:1500]}\n")
    print("forums", len(forums), "->", out)


if __name__ == "__main__":
    main()
