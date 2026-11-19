"""Run a job list on several GPUs (resumable: finished runs are skipped).

Usage: python launch.py --jobs jobs.txt --gpus 0,1 --per_gpu 1
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run_dir(argstr):
    import train  # noqa: F401  (reuse naming logic)

    sys.argv = ["train.py"] + shlex.split(argstr)
    a = train.get_args()
    return os.path.join(a.out, train.run_name(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="jobs.txt")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--per_gpu", type=int, default=1)
    a = ap.parse_args()
    jobs = [l.strip() for l in open(a.jobs) if l.strip()]
    todo = [j for j in jobs if not os.path.exists(os.path.join(run_dir(j), "metrics.json"))]
    print(f"{len(jobs)} jobs, {len(jobs)-len(todo)} already done, {len(todo)} to run")
    slots = [g for g in a.gpus.split(",") for _ in range(a.per_gpu)]
    running = {}
    while todo or running:
        for s in [s for s in range(len(slots)) if s not in running and todo]:
            j = todo.pop(0)
            d = run_dir(j)
            os.makedirs(d, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=slots[s])
            f = open(os.path.join(d, "stdout.log"), "w")
            p = subprocess.Popen([sys.executable, os.path.join(HERE, "train.py")] + shlex.split(j),
                                 env=env, stdout=f, stderr=subprocess.STDOUT, cwd=os.getcwd())
            running[s] = (p, j, f)
            print(f"[gpu {slots[s]}] start {d}")
        for s, (p, j, f) in list(running.items()):
            if p.poll() is not None:
                f.close()
                print(f"[gpu {slots[s]}] {'done' if p.returncode == 0 else 'FAILED rc=%d' % p.returncode}: {run_dir(j)}")
                del running[s]
        time.sleep(10)


if __name__ == "__main__":
    main()
