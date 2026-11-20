"""Run a job list on several GPUs (resumable: finished runs are skipped).

Kaggle-friendly options
  --shard i --n_shards N : run only jobs whose index % N == i (split the grid over several sessions)
  --time_budget_h H      : do not START a job if it would likely end after H hours from launch
                           (estimate = longest finished job so far, or --est_job_h before any job finished).
                           Running jobs are always allowed to finish.

Usage: python launch.py --jobs jobs.txt --gpus 0,1 --per_gpu 1 [--shard 0 --n_shards 3 --time_budget_h 8]
"""
import argparse
import os
import shlex
import subprocess
import sys
import tarfile
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
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--time_budget_h", type=float, default=0.0, help="0 = no budget")
    ap.add_argument("--est_job_h", type=float, default=1.5, help="job-duration estimate until one job finishes")
    ap.add_argument("--hard_stop_h", type=float, default=0.0,
                    help="kill still-running jobs after this many hours so the notebook can save its output "
                         "before the platform's session limit (0 = never). Killed jobs re-run from scratch on resume.")
    ap.add_argument("--snapshot_tar", default="", help="after every finished job, rewrite this tar with the runs "
                    "directory (so finished runs survive even if the session is cut)")
    a = ap.parse_args()
    jobs = [l.strip() for l in open(a.jobs) if l.strip()]
    mine = [j for i, j in enumerate(jobs) if i % a.n_shards == a.shard]
    todo = [j for j in mine if not os.path.exists(os.path.join(run_dir(j), "metrics.json"))]
    print(f"{len(jobs)} jobs in grid, {len(mine)} in shard {a.shard}/{a.n_shards}, "
          f"{len(mine)-len(todo)} already done, {len(todo)} to run", flush=True)
    slots = [g for g in a.gpus.split(",") for _ in range(a.per_gpu)]
    running, durations, skipped, killed = {}, [], [], []
    t0 = time.time()
    runs_dirs = sorted({os.path.dirname(run_dir(j)) for j in mine})

    def snapshot():
        if not a.snapshot_tar:
            return
        tmp = a.snapshot_tar + ".tmp"
        with tarfile.open(tmp, "w") as tf:
            for rd in runs_dirs:
                if os.path.isdir(rd):
                    tf.add(rd, arcname=os.path.basename(rd),
                           filter=lambda ti: None if ti.name.endswith(".pt") else ti)
        os.replace(tmp, a.snapshot_tar)
        print(f"snapshot -> {a.snapshot_tar}", flush=True)

    while todo or running:
        if a.hard_stop_h and (time.time() - t0) / 3600 > a.hard_stop_h and running:
            for s, (p, j, f, ts) in list(running.items()):
                p.terminate()
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    p.kill()
                f.close()
                killed.append(j)
                print(f"[gpu {slots[s]}] KILLED at hard stop ({a.hard_stop_h} h): {run_dir(j)}", flush=True)
                del running[s]
            skipped.extend(todo)
            todo = []
            break
        for s in [s for s in range(len(slots)) if s not in running and todo]:
            elapsed_h = (time.time() - t0) / 3600
            est = max(durations) if durations else a.est_job_h
            if a.time_budget_h and elapsed_h + est > a.time_budget_h:
                skipped.extend(todo)
                todo = []
                print(f"time budget: {elapsed_h:.2f} h elapsed + {est:.2f} h estimate > {a.time_budget_h} h; "
                      f"not starting {len(skipped)} remaining job(s)", flush=True)
                break
            j = todo.pop(0)
            d = run_dir(j)
            os.makedirs(d, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=slots[s])
            f = open(os.path.join(d, "stdout.log"), "w")
            p = subprocess.Popen([sys.executable, os.path.join(HERE, "train.py")] + shlex.split(j),
                                 env=env, stdout=f, stderr=subprocess.STDOUT, cwd=os.getcwd())
            running[s] = (p, j, f, time.time())
            print(f"[gpu {slots[s]}] start {d}", flush=True)
        for s, (p, j, f, ts) in list(running.items()):
            if p.poll() is not None:
                f.close()
                dur = (time.time() - ts) / 3600
                if p.returncode == 0:
                    durations.append(dur)
                print(f"[gpu {slots[s]}] {'done' if p.returncode == 0 else 'FAILED rc=%d' % p.returncode} "
                      f"({dur:.2f} h): {run_dir(j)}", flush=True)
                del running[s]
                snapshot()
        time.sleep(10)
    snapshot()
    left = [j for j in mine if not os.path.exists(os.path.join(run_dir(j), "metrics.json"))]
    print(f"finished in {(time.time()-t0)/3600:.2f} h; jobs not started because of the budget: {len(skipped)}; "
          f"killed at hard stop: {len(killed)}; shard jobs still missing: {len(left)} "
          f"(re-run this shard with its runs tar attached to finish them)", flush=True)


if __name__ == "__main__":
    main()
