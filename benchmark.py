"""
benchmarking NSGA-II and VEGA on CFLP.
"""

import csv
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

from base import Problem
from nsga2 import NSGA2
from vega import VEGA

DATA_DIR = Path(__file__).parent / "data"
PLOT_DIR = Path(__file__).parent / "plots"
RESULTS_DIR = Path(__file__).parent / "results"

INSTANCE_CATEGORIES = {
    "cap61": "small",
    "cap62": "small",
    "cap101": "medium",
    "cap102": "medium",
    "cap121": "large",
    "cap122": "large",
}

ALGORITHMS = {
    "nsga2": NSGA2,
    "vega": VEGA,
}

INSTANCES = [
    "cap61",
    "cap62",
    "cap101",
    "cap102",
    "cap121",
    "cap122",
]

PLOT_INSTANCES = ["cap61", "cap62", "cap101", "cap102", "cap121", "cap122"]

BASE_SEED = 42
HV_REF = (1.1, 1.1)

# Run on available CPU cores - 1 or 1 if none are detected
cores = os.cpu_count()
NUM_WORKERS = cores - 1 if cores is not None and cores > 1 else 1

NUM_RUNS = 10
GENERATIONS = 200

CONFIGS = {
    "baseline": {
        "pop_size": 100,
        "generations": GENERATIONS,
        "crossover_rate": 0.8,
        "mutation_rate": 0.05,
    },
    "high_pop": {
        "pop_size": 200,
        "generations": GENERATIONS,
        "crossover_rate": 0.8,
        "mutation_rate": 0.05,
    },
    "low_crossover": {
        "pop_size": 100,
        "generations": GENERATIONS,
        "crossover_rate": 0.4,
        "mutation_rate": 0.05,
    },
    "high_mutation": {
        "pop_size": 100,
        "generations": GENERATIONS,
        "crossover_rate": 0.8,
        "mutation_rate": 0.2,
    },
}


def instance_path(name: str) -> Path:
    return DATA_DIR / f"{name}.txt"


def extract_objectives(result):
    """Translate either algorithm's return value into (f1, f2) points."""
    points = []
    for ind in result:
        if hasattr(ind, "opening_cost"):
            if ind.rank == 0:
                points.append((ind.opening_cost, ind.customer_cost))
        else:
            points.append((float(ind[1]), float(ind[2])))
    return points


def nondominated_points(points):
    """Keep unique non-dominated (f1, f2) pairs (minimization)."""
    unique = sorted(set(points))
    front = []
    for i, (f1, f2) in enumerate(unique):
        dominated = False
        for j, (g1, g2) in enumerate(unique):
            if i == j:
                continue
            if g1 <= f1 and g2 <= f2 and (g1 < f1 or g2 < f2):
                dominated = True
                break
        if not dominated:
            front.append((f1, f2))
    return front


def run_once(algo_name: str, problem: Problem, params: dict, seed: int) -> dict:
    random.seed(seed)

    algorithm = ALGORITHMS[algo_name](problem, **params)

    start = time.perf_counter()
    raw_result = algorithm.run()
    elapsed = time.perf_counter() - start

    front = nondominated_points(extract_objectives(raw_result))
    return {
        "front": front,
        "n_nondominated": len(front),
        "time_seconds": elapsed,
    }


# -----AI Generated start-----
def _worker_task(args):
    """Worker function executed inside parallel process pool."""
    instance_name, config_name, algo_name, params, run_idx, seed = args
    problem = Problem.from_file(instance_path(instance_name))
    run_result = run_once(algo_name, problem, params, seed)
    return {
        "instance": instance_name,
        "config": config_name,
        "algorithm": algo_name,
        "run_idx": run_idx,
        "seed": seed,
        **run_result,
    }
# -----AI Generated end-----


def shared_bounds(fronts):
    """Shared f1/f2 min-max across every front in the comparison."""
    points = [p for front in fronts for p in front]
    if not points:
        raise ValueError("No points to build bounds from")
    f1s = [f1 for f1, _ in points]
    f2s = [f2 for _, f2 in points]
    return min(f1s), max(f1s), min(f2s), max(f2s)


def normalize(front, bounds):
    """Maps each objective to ~[0, 1] with the shared bounds."""
    f1_min, f1_max, f2_min, f2_max = bounds
    f1_span = (f1_max - f1_min) or 1.0
    f2_span = (f2_max - f2_min) or 1.0
    return [
        ((f1 - f1_min) / f1_span, (f2 - f2_min) / f2_span)
        for f1, f2 in front
    ]


def hypervolume(front, bounds):
    """Normalize with shared bounds, then 2D hypervolume vs HV_REF."""
    ref_f1, ref_f2 = HV_REF
    points = sorted(
        (f1, f2)
        for f1, f2 in normalize(front, bounds)
        if f1 < ref_f1 and f2 < ref_f2
    )
    if not points:
        return 0.0

    hv = 0.0
    for i, (f1, f2) in enumerate(points):
        next_f1 = points[i + 1][0] if i + 1 < len(points) else ref_f1
        width = next_f1 - f1
        height = ref_f2 - f2
        if width > 0 and height > 0:
            hv += width * height
    return hv


def summarize(values):
    """mean / std / best / worst for a list of numbers."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "best": 0.0, "worst": 0.0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "best": max(values),
        "worst": min(values),
    }


def run_cell(problem, instance_name, config_name, params, num_runs):
    """
    Run both algorithms for num_runs seeds on one instance/config.
    Returns raw runs (HV is calculated later with instance-wide bounds).
    """
    runs = {algo: [] for algo in ALGORITHMS}
    for algo_name in ALGORITHMS:
        for run_id in range(num_runs):
            seed = (BASE_SEED + run_id) if BASE_SEED is not None else None
            result = run_once(algo_name, problem, params, seed)
            result["seed"] = seed
            runs[algo_name].append(result)
    return {
        "instance": instance_name,
        "config": config_name,
        "runs": runs,
    }


def print_cell_summary(cell):
    print(f"\n=== {cell['instance']} / {cell['config']} ===")
    for algo_name, algo_runs in cell["runs"].items():
        hv_stats = summarize([r["hv"] for r in algo_runs])
        nd_mean = statistics.mean([r["n_nondominated"] for r in algo_runs])
        time_mean = statistics.mean([r["time_seconds"] for r in algo_runs])
        print(f"{algo_name}:")
        print(
            f"  HV     mean={hv_stats['mean']:.6f}  std={hv_stats['std']:.6f}  "
            f"best={hv_stats['best']:.6f}  worst={hv_stats['worst']:.6f}"
        )
        print(f"  ND     mean={nd_mean:.2f}")
        print(f"  time   mean={time_mean:.3f}s")


def plot_pareto(cell, out_dir=PLOT_DIR):
    """Scatter both algorithms averaged fronts on the same axes."""
    out_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots()

    markers = {"nsga2": "o", "vega": "x"}
    for algo_name, algo_runs in cell["runs"].items():
        sorted_runs = sorted(algo_runs, key=lambda r: r["hv"])
        median_run = sorted_runs[len(sorted_runs) // 2]
        front = median_run["front"]
        if not front:
            continue
        xs = [p[0] for p in front]
        ys = [p[1] for p in front]
        ax.scatter(xs, ys, marker=markers.get(algo_name, "o"), label=algo_name)

    ax.set_xlabel("Facility opening cost (f1)")
    ax.set_ylabel("Customer allocation cost (f2)")
    ax.set_title(f"Pareto front - {cell['instance']} ({cell['config']})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = out_dir / f"pareto_{cell['instance']}_{cell['config']}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {path}")


# -----AI Generated start-----
def save_csv_results(cells, out_csv="results/metrics.csv", raw_csv="results/raw_runs.csv"):
    """Export summary metrics and raw run data to CSV files."""
    summary_rows = []
    raw_rows = []

    for cell in cells:
        instance_name = cell["instance"]
        config_name = cell["config"]
        category = INSTANCE_CATEGORIES.get(instance_name, "unknown")
        params = CONFIGS.get(config_name, {})

        for algo_name, algo_runs in cell["runs"].items():
            for result in algo_runs:
                raw_rows.append({
                    "category": category,
                    "instance": instance_name,
                    "config": config_name,
                    "algorithm": algo_name,
                    "run_idx": result["run_idx"],
                    "seed": result.get("seed"),
                    "hv": round(result["hv"], 6),
                    "n_nondominated": result["n_nondominated"],
                    "time_seconds": round(result["time_seconds"], 4),
                })

            hv_stats = summarize([r["hv"] for r in algo_runs])
            nd_stats = summarize([r["n_nondominated"] for r in algo_runs])
            time_stats = summarize([r["time_seconds"] for r in algo_runs])

            summary_rows.append({
                "category": category,
                "instance": instance_name,
                "config": config_name,
                "algorithm": algo_name,
                "hv_mean": round(hv_stats["mean"], 6),
                "hv_std": round(hv_stats["std"], 6),
                "hv_best": round(hv_stats["best"], 6),
                "hv_worst": round(hv_stats["worst"], 6),
                "nd_mean": round(nd_stats["mean"], 2),
                "nd_std": round(nd_stats["std"], 2),
                "time_mean": round(time_stats["mean"], 4),
                "time_std": round(time_stats["std"], 4),
                **{f"p_{k}": v for k, v in params.items()},
            })

    if out_csv and summary_rows:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Wrote summary metrics to {out_path}")

    if raw_csv and raw_rows:
        raw_path = Path(raw_csv)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=raw_rows[0].keys())
            w.writeheader()
            w.writerows(raw_rows)
        print(f"Wrote raw runs to {raw_path}")
# -----AI Generated end-----


def run_experiment(
    out_csv="results/metrics.csv",
    raw_csv="results/raw_runs.csv",
    num_workers=NUM_WORKERS,
):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # -----AI Generated start-----
    tasks = []
    for instance_name in INSTANCES:
        for config_name, params in CONFIGS.items():
            for algo_name in ALGORITHMS:
                for run_idx in range(NUM_RUNS):
                    seed = (BASE_SEED + run_idx) if BASE_SEED is not None else None
                    tasks.append((instance_name, config_name, algo_name, params, run_idx, seed))

    total_tasks = len(tasks)
    print(f"Starting {total_tasks} runs using {num_workers} parallel workers...")
    t_start = time.perf_counter()

    results_by_config = {}
    completed_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker_task, t) for t in tasks]
        for future in as_completed(futures):
            res = future.result()
            key = (res["instance"], res["config"], res["algorithm"])
            results_by_config.setdefault(key, []).append(res)
            completed_count += 1
            print(
                f"[{completed_count:3d}/{total_tasks}] "
                f"{res['instance']} / {res['config']} / {res['algorithm']} "
                f"(Run {res['run_idx']+1:2d}/{NUM_RUNS}) -> "
                f"ND: {res['n_nondominated']}, Time: {res['time_seconds']:.2f}s"
            )

    elapsed_all = time.perf_counter() - t_start
    print(f"All {total_tasks} runs finished in {elapsed_all:.2f}s across {num_workers} cores.\n")
    # -----AI Generated end-----

    cells = []
    for instance_name in INSTANCES:
        instance_cells = []
        # 1. Assemble cells for this instance
        for config_name in CONFIGS:
            cell_runs = {}
            for algo_name in ALGORITHMS:
                runs = results_by_config.get((instance_name, config_name, algo_name), [])
                runs.sort(key=lambda r: r["run_idx"])  # keep deterministic ordering
                cell_runs[algo_name] = runs

            cell = {
                "instance": instance_name,
                "config": config_name,
                "runs": cell_runs,
            }
            instance_cells.append(cell)

        # 2. Compute one shared bounding box across all configs & algorithms for this instance
        all_instance_fronts = [
            r["front"]
            for cell in instance_cells
            for algo_runs in cell["runs"].values()
            for r in algo_runs
            if r["front"]
        ]
        inst_bounds = shared_bounds(all_instance_fronts)

        # 3. Calculate hypervolume using the shared bounds, then summarize and plot
        for cell in instance_cells:
            cell["bounds"] = inst_bounds
            for algo_runs in cell["runs"].values():
                for result in algo_runs:
                    result["hv"] = hypervolume(result["front"], inst_bounds)
            print_cell_summary(cell)
            if instance_name in PLOT_INSTANCES:
                plot_pareto(cell)
            cells.append(cell)

    # -----AI Generated start-----
    save_csv_results(cells, out_csv=out_csv, raw_csv=raw_csv)
    # -----AI Generated end-----

    return cells


def main():
    print(
        f"Benchmark: {len(INSTANCES)} instances, {len(CONFIGS)} configs, "
        f"{NUM_RUNS} runs, {len(ALGORITHMS)} algorithms\n"
    )
    run_experiment()
    print("\nDone.")




if __name__ == "__main__":
    main()

