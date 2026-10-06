# Multi-Objective Capacitated Facility Location Problem (CFLP): NSGA-II & VEGA

## Setup

For uv:
```bash
uv sync
```

For pip:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib
```

## Run

Run the full benchmark suite across instances and parameter sets:

```bash
uv run benchmark.py
```
OR
```bash
python benchmark.py
```

### Parameters (`benchmark.py`)
Configured at the top of `benchmark.py`:
- `NUM_WORKERS`: Number of parallel CPU processes (default: `cores - 1`)
- `NUM_RUNS`: Number of repeated runs per instance / config (default: `10`)
- `BASE_SEED`: Base random seed for reproducible runs (default: `None`)
- `INSTANCES`: Benchmark files to test from `data/` (`cap61`, `cap62`, `cap101`, `cap102`, `cap121`, `cap122`)
- `CONFIGS`: Parameter configurations to evaluate (`baseline`, `high_pop`, `low_crossover`, `high_mutation`)

### Outputs
- `results/metrics.csv`: Summary metrics (HV mean/std/best/worst, ND mean/std, time mean/std, configuration parameters)
- `results/raw_runs.csv`: Per-run detailed records (instance, config, algorithm, run index, seed, HV, ND, runtime)
- `plots/pareto_*.png`: Final Pareto-front scatter plots comparing NSGA-II and VEGA on the same axes

---

## Single Run (Testing Only)

Run a single instance directly via `nsga2.py` or `vega.py`:

```bash
uv run nsga2.py
# or: python nsga2.py
```

OR

```bash
uv run vega.py
# or: python vega.py
```

### Parameters (`nsga2.py` / `vega.py`)
Configured in the execution block at the bottom of each file:

**NSGA-II (`nsga2.py`)**:
```python
path = "./data/cap61.txt"
instance = Problem.from_file(path)
algorithm = NSGA2(
    instance,
    generations=1000,
    mutation_rate=0.05,
    crossover_rate=0.8,
    pop_size=200,
)
```

**VEGA (`vega.py`)**:
```python
path = "./data/cap61.txt"
instance = Problem.from_file(path)
algorithm = VEGA(
    instance,
    generations=100,
    mutation_rate=0.05,
    crossover_rate=0.2,
    pop_size=200,
)
```

---

## AI Usage Declaration
AI was primarily utilized for documentation and benchmark support.
Components that were developed or assisted using AI are detailed below.
They are also clearly marked in the code.
- `benchmark.py`:
    - Multithreading / parallel process pool execution (`_worker_task`, task pool runner)
    - CSV export functionality (`save_csv_results`)
    - Plotting and Pareto front visualization
