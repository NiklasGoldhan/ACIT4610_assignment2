# ACIT4610 Assignment 2

Multi-objective CFLP with NSGA-II and VEGA.

## Setup

Needs Python 3.13+ and matplotlib.

```bash
uv sync
uv run python benchmark.py
```

Or:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib
python benchmark.py
```

## Running the benchmark

benchmark.py runs both algorithms on the six OR-Library instances, with four shared parameter configs and 10 seeded runs each. It prints hypervolume, number of non-dominated solutions, and runtime, and saves Pareto plots in plots/.

A full run can take a few minutes. Example output is in `benchmark_results.log`.

AI note

The benchmark runner was written with AI support and reviewed by us.
