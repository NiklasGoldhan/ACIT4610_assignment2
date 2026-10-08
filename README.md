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
- `GENERATIONS`: Number of generations per run (default: `200`)
- `BASE_SEED`: Base random seed for reproducible runs (default: `42`; set to `None` for non-deterministic runs)
- `INSTANCES`: Benchmark files to test from `data/` (`cap61`, `cap62`, `cap101`, `cap102`, `cap121`, `cap122`)
- `CONFIGS`: Parameter configurations to evaluate (`baseline`, `high_pop`, `low_crossover`, `high_mutation`)

### Outputs
- `results/metrics.csv`: Summary metrics (HV mean/std/best/worst, ND mean/std, time mean/std, configuration parameters)
- `results/raw_runs.csv`: Per-run detailed records (instance, config, algorithm, run index, seed, HV, ND, runtime)
- `plots/pareto_*.png`: Final Pareto-front scatter plots comparing NSGA-II and VEGA on the same axes

### Benchmark Workflow (Pseudocode)

```text
ALGORITHM: RunBenchmark

1. INITIALIZE CONFIGURATION:
   - Instances   ← [cap61, cap62, cap101, cap102, cap121, cap122]
   - Configs     ← [baseline, high_pop, low_crossover, high_mutation]
   - Algorithms  ← [NSGA-II, VEGA]
   - Runs        ← 10 per combination (seeds = 42, 43, ..., 51)

2. EXECUTE RUNS (IN PARALLEL):
   FOR each instance IN Instances:
       FOR each config IN Configs:
           FOR each algorithm IN Algorithms:
               FOR each run_idx FROM 0 TO 9:
                   - Set random seed ← (42 + run_idx)
                   - Run algorithm with config parameters
                   - Extract non-dominated Pareto front
                   - Record runtime and number of non-dominated solutions

3. EVALUATE & PLOT PER INSTANCE:
   FOR each instance IN Instances:
       - Find shared min/max objective bounds across ALL runs for this instance
       - FOR each run:
           - Normalize Pareto front using the shared bounds
           - Calculate Hypervolume (HV) relative to reference point (1.1, 1.1)
       
       - FOR each config:
           - Calculate summary stats (mean, std, best, worst for HV, solutions, time)
           - Find median run for NSGA-II and VEGA
           - Plot and save median Pareto fronts together: "plots/pareto_<instance>_<config>.png"

4. EXPORT RESULTS:
   - Save aggregated summary table → "results/metrics.csv"
   - Save individual run details   → "results/raw_runs.csv"
```

### Algorithm Workflows (Pseudocode)

#### NSGA-II
```text
ALGORITHM: NSGA-II

1. INITIALIZATION:
   - Create initial Population of size N (random assignment + feasibility repair)
   - Evaluate objectives (f1 = opening cost, f2 = customer cost)
   - Perform Non-Dominated Sorting (assign Pareto ranks: 0, 1, 2, ...)
   - Calculate Crowding Distance for each solution (diversity measure)

2. GENERATION LOOP (Repeat for G generations):
   a. SELECTION & OFFSPRING CREATION:
      - Select parents using Tournament Selection (prefer lower rank, then higher crowding distance)
      - Apply Crossover, Mutation, and Feasibility Repair to produce N offspring
      - Evaluate objectives of offspring

   b. ELITIST SURVIVAL (Combine 2N Solutions):
      - Combine Parents (N) + Offspring (N) into a pool of 2N solutions
      - Sort the 2N pool into Non-Dominated Fronts (Front 0, Front 1, ...)
      - Fill next Population front-by-front until size N is reached:
        - If a front fits completely: Add all solutions from that front
        - If a front partially fits: Sort solutions by Crowding Distance and pick the most diverse

3. OUTPUT:
   - Return all solutions in Front 0 (the final Pareto optimal trade-offs)
```

#### VEGA
```text
ALGORITHM: VEGA

1. INITIALIZATION:
   - Create initial Population of size N (random assignment + feasibility repair)
   - Evaluate objectives (f1 = opening cost, f2 = customer cost)
   - Initialize an external Archive to store non-dominated solutions

2. GENERATION LOOP (Repeat for G generations):
   a. ARCHIVE UPDATE:
      - Add current population to Archive
      - Filter Archive to keep only unique, non-dominated solutions

   b. SUB-POPULATION SELECTION:
      - Divide mating pool into 2 halves (size N/2 each):
        - Sub-pool 1: Select N/2 individuals evaluated ONLY on Objective 1 (opening cost)
        - Sub-pool 2: Select N/2 individuals evaluated ONLY on Objective 2 (customer cost)
      - Combine and shuffle the two sub-pools to form the full mating pool

   c. REPRODUCTION:
      - Apply Crossover, Mutation, and Feasibility Repair to produce new Population of size N
      - Evaluate objectives of the new population

3. OUTPUT:
   - Update Archive with final population and return all non-dominated solutions
```

---

## Single Run (Testing Only)

Run a single instance directly via `nsga2.py` (computes NSGA-II and displays a Pareto front plot window) or `vega.py` (computes VEGA and prints non-dominated objective values to the console):

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
- AI was used for pseudocode formatting and some creation of the pseudo code
