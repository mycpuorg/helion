# EvoX-Helion Hackathon: Execution Plan

## One-Line Pitch

"We applied EvoX-style adaptive strategy switching to Helion's LFBO
autotuner — dynamically adjusting search parameters based on improvement
velocity and population diversity — reaching the same kernel quality
with fewer evaluations on MI350X."

---

## Current State (as of tonight)

### Done

| Item | File | Status |
|------|------|--------|
| Adaptive LFBO search (real Helion integration) | `stage2_evox_autotuning/adaptive_lfbo_search.py` | Complete, reviewed, tested |
| Signal tracker (velocity, diversity, error) | same file | Complete |
| Adaptive config (all thresholds/multipliers) | same file | Complete |
| Standalone prototype (reference only) | `stage2_evox_autotuning/lfbo_wrapper.py` | Complete (not used for demo) |
| Kernel profiler (Magpie-style) | `stage1_magpie/kernel_profiler.py` | Complete |
| Kernel selector (ranking formula) | `stage1_magpie/kernel_selector.py` | Complete |
| Comparison framework | `stage3_comparison/magpie_compare.py` | Complete |
| Demo slide generator | `demo/generate_slides.py` | Complete |
| Plotting utilities | `demo/plot_latency_curves.py` | Complete |
| Pushed to mycpuorg/helion | branch `pt_mar_13_2026` | Done |

### Not Done

| Item | Priority | Effort |
|------|----------|--------|
| Runner script that uses **real Helion kernels** (not simulation) | P0 | 1-2 hrs |
| Actual autotuning runs on GPU hardware | P0 | 3-5 hrs |
| Convergence curve plots from real data | P1 | 1 hr |
| Mode-transition visualization on convergence curve | P2 | 30 min |

---

## Architecture: How the Pieces Fit

```
Helion kernel (e.g. examples/softmax.py)
    |
    v
helion.autotuner.__init__.search_algorithms["AdaptiveLFBOPatternSearch"]
    |
    v
AdaptiveLFBOPatternSearch._autotune()          <-- our code
    |
    |-- inherits: _generate_neighbors()         (random perturbation within radius)
    |-- inherits: _surrogate_select()           (RF classifier + leaf-similarity diversity)
    |-- inherits: _fit_surrogate()              (retrain RF each generation)
    |-- inherits: parallel_benchmark_population (subprocess compilation + benchmarking)
    |
    |-- ADDS: SignalTracker.record_generation() (compute velocity/diversity/error)
    |-- ADDS: _adapt_parameters()               (modulate radius/quantile/similarity/neighbors/frac)
    |-- ADDS: _estimate_population_similarity() (reuse existing RF leaf-similarity)
    |
    v
Best Config returned
```

### What gets modulated per generation

| Parameter | EXPLOIT | EXPLORE | ESCAPE | Clamped to |
|-----------|---------|---------|--------|------------|
| `radius` (perturbation distance) | 1 | 3 | 6 | [1, +inf) |
| `quantile` (good/bad threshold) | 0.08 | 0.12 | 0.15 | [0.05, 0.5] |
| `similarity_penalty` | 0.5 | 1.5 | 2.0 | [0.0, +inf) |
| `num_neighbors` | 210 | 390 | 450 | [50, +inf) |
| `frac_selected` | 0.15 | 0.10 | 0.08 | [0.05, 0.5] |

Values above assume base: radius=2, quantile=0.1, similarity=1.0, neighbors=300, frac=0.10.
All transitions are EMA-smoothed (alpha=0.4) to avoid jarring changes.

---

## Tonight: Prep Work (~2 hours)

### Task 1: Write the real comparison runner

Create `run_comparison.py` that does this:

```python
import helion
from helion.autotuner import search_algorithms, LFBOPatternSearch
from examples.helion_hackathon.stage2_evox_autotuning.adaptive_lfbo_search import (
    AdaptiveLFBOPatternSearch,
)

# Register
search_algorithms["AdaptiveLFBOPatternSearch"] = AdaptiveLFBOPatternSearch

# For each target kernel:
#   1. Run LFBOPatternSearch (baseline) with fixed seed
#   2. Run AdaptiveLFBOPatternSearch with same seed and budget
#   3. Capture per-generation best_perf + wall-clock timestamps
#   4. Dump to JSON
```

Use Helion's existing examples as target kernels — they already have
`@helion.kernel` decorators and work with the autotuner out of the box.

### Task 2: Pick 2-3 target kernels

Good candidates from `helion/examples/`:

| Kernel | Why | Search space |
|--------|-----|-------------|
| `softmax.py` | Fast compile, small space, quick iteration | Small |
| `matmul.py` | Classic benchmark, large space | Large |
| `jagged_dense_bmm.py` | Interesting workload, good demo story | Medium |
| `layer_norm.py` | If exists, matches hackathon narrative | Medium |

Pick the ones that compile successfully on your target hardware.

### Task 3: Test one end-to-end run locally

Before the hackathon, verify this works:

```bash
cd /home/manoj/software/helion
HELION_AUTOTUNER=AdaptiveLFBOPatternSearch \
HELION_LOGS=all \
python examples/softmax.py
```

If it runs and produces a config, the integration is solid.

---

## Tomorrow: Hackathon Execution (~8-10 hours)

### Hour 0-1: Setup and smoke test

- [ ] Confirm target machine has: Helion, PyTorch, Triton, sklearn
- [ ] Run `python examples/softmax.py` with default autotuner — verify it works
- [ ] Run with `HELION_AUTOTUNER=AdaptiveLFBOPatternSearch` — verify no crash
- [ ] If AMD MI350X: verify ROCm compilation works
- [ ] Set the random seed: `autotune_random_seed=42` for reproducibility

### Hours 1-3: Baseline runs (control group)

For each of 2-3 kernels:

```bash
HELION_AUTOTUNER=LFBOPatternSearch \
HELION_LOGS=all \
python run_comparison.py --kernel softmax --seed 42 --mode baseline
```

- [ ] Record: best_perf at each generation, wall-clock per generation
- [ ] Record: total configs tested, total wall-clock time
- [ ] Save autotuner logs (`HELION_LOGS=all` writes to stderr)

### Hours 3-6: Adaptive runs (experimental group)

Same kernels, same seed, same budget:

```bash
HELION_AUTOTUNER=AdaptiveLFBOPatternSearch \
HELION_LOGS=all \
python run_comparison.py --kernel softmax --seed 42 --mode adaptive
```

- [ ] Record same metrics
- [ ] Additionally capture: mode transitions from `_signal_history`
- [ ] Look for the story: "started in EXPLORE, found a good region, shifted
      to EXPLOIT for refinement"

**What to look for in the logs:**

```
[Adaptive] mode=explore, velocity=0.0012, diversity=0.680, errors=5.00%
  → radius=3, quantile=0.120, sim_penalty=1.50, neighbors=390, frac=0.100
...
[Adaptive] mode=exploit, velocity=0.0150, diversity=0.420, errors=2.00%
  → radius=1, quantile=0.080, sim_penalty=0.50, neighbors=210, frac=0.150
```

### Hours 6-8: Build demo visuals

#### Slide 1 — Kernel Selection (use existing code)

```python
from stage1_magpie.kernel_selector import rank_kernels
# Feed real or simulated ProfileResults
# Output: ranked table with selection scores
```

#### Slide 2 — Convergence Curves (the money slide)

Plot best-perf vs generation for both algorithms on same axes:
- Blue line: LFBOPatternSearch (baseline)
- Orange line: AdaptiveLFBOPatternSearch
- Colored background bands showing EXPLOIT/EXPLORE/ESCAPE modes
- Annotate: "reaches within 5% of best at generation N vs generation M"

#### Slide 3 — Summary Table

```
| Kernel   | Baseline (ms) | Adaptive (ms) | Speedup | Time-to-5% (baseline) | Time-to-5% (adaptive) |
|----------|---------------|---------------|---------|----------------------|-----------------------|
| softmax  | X.XX          | X.XX          | X.XXx   | XX.Xs                | XX.Xs                 |
| matmul   | X.XX          | X.XX          | X.XXx   | XX.Xs                | XX.Xs                 |
```

### Hours 8-10: Polish

- [ ] Clean up runner script
- [ ] Ensure all results are saved to `results/` as JSON
- [ ] Prepare 3-minute verbal walkthrough
- [ ] Bonus: show LLM warm-start as "future direction" slide

---

## The Demo Story (3 minutes)

**Minute 1 — Problem:**
"Helion's LFBO autotuner uses fixed search parameters throughout the
entire search. But the optimal strategy changes: early on you want broad
exploration, later you want tight exploitation. Fixed parameters are a
compromise that's suboptimal at every phase."

**Minute 2 — Solution:**
"We applied EvoX's insight: make the search strategy itself adaptive.
We monitor three signals per generation — how fast we're improving,
how diverse our candidates are, and how often compilation fails — and
switch between EXPLOIT, EXPLORE, and ESCAPE modes. This is a single
Python file that subclasses the real LFBOPatternSearch."

**Minute 3 — Results:**
"On [kernel], the adaptive search reaches within 5% of the best config
in N generations vs M generations for the baseline — X% faster with
the same final quality. Here's the convergence curve showing the mode
transitions. All tested on MI350X with AMD tooling."

---

## Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Adaptive doesn't beat baseline on latency | Medium | The convergence speed story still works — show time-to-5%-of-best |
| Hardware issues on MI350X | Medium | Fall back to any NVIDIA GPU; code is hardware-agnostic |
| Autotuning takes too long per kernel | Low | Reduce to `max_generations=10, initial_population=50` |
| Not enough time for 3 kernels | Medium | 1 kernel with a clean convergence curve is a complete demo |
| sklearn not installed | Low | `pip install scikit-learn` — single dependency |
| Adaptive causes crashes | Very low | Code-reviewed, all edge cases tested; worst case use standalone prototype |

---

## File Map

```
examples/helion_hackathon/
├── CLAUDE.md                          # Project docs + architecture
├── HACKATHON_PLAN.md                  # This file
├── run_pipeline.py                    # Standalone simulation (reference)
├── run_comparison.py                  # TODO: real Helion kernel runner
├── requirements.txt
│
├── stage1_magpie/                     # Kernel profiling + selection
│   ├── kernel_profiler.py             #   CUDA event profiling
│   ├── kernel_selector.py             #   Ranking by wall_time * log(space)
│   └── sample_workload.py             #   Simulated transformer workload
│
├── stage2_evox_autotuning/            # Core adaptive autotuning
│   ├── adaptive_lfbo_search.py        #   REAL integration (subclasses LFBOPatternSearch)
│   ├── adaptive_controller.py         #   Original standalone controller (reference)
│   ├── signals.py                     #   Original standalone signals (reference)
│   ├── lfbo_wrapper.py                #   Standalone prototype (reference)
│   └── llm_warm_start.py              #   Path B: LLM-seeded warm start
│
├── stage3_comparison/                 # Result comparison
│   ├── magpie_compare.py              #   Side-by-side metrics + formatting
│   └── results_formatter.py           #   Demo table generation
│
└── demo/                              # Visualization
    ├── generate_slides.py             #   Markdown slide generation
    └── plot_latency_curves.py         #   Matplotlib convergence plots
```
