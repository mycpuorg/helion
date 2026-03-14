The Complete Hackathon Project: EvoX-Helion + Magpie
The pipeline now has three distinct, well-defined stages and a compelling demo arc.

The Story in One Sentence
"We used Magpie to find which kernels matter most in real AMD workloads, applied EvoX-style adaptive autotuning to Helion on those kernels, and showed side-by-side latency speedups on MI350X."
This is differentiated from every other team at the hackathon because: AMD hardware, AMD tooling, and a research paper applied to a real open question the Helion team explicitly said they want answered.

## Implementation Status

### Real Integration (adaptive_lfbo_search.py) — THE MAIN DELIVERABLE
`stage2_evox_autotuning/adaptive_lfbo_search.py` contains `AdaptiveLFBOPatternSearch`,
which **subclasses the real `helion.autotuner.surrogate_pattern_search.LFBOPatternSearch`**.
MRO: AdaptiveLFBOPatternSearch → LFBOPatternSearch → PatternSearch → PopulationBasedSearch → BaseSearch

It modulates these real LFBO parameters per generation based on EvoX signals:
- `self.radius` — perturbation distance in config space
- `self.quantile` — classifier threshold for good/bad split
- `self.similarity_penalty` — diversity penalty coefficient
- `self.num_neighbors` — random neighbors generated per generation
- `self.frac_selected` — fraction of candidates to actually benchmark

Diversity is measured via the **existing** RandomForest leaf-similarity mechanism
(not Hamming distance), giving functional similarity rather than raw parameter distance.

To register in Helion and use via settings:
```python
# In helion/autotuner/__init__.py, add to search_algorithms:
from examples.helion_hackathon.stage2_evox_autotuning.adaptive_lfbo_search import AdaptiveLFBOPatternSearch
search_algorithms["AdaptiveLFBOPatternSearch"] = AdaptiveLFBOPatternSearch

# Then use:  settings.autotune_search_algorithm = "AdaptiveLFBOPatternSearch"
```

### Standalone Prototype (lfbo_wrapper.py) — FOR REFERENCE ONLY
The original standalone reimplementation. Useful for offline testing without GPU,
but does NOT use Helion's real infrastructure (no parallel compilation, no FlatConfig,
no ConfigGeneration, no subprocess benchmarking).

Stage 1: Magpie as the Kernel Selection Oracle
Magpie's discover_kernels tool scans a project and suggests analyzable kernels/configs, while compare does multi-kernel comparison and ranking. github This is your entry point — not a random selection of kernels to optimize.
What you do:
Run Magpie's discover_kernels against a real workload — a training loop or inference stack (you could use a small transformer model, or pull from your existing SGLang/vLLM work). Magpie will profile which kernels are actually hot. This gives you a ranked list grounded in real execution data, not synthetic benchmarks.
Then rank by two criteria:

Wall-time contribution (from Magpie's analyze output — how much total execution time does this kernel account for?)
Configuration space size (from Helion's implicit search space — how many possible configs exist? Larger = more room for EvoX-style adaptive search to win)

The intersection of high wall-time contribution and large search space is your target set. This is where adaptive autotuning has maximum leverage.
Prompt for Magpie's MCP integration (since Magpie has an MCP server with suggest_optimizations and hardware_spec tools github):
Given these kernel profiles from a transformer inference workload on MI350X,
rank them by (execution_time_fraction * log(config_space_size)).
For the top 5, suggest which Helion configuration dimensions have highest
variance in latency outcomes based on the profiling data.
This creates a principled kernel selection that you can show in 2 minutes at demo time: "We didn't just pick LayerNorm because it's famous — Magpie told us it accounts for 18% of wall time and has 10^14 possible configs."

Stage 2: EvoX-Style Adaptive Autotuning in Helion
Build the adaptive strategy switcher described previously, but now targeted specifically at the kernels Magpie selected. Two concrete implementation paths:
Path A (4-6 hours): State Machine Meta-Controller
Instrument Helion's LFBOPatternSearch with three observable signals:

improvement_velocity: rolling average of latency delta over last K evaluations
population_diversity: mean Hamming distance between recently sampled configs
error_pressure: compile error/timeout rate in last batch

Map these to three modes:
Signal StateModeWhat ChangesHigh velocity, any diversityEXPLOITTighten perturbation radius, raise classifier confidence thresholdLow velocity, high diversityEXPLOREWiden perturbations, lower threshold, allow cross-parameter jumpsLow velocity, low diversityESCAPEForce random restart from unexplored region of config space
The existing LFBO approach generates candidates via random multi-parameter perturbations, trains a RandomForest classifier to predict top-10% configs, and penalizes similarity via leaf node co-occurrence. pytorch Your changes touch only the perturbation radius and classifier threshold — the rest of the pipeline is unchanged. This is a minimal, targeted diff.
Path B (2-3 hours, fallback): LLM-Seeded Warm Start
If Path A proves too deep to instrument in time, do the LLM warm-start version: after the initial random phase, serialize top-20 configs + their latencies, prompt an LLM for 10 novel configs exploiting patterns, inject those as additional search seeds. Measurable, self-contained, and directly demonstrates EvoX's insight that the selection mechanism matters.

Stage 3: Magpie for Side-by-Side Evaluation
This is where Magpie closes the loop and makes the demo visual.
Magpie's compare mode does multi-kernel comparison and ranking, with structured JSON output for pipeline integration, and supports both AMD HIP and NVIDIA CUDA. github
The evaluation flow:
For each target kernel from Stage 1:

Run Helion with baseline LFBO Pattern Search → extract best config → compile final Triton kernel
Run Helion with your EvoX-adaptive strategy → extract best config → compile final Triton kernel
Feed both compiled kernels into Magpie's compare mode on MI350X
Magpie outputs structured JSON with latency, correctness verification, and hardware utilization

Magpie kernel config structure for comparison:
yamlkernels:
  - name: "layernorm_baseline_lfbo"
    source: generated_triton/layernorm_lfbo_best.py
    hardware: MI350X
  - name: "layernorm_evox_adaptive"  
    source: generated_triton/layernorm_evox_best.py
    hardware: MI350X
testcases:
  - shape: [2048, 4096]
  - shape: [4096, 8192]
metrics:
  - latency_ms
  - memory_bandwidth_utilization
  - correctness
Magpie then produces the side-by-side table you can put directly in the demo slide.
The metric that makes the story tight:
Don't just show "best latency found." Show three things together:

Best latency achieved (absolute, in ms)
Time to reach that latency (wall-clock seconds of autotuning)
Number of configs evaluated to reach it

This is the full EvoX efficiency argument: not just better final quality, but better quality faster with fewer evaluations.

Hackathon Timeline (Assuming 8-10 hours)
Hours 0-1: Magpie profiling run on a real workload, extract top-5 kernels by the ranking criterion. Confirm Helion can compile those kernels on your hardware.
Hours 1-2: Baseline LFBO autotuning run on all 5 kernels. Record latency-vs-time traces. This is your control group.
Hours 2-6: Implement the adaptive strategy switcher (Path A) or LLM warm-start (Path B). Start autotuning runs in parallel on the same 5 kernels.
Hours 6-8: Feed results into Magpie's compare mode. Generate structured output. Build the side-by-side table.
Hours 8-10: Polish demo, write up the EvoX connection, record latency traces plot (same format as the LFBO blog's Figure 4 — latency of best config over time).

The Demo Slide Structure
Three panels, one slide each:
Slide 1 — Kernel Selection (Magpie)
Table: Kernel name | Wall-time % | Config space size | Selection score
Bottom line: "We targeted these 5 kernels because they represent X% of model execution time"
Slide 2 — Adaptive Autotuning (EvoX + Helion)
Two latency-vs-time curves per kernel: LFBO baseline vs. adaptive strategy
Key callout: time-to-within-5%-of-best, showing adaptive converges faster
Slide 3 — Side-by-Side Results (Magpie compare)
Final comparison table from Magpie JSON output
Best latency per kernel, improvement %, hardware utilization delta on MI350X

