# Provenance — Metamorph evaluation

Generated 2026-08-23. This file records the machine, the code state, the exact
corpus sizes, and which results were produced by which configuration, so every
reported number can be traced to a run.

## Machine

- Apple M3 Ultra (arm64), 32 cores, 512 GB RAM, no discrete GPU.
  Encoder inference uses PyTorch **MPS**; all retrieval and experiment code is
  CPU numpy / hnswlib / FAISS.
- macOS 26.5.2 (build 25F84)
- Python 3.12.13 (project venv `metamorph/.venv`)
- numpy 2.5.2, torch 2.13.0 (MPS available), sentence-transformers 6.0.0,
  hnswlib 0.8.0, faiss-cpu 1.15.0, scikit-learn, pandas, matplotlib

**Concurrency caveat.** Some runs shared the machine with other workloads.
Throughput measurements (`exp7`) were taken with the machine otherwise idle —
the large corpus encode was deliberately paused for them — and this is the
only place where absolute wall-clock is load-sensitive in a way that affects a
claim. Latency measurements (`exp6`) are single-threaded per query but were
taken on a shared machine; they are reported as relative comparisons
(compacting vs shared masked index) rather than as absolute service levels.
The FAISS build times in `exp4b` were likewise measured while large ladder
runs were in flight, so the absolute seconds are conservative; the cold and
warm builds of a given size always ran back to back in the same process, so
the reported *ratios* are unaffected by load.

## Code state

- No git repo (`git_rev: n/a` in all sidecars).
- Full pytest suite: **82 tests, all passing**, including
  - `test_bounds.py`: coverage and *anytime* (repeated-peeking) validity of the
    betting confidence sequence, its dominance over empirical Bernstein in the
    certifier's regime, and a 300-instance randomised soundness check that the
    per-query certificate never exceeds true pool recall;
  - `test_certify_targets.py`: score-mass bounds, truncation-probe behaviour,
    the bias correction only ever lowering the decision bound, and pooling
    windows;
  - `test_semantics_adversarial.py`: the invariants of the migration state
    machine (I1 exactly-once serving, I2 monotonicity, I3 budget conservation,
    I4 legal phase transitions) under **adversarial interleavings** -- in
    particular the case the design specifies semantics for, a scheduler batch
    flipping while queries are being answered. The batch flipped is drawn from
    the candidate set of the query just served, so it targets exactly the
    documents the next query is most likely to touch. Mutation-checked: with an
    I1 violation injected the assertions fire, and `check_invariants()` detects
    a corrupted partition count.
  - a distribution-free check that the split-conformal residual quantile used
    as the fusion-error surrogate covers a held-out residual at its nominal
    rate under three residual distributions.

### Configuration changes, and what was re-run

The certifier's defaults changed twice during development (round-cost caps,
then probe rate and probe depth). Because certification reveals are recycled
into migration progress, those settings change the *migration trajectory*, not
only the certification columns. Every affected experiment was therefore re-run
from scratch under the final configuration rather than mixed:

| setting | final value | rationale |
|---|---|---|
| `bound` | `betting` | anytime-valid; a certifier peeks after every round |
| `metric` | `recall` (`score_mass` is reported alongside) | |
| `probe_frac` | 1.0 | the bias bound is sample-count limited; a cheap probe on every query beats a deep probe on a few |
| `probe_multiplier` | 2 | ditto |
| `min_probe` | 30 | abstain rather than certify against an uninformative bias bound |
| `history_window` | 0 (pool the whole stream) | licensed by the anytime bound |
| `max_round_cost_frac` | 0.02 / 0.012 / 0.008 / 0.002 / 0.001 (smoke / small / full / xl / xxl) | a round costs O(m·\|pool\|) reveals, so its cost *as a fraction of N* falls like 1/N |
| `compact_old_index_ratio` | 0.5 | rebuild the old index when the live set halves |
| `anneal_disagreement` | true (global, coverage-based) | the `per_doc` and `False` forms are ablated in exp2; the ablation shows all three trail traffic-only |

Runs that predate the final configuration were deleted or overwritten; the
`.meta.json` sidecar of every surviving CSV records the exact `argv` and
engine configuration that produced it.

## Encoder pairs

Five real public upgrades. The first four are compact sentence encoders; the
fifth was added to test whether the per-query fusion calibration depends on the
destination being small.

| tag | old | new | dims | new params |
|---|---|---|---|---|
| m->e | all-MiniLM-L6-v2 | e5-small-v2 | 384 -> 384 | 33M |
| m->g | all-MiniLM-L6-v2 | gte-small | 384 -> 384 | 33M |
| a->e | paraphrase-albert-small-v2 | e5-small-v2 | 768 -> 384 | 33M |
| m->eb | all-MiniLM-L6-v2 | e5-base-v2 | 384 -> 768 | 109M |
| **m->eL** | all-MiniLM-L6-v2 | **e5-large-v2** | **384 -> 1024** | **335M** |

The large-destination pair behaves in family. On scifact the pure adapter
reaches 0.263 at a 2% seed and 0.403 at 10% (compact pairs: 0.157-0.273 and
0.297-0.420), the full engine reaches 0.997 final agreement at the lowest
budget-to-0.95 of the four pairs (0.951), and at 50% budget Metamorph leads
random order by 8.8 workload points while the offline prior leads Metamorph by
2.5 -- the same sub-10^4 ordering the other pairs show.

A **controlled** check was also run: fiqa (57,638) with the corpus, workload and
source encoder fixed and only the destination varied.

| destination | dims | params | adapter @2% | @10% | b95 | final |
|---|---|---|---|---|---|---|
| e5-small-v2 | 384 | 33M | 0.256 | 0.309 | 0.907 | 0.971 |
| e5-large-v2 | 1024 | 335M | 0.245 | 0.297 | 0.889 | 0.985 |

The two are indistinguishable on the adapter side and the large destination is
slightly better on both engine metrics.

## Exact scale

Every CSV row carries `n_docs`; every sidecar carries the full configuration.

| dataset | docs | queries | role |
|---|---|---|---|
| nfcorpus | 3,633 | 323 | pair sweep |
| scifact | 5,183 | 300 | pair sweep, ablations |
| fiqa | 57,638 | 648 | pair sweep, headline ablations |
| quora | 522,931 | 8,000 (capped) | scale ladder, live-encoder cost |
| MS MARCO passages | 8,841,823 | 6,980 dev | scale ladder, headline scale |

The MS MARCO bundle is the full v1 passage collection, embedded end to end with
both encoders on this machine: 10,797 s of wall-clock for the two document
passes (`prepare_seconds` in the bundle meta), 13.58 GB per space on disk.

Scale-ladder rungs are **sub-samples of one prepared bundle**, keeping every
qrel-positive document plus a seeded random fill, so dataset, encoder pair and
workload are held fixed while `N` varies. This is the only way the trend can be
read as a scale trend rather than a dataset difference.

## Ladder seed completion (2026-08-24)

The two largest MS MARCO rungs and the two largest quora rungs were originally
single-seed (seed 42). Seeds 43 and 44 were added for all four, so every
reported rung is a three-seed mean:

| tag | rungs | seeds | wall-clock |
|---|---|---|---|
| `msmarco_..._b2` | 4,000,000 | 43, 44 | launched 10:05, done 16:18 (shared machine) |
| `msmarco_..._d2` | 8,841,823 | 43, 44 | launched 10:06, done 16:18 (shared machine) |
| `quora_..._b2` | 250,000 and 522,931 | 43, 44 | 16:20 -> 16:52 |

The added seeds moved the top rungs materially, in both directions -- the
single-seed values were not representative:

| rung | vs random | vs prior |
|---|---|---|
| MS MARCO 4x10^6 | 0.133 -> 0.104 | 0.102 -> 0.079 |
| MS MARCO 8.84x10^6 | 0.118 -> 0.085 | 0.099 -> 0.105 |
| quora 250,000 | 0.045 -> 0.041 | 0.044 -> 0.060 |
| quora 522,931 | 0.033 -> 0.045 | 0.044 -> 0.054 |

All reported values use the three-seed means. The qualitative picture changed
with them: the margin over random no longer "plateaus near +0.12" but
peaks at 4x10^6 and eases at full scale, while the margin over the *prior* --
the comparison the scale argument actually rests on -- rises at every MS MARCO
rung and is widest on the full collection. On quora the dip at the largest rung
present under one seed disappears under three.

## Scale-ladder pooling

Each ladder rung is reported as a mean over **all** comparable rung files, so
the pooled aggregate -- not any single tag -- is the reported value.

Rungs are comparable only under an identical workload. Tag `c`
(`exp9_ladder_msmarco_..._c.csv`) was an exploratory 8.84M pass at
`--max-queries 3000` covering only two methods; it was superseded by tag `d` at
the ladder's standard 4,000 queries and all three methods.
`make_figures.py` therefore drops any rung file whose recorded
`--max-queries` differs from the ladder's 4,000, which excludes tag `c` and
the 2,000-query quora smoke run. Before this filter existed the figure
averaged `c` into the 8.84M point, mixing two workloads at the one place the
scale claim is read off; the reported numbers were unaffected because they were
keyed by tag.

All four MS MARCO rungs and all five quora rungs carry **3 seeds** (42/43/44);
each rung file's sidecar records its seeds.

## Cost accounting

Two stores implement the same interface:

- `EmbeddingStore` reads new-model embeddings from a precomputed oracle and
  charges the budget meter. This is what makes a full migration timeline
  replay in minutes with exact cost accounting.
- `LiveEmbeddingStore` (exp7) runs the **real encoder** on every reveal. Its
  vectors were checked against the oracle: maximum cosine gap **0.0** over
  sampled reveals, so a live run and a replayed run are the same experiment,
  and the budget meter is grounded in measured wall-clock.

## Reproducing

```bash
./.venv/bin/python -m pytest src/tests -q          # 82 tests
./.venv/bin/python experiments/make_figures.py     # -> results/figures
./.venv/bin/python experiments/summarize_exp10.py  # -> results/exp10_summary.md
```

`results/RUNS.md` lists, for every result file, the exact command that produced
it (reconstructed from the `argv` recorded in its `.meta.json` sidecar).

## Drift-Adapter reproduction (exp10, 2026-09-27)

Produced by `experiments/exp10_driftadapter_repro.py` with embeddings from
`experiments/prepare_drift.py` (all-MiniLM-L6-v2, all-mpnet-base-v2,
e5-small-v2, gte-small on AG-News, DBpedia-14 and Emotion; all-mpnet-base-v2 on
fiqa and the 1M MS MARCO rung, i.e. `load_bundle(n_docs=1_000_000, seed=42)`).
Result tags: `paper1`, `paper2`, `paper3a`, `paper3b`, `paper4` (Drift-Adapter's
published recipe, 5 runs), `npsweep`, `seed2pct`, `converged`, `converged2a`,
`converged2b`, `grid`. `summarize_exp10.py` turns the summaries into
`results/exp10_summary.md`. The protocol, and every setting the Drift-Adapter
paper leaves open with our choice, is in `experiments/DRIFT_ADAPTER_REPRODUCTION.md`. These runs do not touch the
migration engine or its certifier configuration, so no earlier result changes.
Runs shared the machine with the embedding jobs; they report retrieval quality
only, which is load-independent.

