# Drift-Adapter reproduction (exp10)

This note documents our reproduction of Drift-Adapter (H. Vejendla, *Drift-Adapter:
A Practical Approach to Near Zero-Downtime Embedding Model Upgrades in Vector
Databases*, EMNLP 2025, pp. 15938–15949; arXiv 2509.23471). It lists every
setting the Drift-Adapter paper states, what we ran, and the settings it leaves
open or states inconsistently together with our choice for each.

No official code accompanies Drift-Adapter (checked 2026-09-27: none linked from
its publication, the ACL Anthology page or arXiv).

## Stated protocol and what we ran

| Setting | Drift-Adapter (stated; section) | This reproduction |
|---|---|---|
| Direction | new-model query mapped into the old space, `q' = g(f_new(q))`; old index searched unchanged (§3) | same |
| Adapters | OP: closed-form orthogonal Procrustes, `R = U V^T` from SVD(A B^T); LA: `U V^T x + t`, rank 64; residual MLP: `x + W2 GELU(W1 x + b1) + b2`, 256 hidden, dropout 0.1 (§3, §4, App. A) | same |
| Diagonal scaling (DSM) | learned diagonal on the adapter output; on for LA/MLP, off for OP (§3) | same; `OP+DSM` also reported as a variant |
| Training | MSE; AdamW lr 3e-4, weight decay 0.01, batch 256, ≤ 50 epochs, early stopping on validation MSE (patience 5); 80/20 split for LA/MLP, OP on all pairs (§4, App. A.2) | same |
| Pairs | Np = 20,000 random database documents (2% of 1M), distinct from queries; 5 runs, fresh sample each (§4, §5.1) | same (Emotion: all 15,956 training items) |
| Datasets | AG-News, DBpedia-14, Emotion, "standard splits and data from the MTEB benchmark"; database = 1M items sampled from the training split; queries = 10,000 test documents (§4) | database = the full original training split (120,000 / 560,000 / 15,956); queries = all test items or a seeded sample of 10,000 (7,600 / 10,000 / 1,986) |
| Models | all-MiniLM-L6-v2 → all-mpnet-base-v2 (§4) | same (sentence-transformers 6.0.0) |
| Normalization | all embeddings L2-normalized before adapters and indexing (§4) | same |
| Index | FAISS HNSW, M=32, ef_construction=200, ef_search=50, over old vectors (§4) | same (faiss-cpu 1.15.0) |
| Ground truth | exhaustive kNN with new-model embeddings for queries and database (§4) | same (FAISS `IndexFlatIP`, k=10; top-100 for the lenient metric) |
| Metric | Recall@10, MRR; ARR = Recall_adapter / Recall_NewModelDirect (§4) | same; the re-embedded system is new queries on a new-model HNSW index with the same parameters |

## Settings left open or stated inconsistently, and our choices

1. **Database size.** A 1M-item database cannot be sampled from these training
   splits without duplication (AG-News 120,000; DBpedia-14 560,000; Emotion
   15,956; MTEB's own `DBpediaClassification` has 2,048). We use the full
   training split, the largest database the stated splits allow.
2. **Dimension mismatch.** The stated pair maps 768 → 384 dimensions, while OP
   (`R ∈ R^{d×d}`), the MLP residual (`x + …`) and the "misaligned" baseline
   (new queries on the old index) are written for equal dimensions. We use the
   rectangular Procrustes solution `R = U V^T` (thin SVD), make that fixed `R`
   the MLP's projection shortcut, and report the misaligned baseline only for
   equal-dimension pairs.
3. **Optimizer.** §3 says the adapters are trained with SGD, §4 and App. A.2
   say AdamW. We use AdamW, the more detailed statement.
4. **MRR.** Not defined. We use MRR@10 of the true nearest neighbour.
5. **DSM initialization, checkpoint.** Not stated. DSM starts at 1; the
   best-validation checkpoint is restored.
6. **LA rank.** As written (`U V^T x + t`, rank 64, no identity term) LA cannot
   represent even an identity upgrade; our identity sanity check gives ARR ≈ 0.59
   for LA. We report LA as written.

## Additional checks (not in the Drift-Adapter paper)

- **Sanity:** identity upgrade (old = new) and a random rotation of the old
  space. A correct pipeline must reach ARR ≈ 1.0 with OP; it does.
- **Training recipes:** `converged` (same hyperparameters, up to 1,000 epochs,
  patience 10) and `grid` (lr ∈ {1e-4, 3e-4, 1e-3} × {LA r=64, MLP 256; LA r=128,
  MLP 512}, up to 500 epochs, patience 10, chosen on validation MSE only), i.e.
  the paper's own sensitivity ranges (App. A.2).
- **Sample size:** Np ∈ {1k, 2k, 5k, 10k, 16k, 20k} on AG-News (cf. Fig. 1 of the Drift-Adapter publication).
- **Direction:** the same adapter families fitted old → new and applied to every
  stored vector (the direction of exp3's pure-adapter ablation).
- **Destinations:** our E5-small-v2 and GTE-small on the same datasets.
- **Retrieval corpora:** fiqa (57,638) and the 1M-passage MS MARCO ladder rung.
- **Lenient metric:** the adapter's top-10 counted against the new model's top-100.
- **Task metric:** class-label precision@10 on the classification corpora, NDCG@10
  on BEIR, each relative to the re-embedded system.

## Running

Prerequisites: the fiqa settings reuse `data/prepared/fiqa_minilm-*_full.npz`
(`prepare_data.py --dataset fiqa --scale full` for pairs `minilm-e5`, `minilm-gte`,
`minilm-e5large`), and the MS MARCO settings use the 1M rung of the full MS MARCO
bundle (`prepare_large.py --dataset msmarco --pair minilm-e5 --tag xxl`, about 28 GB
and 3 h on the reference machine). The three classification corpora need nothing
beyond `prepare_drift.py`.

```bash
# embeddings (MPS): the three classification corpora with four encoders,
# then MPNet for fiqa and the 1M MS MARCO rung
python experiments/prepare_drift.py --datasets emotion agnews dbpedia14 --models minilm mpnet e5s gtes
python experiments/prepare_drift.py --datasets emotion agnews dbpedia14 --labels-only
python experiments/prepare_drift.py --datasets fiqa msmarco1m --models mpnet
# e.g. the reproduction on Drift-Adapter's benchmarks (its published recipe, 5 runs)
python experiments/exp10_driftadapter_repro.py --settings agnews:minilm:mpnet dbpedia14:minilm:mpnet emotion:minilm:mpnet --recipe paper --tag mytag
# Markdown summary of every exp10 result file -> results/exp10_summary.md
python experiments/summarize_exp10.py
```

The exact command behind every shipped exp10 result file is listed in
`results/RUNS.md`. `--recipe` selects `paper` (Drift-Adapter's published training
recipe), `converged` or `grid`; `--directions` selects `query->old` (Drift-Adapter)
and/or `corpus->new`.

PyTorch and FAISS each link their own `libomp` on macOS and abort when both load
in one process, so adapters are trained in a spawned torch-only worker
(`drift_train.py`) and applied in NumPy; every fit checks the NumPy inference
against the torch forward pass.

## Result files

`results/exp10_driftadapter_<tag>.csv` (one row per setting × method × run),
`…_summary.csv` (mean and s.d. over runs), `…csv.meta.json` (protocol and
choices), and `results/exp10_summary.md` (all tables, from `summarize_exp10.py`).
The exact command behind each result file is listed in `results/RUNS.md`.
Tags: `paper1`, `paper2`, `paper3a`, `paper3b`, `paper4` (Drift-Adapter's published
recipe, 5 runs), `npsweep`, `seed2pct`, `converged`, `converged2a`, `converged2b`
(trained to convergence), and `grid` (Drift-Adapter's own hyperparameter ranges).
