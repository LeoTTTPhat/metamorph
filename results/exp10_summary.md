## Drift-Adapter protocol, MiniLM-L6 → MPNet (all-mpnet-base-v2)

Query mapped into the old space; 20,000 training pairs (Emotion: its 15,956 items); FAISS HNSW M=32, efConstruction=200, efSearch=50; 5 runs. *Old* is the un-upgraded system (old queries, old index, no adapter). *Conv.* and *Grid* are the best LA/MLP trained to convergence or tuned over Drift-Adapter's own hyperparameter ranges. *Lenient* counts the best adapter's top-10 against the new model's top-100. *Task* is class-label precision@10 (classification corpora) or NDCG@10 (fiqa, MS MARCO) of the best adapter and of the old model, relative to the re-embedded system.

| Benchmark | Old | OP | OP+DSM | LA | MLP | Conv. | Grid | Lenient | Task (adapter / old) | Published OP / LA / MLP |
|---|---|---|---|---|---|---|---|---|---|---|
| AG-News | 0.546 | 0.500 | 0.499 | 0.308 | 0.481 | 0.484 | 0.485 | 0.901 | 0.99 / 1.00 | 0.974 / 0.983 / 0.992 |
| DBpedia-14 | 0.363 | 0.245 | 0.244 | 0.103 | 0.221 | 0.223 | 0.224 | 0.611 | 1.02 / 0.97 | 0.968 / 0.979 / 0.990 |
| Emotion | 0.306 | 0.390 | 0.387 | 0.263 | 0.359 | 0.359 | 0.360 | 0.839 | 0.99 / 1.10 | 0.953 / 0.967 / 0.984 |
| fiqa | 0.397 | 0.423 | 0.420 | 0.270 | 0.391 | 0.390 | – | 0.888 | 0.68 / 0.73 | – |
| MS MARCO (1M rung) | 0.545 | 0.411 | 0.411 | 0.156 | 0.384 | 0.383 | – | 0.808 | 0.68 / 0.95 | – |

## Checks

| Check | Result (ARR) |
|---|---|
| Sanity, identity upgrade (Emotion) | OP 1.000, LA 0.589, MLP 0.996 |
| Sanity, rotation upgrade (Emotion) | OP 1.001, LA 0.588, MLP 0.886 |
| Training pairs, AG-News, OP | 1k 0.432, 2k 0.466, 5k 0.488, 10k 0.497, 16k 0.500, 20k 0.500 |
| Training pairs, AG-News, MLP | 1k 0.429, 2k 0.457, 5k 0.471, 10k 0.477, 16k 0.480, 20k 0.481 |
| Corpus direction (old corpus → new space), agnews | best 0.524 |
| Corpus direction (old corpus → new space), dbpedia14 | best 0.271 |
| Corpus direction (old corpus → new space), emotion | best 0.403 |
| Destination e5-small-v2, agnews | best adapter 0.416, old 0.478 |
| Destination e5-small-v2, dbpedia14 | best adapter 0.156, old 0.242 |
| Destination e5-small-v2, emotion | best adapter 0.288, old 0.288 |
| Destination gte-small, agnews | best adapter 0.437, old 0.533 |
| Destination gte-small, dbpedia14 | best adapter 0.185, old 0.366 |
| Destination gte-small, emotion | best adapter 0.365, old 0.389 |
| fiqa:minilm:e5s | old 0.351; OP 0.012, LA 0.155, MLP 0.263 |
| msmarco1m:minilm:e5s | old 0.661; OP 0.032, LA 0.155, MLP 0.452 |
| fiqa:minilm:mpnet, 1,153 pairs (2% of the corpus) | OP 0.377, LA 0.014, MLP 0.366 |
| fiqa:minilm:e5s, 1,153 pairs (2% of the corpus) | OP 0.009, LA 0.001, MLP 0.085 |

Largest run-to-run standard deviation of any adapter ARR above: 0.0055.
