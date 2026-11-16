# Checkpoint 3: dev experiments and selection

All retrieval experiments use the same 160 strategy B chunks and 15 answerable dev questions. The original query is used without metadata filters. Dense means Day 6 exact cosine with BGE; lexical means the starter's TF-IDF; hybrid fuses top 20 from each using reciprocal rank fusion `1/(60+rank)`. The first three configurations pack top 3 in a 4,000-character context; a fourth packs top 5 hybrid candidates in 6,000 characters. The gold evidence is used only for evaluation after retrieval. Hit requires the exact labelled evidence to occur in one chunk with matching source and physical page; context hit additionally requires that chunk in the packed context. These automatic metrics can miss semantically equivalent evidence and do not judge answer correctness.

| Configuration | Hit@1 | Evidence Hit@3 | Context evidence | Mean retrieval ms |
|---|---:|---:|---:|---:|
| Dense | 12/15 | 13/15 | 13/15 | 49.4 |
| TF-IDF | 11/15 | 12/15 | 12/15 | 1.2 |
| Hybrid top 3 | 8/15 | 14/15 | 14/15 | 50.7 |
| Hybrid top 5 | 8/15 | 14/15 | 15/15 | 50.7 |

Times are a single warm run, include query embedding for dense and the sum of both searches for hybrid, but exclude one-time index/model initialization. Local service cost is zero, excluding hardware and electricity. Traces and exact observed timings are in `results/checkpoint3/dev/`; all configurations show top-3 ranking at the same cutoff. The hybrid top-5 context is a second improvement configuration, distinct from the hybrid method itself.

Dev examples: dense missed q01 and q09 in top 3, but hybrid recovered both. In contrast, hybrid top 3 missed q05 even though dense ranked its evidence first; expanding hybrid context to five recovered it. The checkpoint 2 agent still answered q05 incorrectly despite having the correct chunk in context, so retrieval hit is insufficient evidence of a correct answer.

Before inspecting the completed hybrid runs or running test, the selection rule was: (1) highest total manually graded grounded-answer score across all 15 answerable dev cases, treating errors as zero for selection; (2) most correct abstentions across the two unanswerable dev cases, requiring no unsupported extra claim; (3) highest automatic context evidence count; then (4) lower mean end-to-end time. `src/agent_dev.py` ran both configurations against all 17 dev questions. The model rewrote the queries; those queries and the final contexts are in the agent traces, separate from the original-query retrieval comparison above.

| Agent | Grounded grades 0 / 1 / 2 | Grade sum / 30 | Correct abstention | Supported citations | Answerable abstentions | Mean time | Errors |
|---|---|---|---|---|---|---|---|
| Dense | 11 / 0 / 4 | 8 / 30 | 1 / 2 | 4 / 11 | 5 / 15 | 34.5 s | 0 |
| Hybrid top 5 | 10 / 1 / 4 | 9 / 30 | 1 / 2 | 5 / 10 | 7 / 15 | 49.7 s | 0 |

All 17 cases per configuration made exactly one retrieval call. Measured final agent-context evidence rates were 13/15 dense and 14/15 hybrid; these differ from the original-query figures because the model chose rewritten searches. The local model and retrieval have no API charge, excluding equipment and electricity. Each grade, citation judgment and reason is in `results/checkpoint3/dev_reviews.json`; the Codex assistant read the cited chunks against the gold, without independent human adjudication. The unanswerable hybrid u02 explicitly says the footprint is not stated and adds a comparison supported by the cited table; it counts as an abstention even though the starter labels any cited response `answered`.

The predeclared rule selects **hybrid top 5**, with a 6,000-character budget, `llama3.2:3b`, and up to three searches; its 9/30 grounded score narrowly exceeds dense's 8/30. Selection and timestamp are frozen in `results/checkpoint3/selection.json` before test. Quality is poor, hybrid is slower, and one of two dev missing-evidence questions still received an unsupported dollar figure. These small, AI-reviewed results do not establish general performance.
