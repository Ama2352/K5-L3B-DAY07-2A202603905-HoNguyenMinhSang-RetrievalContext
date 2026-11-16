# Additional retrieval methods and function tools

The six functions and Ollama function schemas are defined in `src/retrieval_tools.py`. `src/function_agent.py` sends the schemas to Ollama's `/api/chat` and dispatches the model's actual function calls. The functions are:

| Function | Method |
|---|---|
| `search_dense` | BGE query embedding and Day 6 pgvector exact cosine search. |
| `search_tfidf` | TF-IDF lexical retrieval over the same chunks. |
| `search_bm25` | BM25 lexical retrieval, k1=1.5, b=0.75. |
| `search_hybrid` | Dense and BM25 top-20, fused with RRF k=60. |
| `search_rerank` | Hybrid top-20 candidates reranked by distinct query-term coverage; this is a lexical heuristic, not a cross-encoder. |
| `search_multistage` | BM25 top-3, deterministic query expansion from the top result, dense top-20, then RRF. Both queries and intermediate rankings are traced. |

All functions take a non-empty `query` and optional `top_k` (1–10). No metadata filter is applied. Arguments, intermediate steps, ranked results with IDs, scores, text, source, page and metadata, context, citation-label mapping, model calls and token counts, tool/model timing, errors and local service cost are saved. The numbered `flow` shows model requests and responses, function starts and finishes, and the final outcome; `tool_counts` gives attempts, successes and failures for each function. Returned answer text includes a separate **Tools used** line with those counts. Short citation labels such as `[C1]` map to original chunk IDs in `citation_map`; label membership is checked, but semantic support still needs review. Tool failures are reported, and previously completed calls remain in a trace even if a later model call times out. Recorded outputs are redacted for connection strings and known credentials; see `results-and-privacy.md`.

Run from the repository root with Day 6 PostgreSQL, its Python environment and Ollama running. Connection settings are read privately from the existing Day 6 `.env` or the process environment; never paste their values into commands, results or Git:

```powershell
../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion/.venv/Scripts/python.exe src/function_agent.py --day6 ../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion --question "What tokenizer does T5 use?" --output tmp/example-function.json
../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion/.venv/Scripts/python.exe src/benchmark_more.py --day6 ../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion --output tmp/retrieval-dev
../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion/.venv/Scripts/python.exe src/function_agent_dev.py --day6 ../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion --output tmp/function-dev.json
```

## Dev retrieval comparison

Same 160 chunks and 15 answerable dev questions, original question as input, no filter, top-3 ranking, 4,000-character context. Exact quote in one matching source/page is the hit definition. `results/additional-methods/dev-traced/` contains ranked candidates, intermediate stages and per-query outcomes; gold is only used after retrieval for metrics. Errors remain in the denominator. Times are one warm run while a local agent was also active and exclude one-time model/index initialization; USD cost excludes hardware and electricity.

| Method | Hit@1 | Evidence Hit@3 | Context evidence | Mean ms | Errors |
|---|---:|---:|---:|---:|---:|
| Dense | 12/15 (80%) | 13/15 (86.7%) | 13/15 (86.7%) | 111.3 | 0 |
| TF-IDF | 11/15 (73.3%) | 12/15 (80%) | 12/15 (80%) | 2.3 | 0 |
| BM25 | 11/15 (73.3%) | 15/15 (100%) | 15/15 (100%) | 3.4 | 0 |
| Dense + BM25 RRF | 11/15 (73.3%) | 15/15 (100%) | 15/15 (100%) | 118.0 | 0 |
| Hybrid + coverage rerank | 11/15 (73.3%) | 15/15 (100%) | 15/15 (100%) | 120.6 | 0 |
| Two-stage expansion + fusion | 13/15 (86.7%) | 15/15 (100%) | 15/15 (100%) | 144.9 | 0 |

For two-stage retrieval, the original-query BM25 first stage scored 11/15 Hit@1 and 15/15 Hit@3; the fused pipeline scored 13/15 and 15/15. Expansion terms are taken from the top BM25 result, not from gold. q01 and q09 were dense top-3 misses but BM25 and the two-stage pipeline found their evidence. q11's evidence was not first for BM25 or the two-stage pipeline, though both returned it in top-3. Reranking did not improve aggregate Hit@3 over hybrid and can move relevant evidence down, such as q13. These are retrieval metrics, not proof of grounded answers; the small English dev set and exact-quote metric limit generalization.

The old test was already examined before this change. It has not been rerun for these methods, and none of the new methods is claimed as a newly selected or independently tested final configuration. A separate seven-case draft is in `data/benchmark-fresh-test-draft.json`; it was authored after the old test exposure and has not been run. Its independence, source evidence and two proposed absences need review, then the draft and dev-based selection must be frozen before any new test run. The original checkpoint 3/4 traces and selection remain historical and unchanged. Initial native-tool dev attempts and their errors/abstentions are kept under `results/additional-methods/`; no mock result is presented as agent evaluation.

## Native function-agent dev run

All 17 dev questions were run with `llama3.2:3b`; individual records are consolidated in `results/additional-methods/function-dev.json`, with manual AI-assisted labels and reasons in `function-dev-reviews.json`. Of the 15 runs with at least one successful retrieval, all 15 used a retrieval function: four made one call and eleven made two calls. Across all cases, the model attempted `search_dense` 11 times, `search_rerank` 3 times and `search_multistage` 14 times. Two attempts failed because the model supplied an out-of-range `top_k`; q07 and u01 had no successful retrieval. Their recorded raw status says abstain, but the review correctly classifies both as tool errors, not successful abstentions. Future runs explicitly return `error` when all retrieval calls fail.

Among 14 successfully retrieved answerable cases, grounded-answer grades 0/1/2 were 11/2/1, averaging 4/14 on the 0–2 scale. Citation support was 3/7; 3/15 successful answers contained citations. One of two missing-evidence cases was a correct abstention; the other had a tool error. Eleven of 15 answerable questions were refused, with one additional answerable case ending in a tool error. Mean end-to-end time across all 17 was 64.7 seconds; API service cost was 0 USD. Labels were made by Codex against the cited source, not independently adjudicated.

q01 correctly cited the normalization evidence; q02 included the correct scalar detail but also a contradictory claim; q18 included some supported explanations alongside unsupported claims. Thus native function calls and a valid label do not by themselves make an answer grounded. First-run traces were reconstructed into chronological `flow` entries from their recorded model responses and tool calls and explicitly marked as reconstructed; later traces record the flow directly. The consolidated result preserves the underlying actions and outputs.

The `function-flow-smoke.json` entry in `results/additional-methods/exploratory.json` is a later real dev smoke run of the direct chronological flow: it called `search_dense` once and cited page 4 for the correct normalization answer. It is an implementation check, not an independently scored benchmark run.
