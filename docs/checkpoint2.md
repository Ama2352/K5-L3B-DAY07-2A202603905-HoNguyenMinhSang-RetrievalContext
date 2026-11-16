# Checkpoint 2: first real agent and retrieval traces

`src/run_agent.py` connects the starter loop to local Ollama and the Day 6 BGE/pgvector exact cosine store. The local model proposes a search or answer as JSON, with up to three searches, top-3 results per search and a 4,000-character context budget. Each trace records the question, candidate results, context, model actions, token counts, citations, duration and errors. Gold and expected answers are never passed to the model or retriever. Ollama and embedding run locally; service API cost is zero, excluding hardware and electricity.

Reproduce with Docker's existing Day 6 volume running and `DATABASE_URL` set locally:

```powershell
../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion/.venv/Scripts/python.exe src/run_agent.py --day6 ../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion --model llama3.2:3b --question "Approximately how large is the cleaned English C4 corpus described in the paper?" --output results/checkpoint2/example.json
```

The controller asks the model to use the search/answer schema; citation IDs supplied separately by the model are appended to the answer for the starter's citation-format check. This is formatting only: the check does not establish that the citation supports the claim. Source contents are treated as untrusted data.

Initial runs and their outcomes are retained under `results/checkpoint2/`: `dev-q01.json` failed because the model returned malformed answer data before searching; `dev-q01-qwen.json` repeatedly searched the same query and hit the search limit; `dev-q05.json` retrieved and answered but omitted inline citation syntax, so the starter abstained. After citation-format normalization, `dev-q05-normalized.json` returned **6.1 TB**, incorrectly. Its cited page 7 states approximately **750 GB** for the cleaned corpus; the nearby page 26 table shows **6.1 TB for unfiltered C4**. Thus the answer is unsupported by its citation even though that ID exists in context. This is a dev error to investigate in checkpoint 3, not a successful grounded answer.

The TF-IDF implementation in `starter/baseline.py` provides an initial lexical approach; it needs adaptation for the mixed answerable/unanswerable benchmark before comparison. These exploratory dev runs are not test metrics. Model and tool failures are not replaced with mock results.

Later, the answer output was extended with `tools_used`, built from recorded search events and the selected retriever. `results/tool-reporting-smoke.json` shows a real dense run with one pgvector search and its embedding component. The run's answer, “WordPiece tokenizer”, cites an unrelated page 40 chunk and is not a correct, supported answer to the tokenizer question; it validates tool display only, not answer quality. Existing benchmark traces were not modified or rerun. On errors, tool reporting marks the trace incomplete because the starter can lose earlier events when a later action raises.
