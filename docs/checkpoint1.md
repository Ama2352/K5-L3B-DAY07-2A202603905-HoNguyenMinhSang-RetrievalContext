# Checkpoint 1: Day 6 handoff

The corpus is Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, JMLR 21(140), 2020, [PDF](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf), CC BY 4.0. Day 6 selected physical PDF pages 4–40 (37 pages); its `data/ATTRIBUTION.md` and `data/manifest.json` document attribution, normalization, limitations and hashes. Tables and equations in extracted text are approximate; consult the PDF for verification.

Day 6 selected strategy B: paragraph chunks of at most 1,000 characters, 160 chunks, original content-derived IDs and offsets. Its index used PostgreSQL 16.16 / pgvector 0.8.7, table `lab6_chunks`, exact cosine, and FastEmbed `BAAI/bge-small-en-v1.5` (384 dimensions). Queries require the Day 6 prefix `Represent this sentence for searching relevant passages: `; `MyEmbedder` supplies it. The T5 paper is the source, not the embedding model.

From this directory, reconstruct the chunk export and import the 15 previously used Day 6 dev questions:

```powershell
python src/prepare_day6.py --day6 ../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion
```

This creates `data/day6_chunks.json` and `data/day6_dev_answerable.json`. Reconstructing chunks does not prove that the live database contains the same vectors. The Day 6 environment and selected model must be available to query the actual store. Start the existing Day 6 Docker Compose service without deleting its volume, set `DATABASE_URL` locally (do not commit it), then use the Day 6 virtual environment and dependencies:

```powershell
../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion/.venv/Scripts/python.exe src/day6_search.py --day6 ../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion --query "What tokenizer does T5 use?"
```

The script checks the strategy B row count, embeds the query with the original model and searches the existing store without modifying it. On 2026-11-16, a real query succeeded, and all 160 database IDs, texts, metadata and 384-dimensional vector sizes matched the reconstructed snapshot. The saved Day 6 environment records the original model and model-file hashes. This checks metadata and dimensionality, not vector-by-vector equality against a separately saved embedding snapshot.

The imported questions are **dev only** and were already used in Day 6. Gold is kept for evaluation, never passed to the retriever or agent. The Day 6 test was already run and inspected; it is not an independent Day 7 test. The new split below was prepared before Day 7 experiments; do not tune on test.

`data/benchmark.json` contains 15 answerable dev questions (14 imported and one replacement), five newly authored answerable test questions, and two unanswerable questions per split. Each of the 20 gold excerpts matches a chunk. The old q11 crossed a chunk boundary; it was replaced with a different dev question rather than altering the scoring rule. The new test is separate from the old Day 6 test and must remain held out until configuration selection. Four absence cases are restricted to pages 4–40; searches of all normalized pages found none of the recorded energy, emissions and cost terms. Extraction errors or alternative wording remain a limitation, and a human should inspect the source when judging abstention.

`src/verify_day6_export.py` also checked all 45 recorded B dev hits against the reconstructed IDs, text and metadata (38 distinct chunks). No Day 7 retrieval benchmark or agent result is claimed here.
