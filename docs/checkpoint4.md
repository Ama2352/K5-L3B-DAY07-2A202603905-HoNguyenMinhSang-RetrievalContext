# Checkpoint 4: held-out test

The selected hybrid and dense baseline were run on all seven test questions after `results/checkpoint3/selection.json` was frozen. The original-query retrieval ranking and model-rewritten-query agent traces are stored separately in `results/checkpoint4/`. Per-question answer, citation and context grades and explanations are in `test_reviews.json`; reviews were performed by the Codex assistant and have not been independently adjudicated.

Dense answered one of five answerable questions correctly with supporting citation; hybrid answered none of four successful answerable questions correctly and timed out on the fifth. Dense abstained correctly on 0/2 missing-evidence questions; hybrid on 1/2. Several fluent answers cited unrelated text or invented values. Results, definitions, timing, tool use, error handling and examples are detailed in the [individual report](../report/individual/2A202603905_HoNguyenMinhSang.md). The sample is small and these are low-quality results, not a claim of production readiness.

The hybrid timeout happened after the initial model action; `run_agent` raised before returning its accumulated search events, so its pre-timeout context was not saved. The error, available model action and elapsed time are retained. It is excluded from successful-run quality metrics and reported separately, and is not counted as a successful context in the five-case denominator. A future run should persist each event as it happens; the held-out run was not repeated or tuned after inspecting results.

The optional advanced challenge is not complete. It requires a separate 100-page multimodal corpus and benchmark, full test evidence review and at least 90% test pass rate; these requirements are not satisfied by the base results.
