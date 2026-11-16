# Optional advanced challenge: preparation, not complete

Candidate source: NASA, *FY 2024 Agency Financial Report*, 127 physical PDF pages, [original](https://www.nasa.gov/wp-content/uploads/2023/11/nasa-fy-2024-afr-1.pdf), SHA-256 `65ee29adf5f3a92fcd74701696c25dae8f2d18f1ba2e7f4fba2bcfc059acbefc`. NASA's [media guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/) permit educational use of NASA material subject to conditions and note that third-party copyrighted material and NASA identifiers have separate restrictions. Rights of individual included images have not been fully audited. The PDF and derived image assets are ignored by Git; obtain the source independently and check rights before sharing them.

`src/advanced_ingest.py` extracts text and an inventory of embedded images linked to physical pages. For this snapshot: 127/127 pages have extracted text (341,302 characters), and 97 embedded images appear on 55 pages. Some JPEG 2000 image masks have mismatched sizes and require visual checking. Extraction of embedded images does not capture vector-drawn charts. `src/advanced_render.py` rendered 127/127 pages with PyMuPDF 1.26.7 at 110 dpi, recorded text/image block bounding boxes and vector-path counts, and reported zero page-render errors. Page 32's chart and page 44's prose were visually checked; this is not a full audit of the report. `src/advanced_text_index.py` created 448 real BGE-small-en-v1.5 text vectors, with a search smoke test. The generated files, including full-page PNGs, are local and ignored.

From the repo, after obtaining the exact PDF as `data/advanced/nasa-fy-2024-afr.pdf`, with the Day 6 venv and model cache available:

```powershell
../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion/.venv/Scripts/python.exe src/advanced_ingest.py --pdf data/advanced/nasa-fy-2024-afr.pdf
# Install PyMuPDF 1.26.7 into data/advanced/generated/vendor, then:
../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion/.venv/Scripts/python.exe src/advanced_render.py --pdf data/advanced/nasa-fy-2024-afr.pdf
../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion/.venv/Scripts/python.exe src/advanced_text_index.py --day6 ../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion
```

`data/advanced/benchmark-dev-draft.json` contains 15 answerable dev examples covering charts, tables, photographs, text and two multi-hop calculations, plus three provisional missing-evidence examples. `benchmark-test-draft.json` contains 30 separate test drafts: six tables, five images, five charts, four multi-hop, four text and six missing-evidence cases. The answerable evidence was checked against rendered original pages. Absence checks cover extracted text but not every visual region; the questions, independence and absences still require full review. Neither draft is frozen or has been run.

The PDF's extractable text includes repeated editorial text about a chart color option that is not visible on some rendered pages. This demonstrates that raw extraction can include hidden or non-content text; do not treat it as verified evidence. The full index requires a documented approach to filtering such material and checking chart/image interpretations against rendered originals.

`src/advanced_agent.py` is an untested adapter for a vision-capable Ollama model: it asks the model for a query, retrieves distinct pages via the text index, and supplies the rendered page images and extracted snippets for a cited answer. It never reads benchmark gold. This would allow visual inspection of retrieved pages, but initial retrieval is text-only and can miss useful visuals without matching text; the adapter has not been validated with a vision model. It logs page IDs, snippets, image paths, model actions, timing and errors. Valid page IDs still require semantic review.

This is **not yet a complete multimodal index**: extracted text, page images and geometry do not by themselves interpret every diagram, chart or photograph. Full visual-content coverage and source-region verification are incomplete, as are benchmark review, validated visual-agent runs, test traces and per-case scoring. The local Ollama inventory currently exposes only text models (`llama3.2:3b` and `qwen2.5:1.5b` variants); no vision-capable model is available. The base agent's poor test score also does not support claiming the 90% advanced threshold. No bonus score is claimed.
