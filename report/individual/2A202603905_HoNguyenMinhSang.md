# Báo cáo Day 7 — Hồ Nguyễn Minh Sang, 2A202603905

## Dữ liệu và benchmark

Đây là báo cáo đợt **chạy lại thật** bằng native function agent, thay cho các kết quả cũ đã gộp. Dev bắt đầu 2026-11-16T07:47:17.113001+00:00; test bắt đầu 2026-11-16T08:16:12.131345+00:00. Có đủ 51 lượt agent dev, 14 lượt test và 65 nhãn đọc đối chiếu; retrieval có 45 bản ghi dev và 10 bản ghi test. Kết quả sai, lỗi hệ thống và raw answer đều được giữ, không dùng mock làm benchmark.

Corpus Day 6: Raffel và cộng sự, *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, JMLR 21(140), 2020, [nguồn PDF](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf), CC BY 4.0. Chọn trang PDF vật lý 4–40 (37 trang), 160 chunk theo đoạn tối đa 1.000 ký tự; giữ ID/nguồn/trang/offset. Vector store PostgreSQL/pgvector Day 6, exact cosine, FastEmbed `BAAI/bge-small-en-v1.5`, 384 chiều và prefix query tương thích. Bảng/công thức có thể mất định dạng; bản gốc là nguồn kiểm chứng.

`data/benchmark.json` cung cấp dev 15 câu có đáp án + 2 thiếu bằng chứng; các câu test lịch sử trong tệp này không dùng trong đợt đánh giá mới. Test thực tế dùng riêng `data/benchmark-test.json`: 5 câu có đáp án + 2 thiếu bằng chứng. Tệp `data/benchmark-validation.json` ghi kiểm tra nguồn, hash PDF, phạm vi trang và tính độc lập của test mới. Dev/gold chỉ được dùng để đánh giá, không truyền vào agent/retrieval. Cấu hình được chốt tại 2026-11-16T08:16:01.471370+00:00 trước khi chạy test; không chỉnh pipeline từ kết quả test.

## Retrieval và thử nghiệm

Ba phương pháp: dense trên vector store Day 6; lexical BM25 k1=1,5/b=0,75; hybrid dense+BM25 bằng RRF k=60 (20 candidate mỗi nguồn). Không metadata filter. Dense/BM25 dùng top-3 và 4.000 ký tự context; hybrid top-5 và 6.000 ký tự. Ranking query gốc và trace query model viết lại được lưu riêng trong cùng tệp để tránh trộn hai loại kết quả.

| Cấu hình/split | Hit@1 | Evidence Hit@3 | Context Evidence Rate query gốc | Retrieval trung bình | Lỗi |
|---|---|---|---|---|---|
| dense_baseline/dev | 9/15 (60.00%) | 10/15 (66.67%) | 10/15 (66.67%) | 1305.82 ms | 4 |
| bm25/dev | 11/15 (73.33%) | 15/15 (100.00%) | 15/15 (100.00%) | 4.74 ms | 0 |
| hybrid/dev | 11/15 (73.33%) | 15/15 (100.00%) | 15/15 (100.00%) | 60.26 ms | 0 |
| dense_baseline/test | 1/5 (20.00%) | 2/5 (40.00%) | 2/5 (40.00%) | 68.42 ms | 0 |
| bm25/test | 3/5 (60.00%) | 4/5 (80.00%) | 4/5 (80.00%) | 1.47 ms | 0 |

Hit yêu cầu quote chuẩn xuất hiện trong chunk đúng nguồn/trang, top-1 hoặc top-3 theo cùng cutoff. Context Evidence Rate query gốc yêu cầu evidence có trong context đã đóng gói. Đoạn thay thế hợp lệ có thể bị metric exact-quote bỏ sót; nhãn context thủ công đánh giá thêm bằng chứng tương đương. Câu thiếu evidence không vào mẫu số retrieval. Các context được khử trùng ID và đóng gói nguyên đoạn trong budget.

Hai giả thuyết cải thiện được công bố trong cấu hình trước chạy:

1. **BM25 thay dense:** truy vấn chứa thuật ngữ kỹ thuật có thể hưởng lợi từ lexical matching. Dev Evidence Hit@3 tăng từ 10/15 lên 15/15 và grounded tăng 6/30 → 11/30. Tuy nhiên dense có 4 lỗi database, nên mức chênh không thể quy hoàn toàn cho thuật toán.
2. **Hybrid + context lớn:** kết hợp lexical/semantic và giữ thêm candidate có thể tăng evidence. Query gốc dev Hit@3 đạt 15/15 nhưng grounded chỉ 7/30, thấp hơn BM25. Thử nghiệm thay cả fusion và budget; không tách riêng tác động của từng thay đổi.

Phân tích lỗi trước/sau trên dev: q01 dense có evidence đúng ở C3 nhưng trỏ C1 không hỗ trợ; BM25 đưa đoạn đúng lên C1 và tăng grounded từ 0 lên 1, vẫn diễn giải sai gamma/beta. q15 dense từ chối (0), BM25 trả đúng span length 3 (2), hybrid lại từ chối dù context đủ (0). q05 mọi context đều có khoảng 750 GB, nhưng BM25/hybrid bịa số lượng tài liệu; tăng retrieval chưa sửa được lỗi sinh câu trả lời. Đây là các lỗi đã đọc, không phải chỉ suy từ metric.

## Agent và đánh giá

Model thực tế `qwen2.5:1.5b`, temperature 0, num_predict 256, num_ctx 4096. Mỗi cấu hình chỉ cung cấp schema function tương ứng (`search_dense`, `search_bm25`, `search_hybrid`); model chọn query và gọi function thật. Tối đa 3 tool call, có điều kiện dừng và một lượt nhắc tìm khi chưa truy xuất. Source/tool output được coi là dữ liệu, không phải chỉ dẫn. Từ chối khi thiếu evidence/citation hợp lệ; kiểm tra cú pháp ID không thay thế việc đọc citation.

Tiêu chí lựa chọn: tổng grounded dev (lỗi tính 0), tiếp theo số abstention đúng, tổng điểm context, rồi tổng thời gian. Cấu hình chọn **bm25**, được khóa cùng hash code/test/dev tại `results/basic/selection.json`. Grounded 2 yêu cầu trả lời đầy đủ và citation hỗ trợ mọi khẳng định chính; đúng một phần hoặc dẫn nhầm label chấm 1; sai/thiếu evidence/từ chối câu có đáp án chấm 0. Abstention được đọc theo nội dung, không chỉ dựa runtime status. Nhãn do Codex hỗ trợ đọc bản gốc, chưa có người chấm độc lập.

| Cấu hình/split | Tool use | Grounded 0/1/2; trung bình /2 | Citation hỗ trợ | Câu có citation | Abstention | Từ chối nhầm | Thời gian TB | Lỗi |
|---|---|---|---|---|---|---|---|---|
| dense_baseline/dev | 13/13 (100.00%) | 11/2/2; 0.400 | 2/7 (28.57%) | 7/17 (41.18%) | 1/2 (50.00%) | 5/15 (33.33%) | 32.09 s | 4 |
| bm25/dev | 17/17 (100.00%) | 8/3/4; 0.733 | 6/10 (60.00%) | 10/17 (58.82%) | 2/2 (100.00%) | 5/15 (33.33%) | 24.96 s | 0 |
| hybrid/dev | 16/17 (94.12%) | 10/3/2; 0.467 | 4/12 (33.33%) | 12/17 (70.59%) | 1/2 (50.00%) | 4/15 (26.67%) | 29.23 s | 0 |
| dense_baseline/test | 7/7 (100.00%) | 5/0/0; 0.000 | 1/7 (14.29%) | 5/7 (71.43%) | 0/2 (0.00%) | 2/5 (40.00%) | 23.77 s | 0 |
| bm25/test | 7/7 (100.00%) | 4/0/1; 0.400 | 2/4 (50.00%) | 4/7 (57.14%) | 1/2 (50.00%) | 2/5 (40.00%) | 34.59 s | 0 |

Đã đọc đủ 5 context có đáp án của mỗi cấu hình test, không còn trường hợp mất context như đợt cũ:

- dense_baseline: mức 0/1/2 = 2/0/3; đủ evidence 3/5 (60.00%).
- bm25: mức 0/1/2 = 1/0/4; đủ evidence 4/5 (80.00%).

Mọi lượt test đều có một retrieval function thành công. Dev dense có 4 timeout kết nối database lúc RAM máy bị áp lực bởi model vision; giữ trong mẫu số và báo riêng, không thay bằng kết quả giả. Thời gian là một lần chạy, chịu ảnh hưởng tải máy, không bao gồm khởi tạo embedding/index. API local 0 USD; chưa tính điện, phần cứng hay công chuẩn bị/đối chiếu corpus bằng trợ lý.

Ba ví dụ trình bày: BM25 test02 trả ReCoRD Exact Match 68,16 → 73,72, chênh 5,56 điểm, cite C1 trang PDF 27 đầy đủ. BM25 test07 cuối cùng từ chối yêu cầu tọa độ data center, nhưng raw answer đã bịa tọa độ New York rồi bị chặn do thiếu citation; cần giữ giới hạn này. BM25 test05 nhầm Small T5 thành 12/12 layer và khoảng 100 triệu tham số; bản gốc trang 36 ghi 6/6 layer và khoảng 60 triệu; context thiếu thông số kiến trúc. Test06 cả hai trả đoạn liên quan nhưng không từ chối rõ yêu cầu serial number, nên abstention không đạt.

`results/basic/dev.json` và `test.json` chứa ranking, query gốc, các tool call thật, context, answer/raw answer, citation, thời gian, lỗi và nhãn từng câu. Function trace có `name`, `call_id`, `call_number`, arguments, results, `tool_call_count`, `tool_counts`, flow và số request model. `selection.json` lưu quyết định trước test. Không tạo một JSON cho mỗi câu, không trình bày kết quả gộp lịch sử thành lượt chạy mới.

Credential chỉ đọc riêng từ environment hoặc `.env` Day 6; không có giá trị secret trong lệnh mẫu. `src/common/submission_audit.py` chặn tệp env, kể cả example, token/private key, URL có mật khẩu và credential gán literal; kết quả chỉ ghi metadata, không in secret. Audit working tree và reachable history đã thực hiện; cần chạy lại trước nộp. Không nộp `tmp/`, cache/model, PDF/ảnh sinh hoặc bản backup.

Chạy từ gốc repo bằng PowerShell. Giữ môi trường Day 6 và volume pgvector cũ; không thay embedding model/index bằng vector ngẫu nhiên.

```powershell
$day6 = '../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion'
$py = Join-Path $day6 '.venv/Scripts/python.exe'
& $py -m pip install -r (Join-Path $day6 'starter/requirements.lock.txt')
docker compose -f (Join-Path $day6 'starter/docker-compose.yml') up -d
& $py -m src.data_pipeline.day6_search --day6 $day6 --query 'What tokenizer does T5 use?'
& $py -m unittest discover -s tests -v
& $py -m src.common.submission_audit --output tmp/security-audit.json
```

Ollama cần đang chạy; tải model bằng `ollama pull qwen2.5:1.5b`. Tái lập vào thư mục riêng để không ghi đè bằng chứng đã chấm:

```powershell
& $py -m src.evaluation.evaluate_lab --day6 $day6 --split dev --output-root results/reproduction/basic
# Chấm từng câu theo schema reviews trong kết quả gốc trước khi chốt.
& $py -m src.evaluation.summarize_lab --split dev --reviews <file-nhan-dev> --freeze --output-root results/reproduction/basic
& $py -m src.evaluation.evaluate_lab --day6 $day6 --split test --output-root results/reproduction/basic
& $py -m src.evaluation.summarize_lab --split test --reviews <file-nhan-test> --output-root results/reproduction/basic
```

Không dùng test đã xem để chỉnh pipeline rồi gọi là đánh giá độc lập. Nếu thay pipeline bằng thông tin từ test, tạo test mới. Thư mục tái lập được Git bỏ qua.

`src/agents/`: agent; `src/retrieval/`: functions/fusion; `src/evaluation/`: benchmark/metrics; `src/data_pipeline/`: Day 6; `src/advanced/`: PDF/catalog/nâng cao; `src/common/`: bảo mật/trace. README và các tệp đề bài được giữ đúng commit gốc; báo cáo cá nhân này là tài liệu nộp bằng tiếng Việt.

## Thử thách nâng cao (nếu thực hiện)

NASA FY2024 Agency Financial Report, 127 trang, [nguồn](https://www.nasa.gov/wp-content/uploads/2023/11/nasa-fy-2024-afr-1.pdf), SHA-256 `65ee29adf5f3a92fcd74701696c25dae8f2d18f1ba2e7f4fba2bcfc059acbefc`. Không rehost PDF/ảnh khi chưa rà hết quyền bên thứ ba; giữ nguồn và cách tái tạo. Các trang được index toàn bộ text/layout, bổ sung catalog trực quan source-only được đối chiếu từ ảnh gốc, không dùng gold để xây index. Chưa tuyên bố đạt bonus cho tới khi hoàn tất và chấm toàn bộ test đạt ít nhất 90%. Kết quả mới sẽ ở `results/advanced/`; các số liệu Advanced cũ không phải kết quả của pipeline mới.

Dev có 18 câu (15 có đáp án, 3 thiếu evidence). Test độc lập có 30 câu: 6 bảng, 5 ảnh, 5 biểu đồ, 4 multi-hop, 4 text và 6 thiếu evidence. Đạt bonus cần ít nhất 27/30 câu test theo chấm nghiêm ngặt; kết quả dev hoặc kiểm tra tự động không thay thế ngưỡng này. Không chỉnh pipeline bằng đáp án test.

Benchmark được nộp tại `data/advanced/benchmark-dev.json` và `data/advanced/benchmark-test.json`. Việc bỏ hậu tố `-draft` chỉ đổi tên, không đổi câu hỏi/gold; SHA-256 test giữ nguyên `7441e969e4c03c11c3dcc0a12c82f21b974f9d5da2df4c5645b51d4e7cad848b` trước khi chốt pipeline.

`data/advanced/visual-catalog.json` là chú thích toàn bộ 127 trang được trợ lý hỗ trợ đối chiếu ảnh nguồn, không phải kết quả một lượt OCR/vision Ollama đầy đủ. Catalog giữ mô tả vùng, nhãn và giới hạn kiểm chứng; không khẳng định đã gõ lại độc lập mọi ô bảng. Dense BGE 384 chiều kết hợp BM25 và RRF truy xuất text/mô tả trực quan. Model trả lời local đọc catalog và text; ảnh gốc là nguồn kiểm chứng ngoài luồng, không được gửi cho text model. `results/advanced/index-manifest.json` ghi thống kê, hash và liên kết từng trang.

Đợt dev tiếp theo dùng Ollama `qwen3:4b-instruct`, context 4.096 token, tối đa 384 token sinh/lượt, budget evidence 8.000 ký tự và tối đa 5 tool call. Ba function: `search_evidence` (top-k từ 1 đến 8), `inspect_page`, `calculate`. Các thay đổi dựa trên lỗi dev: yêu cầu top-k 5 từng bị từ chối, nhầm loại hàng tổng và từ chối dù đã có thông tin được hỏi. Cấu hình chỉ được chốt sau khi đọc đủ dev; chưa có kết quả test của cấu hình này. Theo lựa chọn hiện tại của chủ repo, chỉ chạy inference Ollama local, không dùng Groq hoặc Codex CLI.

Context giữ từng vùng trực quan cùng tiêu đề, đơn vị, chú giải và nhãn thay vì cắt các dòng JSON riêng lẻ. Các đoạn text gốc giữ hàng lân cận; khi inspect trang, giữ toàn bộ text gốc theo thứ tự. Nếu query của model nêu rõ duy nhất một loại nguồn (biểu đồ, ảnh chụp, sơ đồ hoặc bảng), retriever ưu tiên metadata loại vùng tương ứng trong tập ứng viên RRF và giữ nguồn khác làm fallback; query yêu cầu nhiều loại giữ fusion thông thường. Suy luận này chỉ dùng query và catalog nguồn, không dùng nhóm/gold benchmark. Trace ghi `query_modality_preference` và nguồn suy luận. Ranking trung gian giữ ID/rank/score/nguồn/trang; snippet đã chọn, context model thực nhận và function call vẫn được lưu đầy đủ, tránh lặp nguyên 40 trang trong mỗi trace.

Lượt dev đầy đủ đầu tiên đạt 16/18 (88,89%), không có lỗi hệ thống; tổng thời gian các trace 2.258,28 giây, 47 lượt model và 18 lượt search. Hai lỗi dùng cùng nguồn Note 14: context giữ prose làm tròn 12 tỷ USD nhưng bỏ hàng Total Undelivered Orders 12.214 triệu USD. Câu tra bảng vì vậy thiếu độ chính xác; câu multi-hop funding trừ undelivered orders đã từ chối. Ba câu thiếu bằng chứng đều từ chối đúng. Kết quả này chưa đủ để công bố đạt Advanced.

Thay đổi tiếp theo trên dev: loại từ chức năng khỏi chấm relevance excerpt và cộng trọng số 3 cho term khớp tiêu đề vùng nguồn, giữ bảng cùng năm/đơn vị/nhãn số trước prose. Thay đổi áp dụng chung cho mọi trang và query, không chứa đáp án/ID benchmark. Kiểm tra hồi quy dùng bảng giả lập với số liệu khác benchmark để xác nhận bảng chính xác sống sót khi prose làm tròn dài lấn budget. Cấu hình mới phải chạy lại dev trước khi chốt và mở test.

Đã chạy lại đầy đủ và đối chiếu từng câu của cấu hình mới: **dev 18/18 (100%)**, gồm bảng 4/4, ảnh 3/3, biểu đồ 3/3, text 3/3, multi-hop 2/2 và thiếu bằng chứng 3/3. Không có lỗi hệ thống; 18 lượt `search_evidence`, 46 lượt model, tổng thời gian trace 2.198,08 giây. Hai câu lỗi trước sửa nay trả đúng 12.214 và 17.837 triệu USD với nguồn cần thiết. So sánh trước/sau và hai trace lỗi cũ được lưu gọn trong `development_experiments` của `results/advanced/dev.json`, không trộn vào mẫu số lượt mới.

Pipeline được khóa lúc **2026-11-16 15:40:12 UTC** trong `results/advanced/selection.json`, trước khi chạy toàn bộ test. Hash kết quả dev là `a701c3f7465a64c9fd6aab526751231a88007b608968aa8ddc88fc635ebbcb34`; source code được ghi nhận ở commit `f6d50b5`. Các nhãn dev có hash từng trace mới để đối chiếu. Test đang chạy, chưa công bố tỷ lệ đạt hoặc bonus. Thời gian tổng là tổng duration của các trace hoàn tất, không gồm thời gian chủ repo tạm dừng hay probe dev bị bỏ.

Chi phí inference của đợt Ollama local là 0 USD phí API, chưa tính điện/phần cứng. Trước đợt này đã có thử nghiệm Codex CLI bị dừng theo yêu cầu chủ repo: 7 câu dev và 2 probe đã có số usage, tổng 537.052 input token (trong đó 372.224 cached) và 1.845 output token; usage của lượt bị hủy dở có thể chưa được ghi đầy đủ. Chưa xác minh khoản tiền thực tế, nên không tuyên bố các thử nghiệm đó miễn phí. Chúng không thuộc kết quả benchmark local đang nộp; không chạy thêm Codex CLI hoặc Groq.

Tái tạo Advanced từ nguồn công khai và catalog đã nộp (chạy từ gốc repo; tải PDF chỉ vào đường dẫn được Git bỏ qua):

```powershell
$day6 = '../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion'
$py = Join-Path $day6 '.venv/Scripts/python.exe'
$pdf = 'data/advanced/nasa-fy-2024-afr.pdf'
Invoke-WebRequest -Uri 'https://www.nasa.gov/wp-content/uploads/2023/11/nasa-fy-2024-afr-1.pdf' -OutFile $pdf
& $py -m pip install --target data/advanced/generated/vendor PyMuPDF==1.26.7
& $py -m src.advanced.advanced_render --pdf $pdf --dpi 110
& $py -m src.advanced.advanced_layout_index --day6 $day6 --pdf $pdf
& $py -m src.advanced.build_verified_catalog --pdf $pdf
ollama pull qwen3:4b-instruct
& $py -m src.advanced.evaluate_multimodal --day6 $day6 --split dev --output-root results/reproduction/advanced
# Đối chiếu từng câu với PDF gốc, lập nhãn theo schema reviews trong kết quả.
& $py -m src.advanced.score_results --split dev --reviews <file-nhan-dev> --output-root results/reproduction/advanced
& $py -m src.advanced.evaluate_multimodal --freeze --output-root results/reproduction/advanced
& $py -m src.advanced.evaluate_multimodal --day6 $day6 --split test --output-root results/reproduction/advanced
& $py -m src.advanced.score_results --split test --reviews <file-nhan-test> --output-root results/reproduction/advanced
```

Constructor của retriever tạo vector thật khi chưa có `multimodal_vectors.json`; không dùng vector ngẫu nhiên. Catalog/index/model/code/benchmark được ghi hash trong kết quả và kiểm tra khi resume hoặc chạy test. Kết quả test, nhãn và tỷ lệ theo nhóm chỉ được công bố sau khi chạy và đối chiếu đủ toàn bộ 30 câu; không loại lỗi hay timeout khỏi mẫu số.
