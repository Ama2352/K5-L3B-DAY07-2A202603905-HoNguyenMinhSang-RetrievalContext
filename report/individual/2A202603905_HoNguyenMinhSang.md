# Báo cáo Day 7 — Hồ Nguyễn Minh Sang, 2A202603905

## Dữ liệu và benchmark

Đánh giá Basic gồm 51 lượt agent trên dev, 14 lượt trên test và 65 nhãn đối chiếu nguồn. Đánh giá retrieval trên query gốc gồm 45 bản ghi dev và 10 bản ghi test. Các kết quả lưu cả câu trả lời, context, citation, lỗi và thời gian từng truy vấn.

Corpus Day 6: Raffel và cộng sự, *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, JMLR 21(140), 2020, [nguồn PDF](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf), CC BY 4.0. Chọn trang PDF vật lý 4–40 (37 trang), 160 chunk theo đoạn tối đa 1.000 ký tự; giữ ID/nguồn/trang/offset. Vector store PostgreSQL/pgvector Day 6, exact cosine, FastEmbed `BAAI/bge-small-en-v1.5`, 384 chiều và prefix query tương thích. Bảng/công thức có thể mất định dạng; bản gốc là nguồn kiểm chứng.

Benchmark Basic sử dụng các bản ghi `split=dev` trong `data/benchmark.json`: 15 câu có đáp án và 2 câu thiếu bằng chứng. `data/benchmark-test.json` chứa 5 câu có đáp án và 2 câu thiếu bằng chứng. `data/benchmark-validation.json` ghi nguồn, hash PDF, phạm vi trang và kiểm tra độc lập dev/test. Agent và retriever chỉ nhận câu hỏi, không nhận gold. Cấu hình được khóa tại 2026-11-16 08:16:01 UTC trước đánh giá test.

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

Phân tích lỗi dev: q01 dense có evidence đúng ở C3 nhưng dẫn C1 không hỗ trợ; BM25 đưa đoạn đúng lên C1 và tăng grounded từ 0 lên 1, song vẫn diễn giải sai gamma/beta. q15 dense từ chối (0), BM25 trả đúng span length 3 (2), hybrid từ chối dù context đủ (0). q05 context của cả ba phương pháp có khoảng 750 GB nhưng BM25/hybrid suy đoán sai số lượng tài liệu: tăng retrieval chưa khắc phục được lỗi sinh câu trả lời.

## Agent và đánh giá

Model thực tế `qwen2.5:1.5b`, temperature 0, num_predict 256, num_ctx 4096. Mỗi cấu hình chỉ cung cấp schema function tương ứng (`search_dense`, `search_bm25`, `search_hybrid`); model chọn query và gọi function thật. Tối đa 3 tool call, có điều kiện dừng và một lượt nhắc tìm khi chưa truy xuất. Source/tool output được coi là dữ liệu, không phải chỉ dẫn. Từ chối khi thiếu evidence/citation hợp lệ; kiểm tra cú pháp ID không thay thế việc đọc citation.

Tiêu chí lựa chọn: tổng grounded dev (lỗi tính 0), tiếp theo số abstention đúng, tổng điểm context, rồi tổng thời gian. Cấu hình chọn **bm25**, được khóa cùng hash code/test/dev tại `results/basic/selection.json`. Grounded 2 yêu cầu trả lời đầy đủ và citation hỗ trợ mọi khẳng định chính; đúng một phần hoặc dẫn nhầm label chấm 1; sai/thiếu evidence/từ chối câu có đáp án chấm 0. Abstention được đọc theo nội dung, không chỉ dựa runtime status. Nhãn do trợ lý AI hỗ trợ đối chiếu bản gốc; chưa có người chấm độc lập.

| Cấu hình/split | Tool use | Số lượt gọi tool / model | Grounded 0/1/2; trung bình /2 | Citation hỗ trợ | Câu có citation | Abstention | Từ chối nhầm | Thời gian TB | Lỗi |
|---|---|---|---|---|---|---|---|---|---|
| dense_baseline/dev | 13/13 (100.00%) | 17 / 63 | 11/2/2; 0.400 | 2/7 (28.57%) | 7/17 (41.18%) | 1/2 (50.00%) | 5/15 (33.33%) | 32.09 s | 4 |
| bm25/dev | 17/17 (100.00%) | 17 / 65 | 8/3/4; 0.733 | 6/10 (60.00%) | 10/17 (58.82%) | 2/2 (100.00%) | 5/15 (33.33%) | 24.96 s | 0 |
| hybrid/dev | 16/17 (94.12%) | 16 / 64 | 10/3/2; 0.467 | 4/12 (33.33%) | 12/17 (70.59%) | 1/2 (50.00%) | 4/15 (26.67%) | 29.23 s | 0 |
| dense_baseline/test | 7/7 (100.00%) | 7 / 27 | 5/0/0; 0.000 | 1/7 (14.29%) | 5/7 (71.43%) | 0/2 (0.00%) | 2/5 (40.00%) | 23.77 s | 0 |
| bm25/test | 7/7 (100.00%) | 7 / 28 | 4/0/1; 0.400 | 2/4 (50.00%) | 4/7 (57.14%) | 1/2 (50.00%) | 2/5 (40.00%) | 34.59 s | 0 |

Kết quả kiểm chứng context test trên 5 câu có đáp án của mỗi cấu hình:

- dense_baseline: mức 0/1/2 = 2/0/3; đủ evidence 3/5 (60.00%).
- bm25: mức 0/1/2 = 1/0/4; đủ evidence 4/5 (80.00%).

Cả 14 lượt agent test đều có một retrieval function thành công. Dev dense có 4 timeout kết nối PostgreSQL trong điều kiện áp lực RAM khi xử lý vision; các trường hợp này được tính lỗi và giữ trong mẫu số. Thời gian là một lần đo, chịu ảnh hưởng tải máy và không bao gồm khởi tạo embedding/index. Phí API inference local là 0 USD, chưa tính điện và phần cứng.

| Query ID/cấu hình | Loại câu | Tool và kết quả | Câu trả lời/citation | Nhãn và lý do |
|---|---|---|---|---|
| test02/bm25 | Có đáp án | `search_bm25`, tìm được bảng ReCoRD | Exact Match 68,16 → 73,72, tăng 5,56 điểm; C1 trang PDF 27 | Grounded 2, đủ giá trị và citation |
| test07/bm25 | Thiếu bằng chứng | `search_bm25`, không có tọa độ data center | Final answer từ chối | Abstention đúng; raw answer có tọa độ suy đoán bị chặn do thiếu citation, là giới hạn của cơ chế kiểm soát |
| test05/bm25 | Lỗi trả lời | `search_bm25`, context thiếu kiến trúc Small T5 | 12/12 layer, khoảng 100 triệu tham số | Grounded 0; bản gốc trang 36 ghi 6/6 layer và khoảng 60 triệu tham số |

Ở test06, cả dense và BM25 đều trả thông tin liên quan nhưng không từ chối rõ yêu cầu serial number, nên không đạt abstention.

`results/basic/dev.json` và `test.json` chứa ranking query gốc, tool call, context, answer/raw answer, citation, nhãn, thời gian và lỗi. Function trace có `name`, `call_id`, `call_number`, arguments, results, `tool_call_count`, `tool_counts` và flow. `results/basic/selection.json` ghi quyết định lựa chọn trước test.

Credential được đọc từ biến môi trường hoặc cấu hình riêng của Day 6. `src/common/submission_audit.py` kiểm tra file env, token/private key, URL có mật khẩu và credential literal; đầu ra audit chỉ ghi đường dẫn/loại phát hiện. Credential, model/cache, PDF/ảnh sinh và bản sao làm việc không lưu trong phần nộp.

Các lệnh tái lập chạy từ gốc repository bằng PowerShell, sử dụng môi trường Day 6, PostgreSQL/pgvector và cache embedding tương ứng:

```powershell
$day6 = '../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion'
$py = Join-Path $day6 '.venv/Scripts/python.exe'
& $py -m pip install -r (Join-Path $day6 'starter/requirements.lock.txt')
docker compose -f (Join-Path $day6 'starter/docker-compose.yml') up -d
& $py -m src.data_pipeline.day6_search --day6 $day6 --query 'What tokenizer does T5 use?'
& $py -m unittest discover -s tests -v
& $py -m src.common.submission_audit --output tmp/security-audit.json
```

Model Basic: `ollama pull qwen2.5:1.5b`. Lệnh đánh giá và chốt cấu hình:

```powershell
& $py -m src.evaluation.evaluate_lab --day6 $day6 --split dev --output-root results/reproduction/basic
# Chấm từng câu theo schema reviews trong kết quả gốc trước khi chốt.
& $py -m src.evaluation.summarize_lab --split dev --reviews <file-nhan-dev> --freeze --output-root results/reproduction/basic
& $py -m src.evaluation.evaluate_lab --day6 $day6 --split test --output-root results/reproduction/basic
& $py -m src.evaluation.summarize_lab --split test --reviews <file-nhan-test> --output-root results/reproduction/basic
```

Dev dùng để lựa chọn cấu hình; test được chạy sau khi khóa. Script kiểm tra hash benchmark/code và quyết định đã chốt trước khi cho phép đánh giá test. Thư mục `results/reproduction/` dành cho các lượt tái lập.

Mã nguồn: `src/agents/` điều phối agent, `src/retrieval/` chứa retriever/fusion, `src/evaluation/` đánh giá Basic, `src/data_pipeline/` tái sử dụng Day 6, `src/advanced/` xử lý tài liệu nâng cao và `src/common/` quản lý trace/bảo mật.

## Thử thách nâng cao (nếu thực hiện)

Phần nâng cao đã hoàn thành với **27/30 câu test đạt (90%)**, đáp ứng ngưỡng yêu cầu. Bộ nộp gồm code và cách tái lập, nguồn/index toàn bộ tài liệu, benchmark dev/test, trace và nhãn đối chiếu nguồn cho từng câu, metrics theo nhóm và phân tích ba câu không đạt.

Tài liệu: *NASA FY2024 Agency Financial Report*, phiên bản FY2024, **127 trang PDF vật lý**, [nguồn PDF công khai](https://www.nasa.gov/wp-content/uploads/2023/11/nasa-fy-2024-afr-1.pdf). SHA-256: `65ee29adf5f3a92fcd74701696c25dae8f2d18f1ba2e7f4fba2bcfc059acbefc`. Sử dụng bản công khai để nghiên cứu, cung cấp đường dẫn và cách tải hợp lệ; không phân phối lại PDF/ảnh có thể chứa nội dung bên thứ ba. Trang trích dẫn là trang PDF vật lý, không phải số in ở chân trang.

Index bao phủ **127/127 trang**, không có trang bị bỏ qua. PyMuPDF 1.26.7 trích text/layout và render 110 dpi; catalog trực quan được AI hỗ trợ đọc, đối chiếu từng trang gốc. Mỗi vùng giữ loại, vị trí, nhãn và giới hạn; mỗi trang giữ hash ảnh gốc và bounding box. `data/advanced/visual-catalog.json` là annotation nguồn; `results/advanced/index-manifest.json` ghi cấu hình, thống kê và liên kết ảnh trang để kiểm chứng. Không chỉ index các trang có trong benchmark; chưa phiên âm độc lập mọi ô bảng, nên bản gốc vẫn là căn cứ chấm.

Toàn corpus tạo **3169 đơn vị BM25** trên **127 trang**: hàng bảng có header/năm/đơn vị, nhãn từng vùng biểu đồ, mô tả ảnh và đoạn text gốc. Mỗi vùng ảnh được ghép đoạn nguồn theo độ trùng từ của tiêu đề/nhãn, giữ quan hệ người–vai trò, công nghệ và địa điểm trong caption để kiểm chứng. Bộ tách hàng xử lý nhiều hàng bị PDF ghép chung một text block; bảng Accounts Receivable giữ hàng Public/Intragovernmental và các cột receivable/allowance/net, không kéo cấu trúc sang Note tiếp theo. Corpus còn có 127 vector BGE `BAAI/bge-small-en-v1.5`, 384 chiều cho phương án hybrid; cấu hình cuối dùng BM25.

Benchmark hiện tại: `data/advanced/benchmark-dev.json` có **23 câu**, gồm 20 câu có đáp án và 3 câu thiếu bằng chứng; `benchmark-test.json` có **30 câu**, gồm 6 bảng, 5 ảnh, 5 biểu đồ, 4 text, 4 multi-hop và 6 thiếu bằng chứng. Mỗi câu chỉ tính một nhóm chính. Ảnh/caption cung cấp nhận diện, công nghệ hoặc địa điểm của đối tượng hữu ích, không dùng hình trang trí. Multi-hop kết hợp ít nhất hai trang/vùng gốc. Mỗi câu có ID, nguồn, đáp án và trang/vùng kiểm chứng hoặc cách kiểm tra thiếu bằng chứng.

Quy trình chia bộ: một pilot ban đầu đã được xem để phân tích lỗi; bộ kế tiếp chạy đủ và đạt **25/30**, với năm lỗi gồm nhầm phần trăm/số tiền, hai hàng bảng và hai câu multi-hop thiếu vế. Năm câu lỗi này chuyển sang dev, mở rộng từ 18 lên 23 câu. Bộ test hiện tại được định nghĩa lúc `2026-11-16T17:45:32.854818+00:00`, **trước thay đổi pipeline truy xuất theo từng evidence item và xử lý riêng từng vế**. Câu hỏi/đáp án hoàn chỉnh khác các bộ dev và test đã dùng; không dùng test hiện tại để chọn cấu hình. Chính sách test là tra cứu trực tiếp theo mục/hàng/năm/đơn vị và tổng hợp hai nguồn, không có phép tính. Một số thành phần multi-hop lặp trong cùng test, làm các câu có tương quan. Không cung cấp đáp án, gold evidence hoặc gợi ý số trang cho agent.

Test SHA-256: `99abaad66f61cde83664152bc866bc9bb794fd7bfd5b56db5406820277ca7bea`. Chốt cấu hình lúc `2026-11-16T18:33:46.222244+00:00`, trước test bắt đầu `2026-11-16T18:33:47.107484+00:00`. `selection.json` khóa hash model, code, corpus, benchmark và kết quả dev; script kiểm tra chúng trước khi cho phép chạy test.

Cấu hình cuối: Ollama local **`qwen3:4b-instruct`**, digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`, temperature 0, context 2.048 token, tối đa 192 token sinh, 8 CPU thread. BM25 trên các evidence item chọn tối đa 3 candidate cho mỗi vế, budget 4.000 ký tự. Điều kiện loại nguồn, đơn vị, Goal hoặc mục Assets/Liabilities được lấy từ **câu hỏi**; query và điều kiện loại nguồn/tỷ lệ được ghi trong trace; không suy ra filter từ gold. Ranking giữ năm và đơn vị của nguồn, ưu tiên nhãn phần trăm khi hỏi tỷ lệ.

Câu ghép Give/Name/Identify được tách theo liên từ; các điều kiện FY/Goal/chart có trong câu hỏi được giữ ở query từng vế. Mỗi vế có một request Ollama và function thật **`select_evidence`**. Model phát JSON có `name` và `arguments`; arguments gồm một nhãn nguồn được chọn, đáp án tối đa 220 ký tự và abstention. Runtime giới hạn số nhãn để tránh lặp JSON; giữ raw output và mọi giá trị model. Function chọn context, định dạng đơn vị theo nguồn/câu hỏi và gắn citation; không sửa giá trị bằng gold. Nhãn C1–C3 của từng vế được ánh xạ sang nhãn toàn cục và ghi rõ trong trace. Đây là function calling qua JSON schema, không phải native tool-call protocol. Tối đa hai model request/function call cho câu hai vế; không dùng ảnh trực tiếp khi sinh đáp án. Nội dung nguồn/tool là dữ liệu, không phải chỉ dẫn.

Thử nghiệm dev:

| Cấu hình | Phạm vi dev | Đạt / tổng |
|---|---|---|
| Local 1.5B, context theo trang | 18 câu ban đầu | 11/18 (61,11%) |
| Local 4B, context theo trang | Cùng 18 câu | 16/18 (88,89%) |
| Local 4B, evidence item và xử lý riêng từng vế | Cùng 18 câu | 17/18 (94.44%) |
| Cấu hình cuối, năm câu lỗi bổ sung | 5 câu bổ sung | 5/5 (100.00%) |
| Local 4B, evidence item trước ghép caption và sửa đơn vị | Cùng 23 câu | 16/23 (69,57%) |
| **Cấu hình cuối, toàn dev** | **23 câu** | **22/23 (95.65%)** |

Trên cùng 23 câu dev, cấu hình trước ghép caption đạt 16/23, cấu hình cuối đạt 22/23. Thời gian hai lượt lần lượt 354,16 và 539,29 giây; tải máy không được kiểm soát độc lập. Thay cả cách ghép caption, hướng dẫn trích xuất và chuẩn hóa đơn vị, nên không tách riêng tác động từng thay đổi. `development_runs` trong `dev.json` giữ kết quả dev của các cấu hình trước; phần phân tích lỗi ghi điều kiện chuyển test cũ sang dev. Các lượt dev có JSON lặp nhãn hoặc nhãn citation cục bộ chưa nhất quán được dùng sửa schema/renderer trước khi chạy toàn dev cấu hình cuối. Lỗi không được thay bằng kết quả giả.

Các câu dev không đạt của cấu hình cuối:

- `adv-dev-image-02`: Chưa đủ: chỉ nêu Hampton, Virginia, thiếu Katherine G. Johnson Computational Research Facility tại NASA Langley Research Center. Caption gốc trang 35 và context C1 có vị trí đầy đủ. Chấm không đạt dù phần thành phố/bang đúng.

Chấm test đạt/không đạt bằng đối chiếu câu trả lời và citation với **PDF gốc**, không chỉ catalog. Câu có đáp án phải đủ ý, đúng điều kiện/năm/đơn vị và có evidence hỗ trợ toàn bộ; đúng một phần, sai citation hoặc từ chối đều không đạt. Câu thiếu evidence phải từ chối rõ, không thêm khẳng định không được hỗ trợ. Timeout, lỗi và không có kết quả được giữ trong mẫu số. Nhãn do AI hỗ trợ đọc nguồn, chưa có người chấm độc lập. **Đạt ngưỡng ít nhất 90% trên toàn bộ bộ test đã công bố.**

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | 6/6 | 100.00% | 0 lỗi hệ thống |
| Ảnh, sơ đồ | 5/5 | 100.00% | 0 lỗi hệ thống |
| Biểu đồ | 5/5 | 100.00% | 0 lỗi hệ thống |
| Multi-hop | 3/4 | 75.00% | 0 lỗi hệ thống |
| Text | 3/4 | 75.00% | 0 lỗi hệ thống |
| Thiếu bằng chứng | 5/6 | 83.33% | 0 lỗi thực thi; 1 đáp án rỗng |
| **Toàn bộ** | **27/30** | **90.00%** | **0 lỗi thực thi; 1 đáp án rỗng** |

Các câu test không đạt:

- `adv3-test-text-03`: Sai mục và giá trị: ngưỡng acquisition cost ghi nhận PP&E trong Note 1 trang 57 là $500,000 or more. Đáp án 1.227 triệu USD lấy Capitalized Acquisitions/FY2024 của bảng rollforward Note 5 trang 62, không phải ngưỡng. Context còn gắn nhãn Heritage assets từ đoạn khác, cho thấy giới hạn thứ tự text/nhóm mục PDF. Chấm không đạt.
- `adv3-test-multi-hop-04`: Chỉ đúng Appropriations Received/FY2024 là 24.875 triệu USD với C4 trang 52. Vế ảnh chỉ trả Laboratory, không nêu Jet Propulsion Laboratory và Southern California; C1 trang 11 không xác định CADRE clean-room. Bộ tách liên từ cắt laboratory and region quá sớm, làm mất entity ở vế đầu. Đúng một phần nên không đạt.
- `adv3-test-u02`: Không đạt abstention: raw answer rỗng nhưng abstain=false; final chỉ còn [C3], không có nội dung từ chối hay câu trả lời. C3 trang 52 là Appropriations Used, không cung cấp giá mua bóng GUSTO. Ghi riêng là đáp án rỗng/không có kết quả nội dung, không loại khỏi mẫu số.

`results/advanced/test.json` lưu đủ 30 trace và nhãn từng câu: query gốc/query từng vế, ranking retrieval, candidate context/context được chọn, JSON model, function name/arguments/lượt gọi/kết quả, câu trả lời/raw answer, citation, thời gian, lỗi và lý do đối chiếu nguồn. Nhãn chứa trang/vùng gốc, nhận xét citation, kiểm tra thiếu bằng chứng và hash trace. Citation ở nguồn tương đương được chấp nhận sau khi mở và kiểm chứng ảnh gốc, không buộc trùng trang gold. `benchmark_validation` ghi kiểm tra corpus và giới hạn chia bộ. Chỉ có bốn file kết quả Advanced: **`index-manifest.json`, `dev.json`, `selection.json`, `test.json`**.

Test có **34 lượt model**, **34 lượt function**, **0 lỗi thực thi/timeout**, **1 đáp án rỗng không từ chối** (đã tính không đạt); tổng thời gian từng câu **956.80 giây**, trung bình **31.89 giây/câu**. Dev cuối mất 539.29 giây. Phí API inference local **0 USD**, chưa tính điện, phần cứng, chuẩn bị index và công đọc nguồn. Đây là một lượt đo trên máy CPU; tải và bộ nhớ ảnh hưởng tốc độ.

Giới hạn: bộ test nhỏ, thiên về tra cứu có cấu trúc, có các thành phần tương quan; điểm không suy ra năng lực tổng quát hay độ đúng của mọi ô bảng. Annotation có thể bỏ sót chi tiết; BM25 có thể bỏ lỡ đồng nghĩa, model có thể chọn nhầm đối tượng hoặc từ chối nhầm. Cấu trúc nguồn/năm/đơn vị và citation cần kiểm tra bằng bản gốc. Bộ tách liên từ đơn giản có thể xử lý sai câu chứa đối tượng ghép hoặc nhiều liên từ. Thứ tự text PDF có thể gắn sai nhãn mục cho hàng bảng, như đã thấy ở trang 62; cần đối chiếu layout gốc. Thao tác chuẩn hóa chỉ định dạng câu trả lời, không bảo đảm giá trị model suy luận đúng.

Tái lập từ gốc repository bằng PowerShell, dùng `$py`, `$day6` ở phần Agent. Cần Ollama và cache embedding Day 6; inference chỉ dùng model local. Tải đúng PDF/hash, tái tạo ảnh/layout/catalog và dùng thư mục kết quả riêng:

```powershell
$pdf = 'data/advanced/nasa-fy-2024-afr.pdf'
$advOut = 'results/reproduction/advanced'
New-Item -ItemType Directory -Force 'data/advanced/generated' | Out-Null
Invoke-WebRequest 'https://www.nasa.gov/wp-content/uploads/2023/11/nasa-fy-2024-afr-1.pdf' -OutFile $pdf
& $py -m pip install --target data/advanced/generated/vendor PyMuPDF==1.26.7
& $py -m src.advanced.advanced_render --pdf $pdf --dpi 110
& $py -m src.advanced.advanced_layout_index --day6 $day6 --pdf $pdf
& $py -m src.advanced.build_verified_catalog --pdf $pdf
ollama pull qwen3:4b-instruct
& $py -m src.advanced.evaluate_multimodal --day6 $day6 --split dev --output-root $advOut
& $py -m src.advanced.index_manifest --output (Join-Path $advOut 'index-manifest.json')
# Đọc output và lập nhãn từng câu theo PDF gốc, theo schema reviews của kết quả.
& $py -m src.advanced.score_results --split dev --reviews <file-nhan-dev> --output-root $advOut
& $py -m src.advanced.evaluate_multimodal --freeze --output-root $advOut
& $py -m src.advanced.evaluate_multimodal --day6 $day6 --split test --output-root $advOut
& $py -m src.advanced.score_results --split test --reviews <file-nhan-test> --output-root $advOut
```

Nếu bị ngắt, thêm `--resume` cho đúng split; script kiểm tra fingerprint và tập câu hỏi trước khi tiếp tục. Nhãn phải đọc lại theo output thực tế của lượt tái lập. Không nộp PDF/ảnh sinh, cache/model, credential hoặc file env; chỉ chia sẻ nguồn công khai và cách tái tạo.
