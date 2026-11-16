# Báo cáo Day 7 — Hồ Nguyễn Minh Sang, 2A202603905

## Dữ liệu và benchmark

Corpus Day 6 là Raffel và cộng sự, *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, JMLR 21(140), 2020, [PDF gốc](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf), CC BY 4.0. Chọn trang PDF vật lý 4–40 (37 trang); chunk B theo đoạn tối đa 1.000 ký tự, 160 chunk, giữ ID, nguồn, trang và offset. Bảng/công thức trích xuất có thể sai định dạng; PDF là nguồn đối chiếu. Xem `docs/checkpoint1.md` và attribution/manifest trong repo Day 6.

Day 6 dùng PostgreSQL 16.16, pgvector 0.8.7, exact cosine, `BAAI/bge-small-en-v1.5` FastEmbed 384 chiều và prefix query của model. Kết nối và 160 ID/text/metadata/số chiều đã kiểm tra thực tế. Agent dùng Ollama `llama3.2:3b`; thông tin chạy và đường dẫn Day 6 trong README. Chi phí API 0 USD, không bao gồm phần cứng và điện.

`data/benchmark.json`: dev 15 câu có đáp án + 2 thiếu evidence; test mới 5 có đáp án + 2 thiếu evidence. Day 6 test đã dùng nên không tái sử dụng làm test độc lập. Gold có nguồn/trang/quote và không truyền vào model hoặc retrieval; câu thiếu evidence giới hạn trong trang 4–40, đã tìm các thuật ngữ liên quan trong toàn bộ text trích xuất, vẫn có giới hạn lỗi trích xuất hoặc cách diễn đạt khác. Các câu trả lời/nhãn được Codex đọc đối chiếu; chưa có người chấm độc lập. Chốt lựa chọn bằng dev trước khi chạy test.

## Retrieval và thử nghiệm

Ba phương pháp là dense, TF-IDF và hybrid RRF; thêm cấu hình hybrid top-5 với budget lớn hơn. Tất cả dùng query gốc, cùng corpus, không filter; hybrid lấy 20 candidate mỗi nguồn và RRF k=60. Context đóng gói nguyên chunk, khử trùng ID. Hit yêu cầu quote trong một chunk đúng nguồn/trang, nên có thể bỏ sót evidence tương đương.

| Cấu hình | Split | Hit@1 | Evidence Hit@3 | Context evidence | Thời gian truy xuất trung bình |
|---|---|---|---|---|---|
| Dense, top-3 / 4.000 ký tự | dev | 12/15 | 13/15 | 13/15 | 49,4 ms |
| TF-IDF, top-3 / 4.000 | dev | 11/15 | 12/15 | 12/15 | 1,2 ms |
| Hybrid, top-3 / 4.000 | dev | 8/15 | 14/15 | 14/15 | 50,7 ms |
| Hybrid, top-5 / 6.000 | dev | 8/15 | 14/15 | 15/15 | 50,7 ms |
| Dense baseline | test | 1/5 | 3/5 | 3/5 | 143,2 ms |
| Hybrid top-5 đã chọn | test | 1/5 | 3/5 | 4/5 | 145,9 ms |

Thời gian là một lần chạy warm, không gồm khởi tạo model/index; test chạy trong lúc agent hoạt động nên không so trực tiếp với dev. Retrieval test dùng query gốc. Trace và thời gian chính xác: `results/checkpoint3/dev/`, `results/checkpoint4/retrieval-test/`. Hybrid và hybrid top-5 là hai cấu hình cải thiện có giả thuyết riêng; xem `docs/checkpoint3.md` về q01/q09 được hybrid phục hồi và q05 bị mất ở top-3 nhưng xuất hiện trong top-5. Agent q05 trả sai dù context có evidence, cho thấy retrieval tốt không bảo đảm grounded answer.

## Agent và đánh giá

Model chọn query tìm kiếm; tối đa 3 lượt, dense top-3 hoặc hybrid top-5. Trace ghi action, kết quả, context, citation, token model, lỗi và thời gian. Bộ kiểm tra ID chỉ xác nhận citation tồn tại trong context; nhãn hỗ trợ được đọc riêng. Không có fallback mock. Quy tắc chọn được ghi trước test: tổng điểm grounded answer dev, rồi abstention, context evidence và thời gian. Hybrid thắng hẹp 9/30 so với 8/30, chậm hơn và còn sai nhiều. Quyết định tại `results/checkpoint3/selection.json`.

| Cấu hình / split | Tool use | Lượt search | Grounded 0 / 1 / 2; trung bình /2 | Citation hỗ trợ | Câu có citation | Abstention đúng | Từ chối nhầm | Lỗi |
|---|---|---|---|---|---|---|---|---|
| Dense / dev | 17/17 | 1 mỗi câu | 11 / 0 / 4; 8/15 | 4/11 | 11/17 | 1/2 | 5/15 | 0 |
| Hybrid / dev | 17/17 | 1 mỗi câu | 10 / 1 / 4; 9/15 | 5/10 | 10/17 | 1/2 | 7/15 | 0 |
| Dense / test | 7/7 | 1 mỗi câu | 4 / 0 / 1; 2/5 | 1/6 | 6/7 | 0/2 | 1/5 | 0 |
| Hybrid / test | 6/6 thành công | 1 mỗi câu thành công | 4 / 0 / 0; 0/4 | 0/5 | 5/6 | 1/2 | 1/4 thành công | 1 timeout |

Trung bình grounded trong bảng là tổng điểm chia số câu có đáp án chạy thành công, trên thang 2. Lỗi được báo riêng theo benchmark; nếu tính lỗi hybrid là 0, tổng vẫn 0/5. Một câu hybrid thiếu evidence nói không có giá nhưng starter gắn `answered` vì có citation; chấm abstention dựa nội dung, citation không chứng minh sự vắng mặt. Test hybrid t04 timeout 185 giây; script không lưu được context đã có trước timeout, nên không thể chấm context câu này. Trung bình thời gian cả lỗi: dense 43,7 s/câu, hybrid 70,6 s/câu. Tất cả là một lần chạy, mẫu nhỏ.

Context cuối theo query do agent viết lại, được đọc cho từng câu test có đáp án: dense mức 0/1/2 = 1/2/2; hybrid = 1/0/3 và 1 lỗi không có context. Context đủ evidence tự động: dense 2/5; hybrid 3/5 tính cả lỗi là không có context. Đây khác bảng query gốc. Lý do từng câu và citation tại `results/checkpoint4/test_reviews.json`, trace tại `results/checkpoint4/agent-test/`; nhãn do Codex kiểm tra, không phải đánh giá mù hay do học viên xác nhận.

Ví dụ: dense t05 trả 17 GB và dẫn đúng trang 26. Dense u04 bịa 0,5 kWh; hybrid u04 bịa một nghìn tỷ kWh, đều không được nguồn hỗ trợ. Hybrid t02 trả đúng 0,99 nhưng dẫn trang 25 thiếu ngưỡng; nguồn đúng ở trang 6. Các lỗi này cho thấy không thể tin citation hợp lệ về cú pháp, và hệ thống chưa sẵn sàng dùng cho câu hỏi thực tế.

## Thử thách nâng cao

Chưa hoàn thành: chưa có corpus đa phương thức từ 100 trang, benchmark riêng, truy xuất hình/bảng/biểu đồ, hoặc test đạt 90%. Không tuyên bố điểm bonus.
