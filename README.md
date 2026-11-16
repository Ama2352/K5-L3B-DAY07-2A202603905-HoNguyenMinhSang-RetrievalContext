# Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.

## Bài làm hiện tại

- [Checkpoint 1: corpus, kết nối và benchmark](docs/checkpoint1.md)
- [Checkpoint 2: agent và trace đầu tiên](docs/checkpoint2.md)
- [Checkpoint 3: so sánh dev và chọn cấu hình](docs/checkpoint3.md)
- [Checkpoint 4: đánh giá test](docs/checkpoint4.md) · [Báo cáo cá nhân](report/individual/2A202603905_HoNguyenMinhSang.md)

Dữ liệu Day 6 được giả định nằm ở `../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion` tính từ repo này. Xem checkpoint 1 để dựng lại chunk, khởi động pgvector và cấu hình `DATABASE_URL`; không commit mật khẩu hay `.env`. Model embedding BGE và model agent `llama3.2:3b` qua Ollama chạy local. Các trace thực tế nằm trong `results/`; kết quả mock của test chỉ dùng kiểm tra code.

Từ repo này, dùng Python/venv của Day 6 khi chạy truy xuất thật:

```powershell
python -m unittest discover -s tests -v
../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion/.venv/Scripts/python.exe src/retrieval_dev.py --day6 ../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion --split dev --output results/reproduction/retrieval-dev
../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion/.venv/Scripts/python.exe src/agent_dev.py --day6 ../../lab6/K5-L3B-DAY06-2A202603905-DataIngestion --split dev --output results/reproduction/agent-dev
```

Các lệnh agent cần Ollama chạy và có model `llama3.2:3b`; lệnh có thể mất nhiều thời gian trên CPU. Bản test chỉ chạy sau khi chốt cấu hình trên dev, với `--split test` và thư mục output riêng; không dùng test để điều chỉnh hệ thống.
