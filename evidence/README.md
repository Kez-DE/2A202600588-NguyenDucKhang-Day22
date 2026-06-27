# Evidence — Day 22: LangSmith + Prompt Versioning

Bài nộp của **Nguyễn Đức Khang** (2A202600588).
Cấu hình: LLM = OpenRouter `owl-alpha` (bước 1–3) · Embeddings = FastEmbed `BAAI/bge-small-en-v1.5` (local) · LangSmith region = **APAC**.

## Danh sách bằng chứng

| Tệp | Nhiệm vụ | Trạng thái |
|-----|----------|-----------|
| `01_langsmith_traces.png`  | 1 — ≥50 traces trên LangSmith        | ✅ Đã có |
| `02_ab_routing_log.txt`    | 2 — log A/B routing (nhãn v1/v2)     | ✅ Đã có |
| `02_prompt_hub.png`        | 2 — 2 phiên bản prompt trên Hub      | ⬜ Cần chụp (Prompts trên apac.smith.langchain.com) |
| `03_ragas_report.json`     | 3 — báo cáo RAGAS                    | ✅ Đã có (số liệu thật) |
| `03_ragas_scores.png`      | 3 — bảng điểm RAGAS                  | ⬜ Tùy chọn (xem số liệu trong JSON) |
| `04_pii_demo_log.txt`      | 4 — demo PII detector                | ✅ Đã có |
| `04_json_demo_log.txt`     | 4 — demo JSON formatter              | ✅ Đã có |

---

## Phân tích V1 vs V2 (RAGAS — số liệu thật)

Hai system prompt khác biệt rõ về ngữ nghĩa:
- **V1 — NGẮN GỌN:** trả lời súc tích 2–4 câu, đi thẳng trọng tâm.
- **V2 — CÓ CẤU TRÚC:** giọng chuyên gia, phân tích facts, trả lời mạch lạc 3–5 câu.

Đánh giá 50 cặp QA qua **cả 2** phiên bản (LLM sinh + đánh giá: `owl-alpha`):

| Metric | V1 | V2 | Winner |
|--------|------|------|--------|
| faithfulness        | **0.9830** ⭐ | **0.9419** ⭐ | V1 |
| context_recall      | 1.0000 | 1.0000 | hòa |
| context_precision   | 0.9467 | 0.9533 | V2 |
| answer_relevancy    | _(xem ghi chú)_ | _(xem ghi chú)_ | — |

### Nhận xét

- **Faithfulness đạt mục tiêu ≥ 0.8 cho CẢ HAI phiên bản**, thậm chí **≥ 0.9 cả hai** (V1=0.983, V2=0.942) → đủ điều kiện điểm thưởng.
- **V1 (ngắn gọn) có faithfulness cao hơn V2 (có cấu trúc):** prompt ngắn gọn bám sát context, ít suy diễn thêm nên độ trung thực với nguồn cao hơn; V2 diễn giải dài hơn (3–5 câu, giọng chuyên gia) nên có nhiều "không gian" để lệch khỏi context một chút → faithfulness thấp hơn nhẹ.
- **context_recall = 1.0 ở cả hai và context_precision gần như nhau** vì cả hai dùng **cùng một retriever** (FAISS top-3) → phần truy xuất giống hệt nhau, chênh lệch chỉ đến từ phong cách diễn đạt của LLM, không phải từ retrieval.
- Tổng thể: **V1 nhỉnh hơn về độ trung thực**, V2 nhỉnh hơn rất nhẹ ở context_precision. Với hệ RAG bám nguồn, V1 (ngắn gọn) là lựa chọn an toàn hơn.

### Ghi chú trung thực về `answer_relevancy`

Trong lượt đánh giá hoàn chỉnh, `answer_relevancy` chưa tính được (trả về NaN) do **lỗi tương thích telemetry giữa RAGAS 0.4.3 và FastEmbedEmbeddings**: sự kiện `EmbeddingUsageEvent` của RAGAS yêu cầu trường `model` là `str`, nhưng `FastEmbedEmbeddings.model` là một object → `ValidationError`.

**Đã khắc phục trong mã nguồn** (`src/utils/llm_factory.py`): thêm lớp `_FastEmbedRagasSafe` bọc FastEmbed và phơi `.model` dạng string. Đã kiểm chứng `answer_relevancy` chạy ra số thật trên mẫu (≈ 0.59–1.0). Có thể tính đầy đủ 4/4 metrics bằng cách chạy lại `python 03_ragas_evaluation.py`.

---

## Ghi chú kỹ thuật

- **RAG pipeline:** knowledge base chia 107 chunks (size 500, overlap 50) → FAISS (FastEmbed 384-dim) → retriever top-3 → prompt → LLM → parser. Hàm `ask()` gắn `@traceable` → mỗi câu hỏi tạo 1 trace trên LangSmith.
- **A/B routing tất định:** `int(md5(request_id), 16) % 2` → V1/V2 (cùng request_id luôn cho cùng phiên bản). Log: V1=19 câu, V2=31 câu / 50.
- **Prompt Hub:** push & pull 2 phiên bản qua `client.push_prompt` / `client.pull_prompt` (region APAC).
- **PII detector:** regex phát hiện 4 loại (EMAIL, PHONE, SSN, CREDIT_CARD), dùng `FailResult(fix_value=...)` + `OnFailAction.FIX` để thay output bằng chuỗi đã che.
- **JSON formatter:** tự sửa 3 lỗi (gỡ markdown fences, nháy đơn → nháy đôi, xóa dấu phẩy thừa); trả JSON dự phòng khi không sửa được.
- **Môi trường:** langchain 0.3.x + ragas 0.4.3 + guardrails-ai 0.6.8 + openai 1.x + fastembed (xem `requirements.txt`). OpenRouter không có API embeddings nên dùng FastEmbed local.
