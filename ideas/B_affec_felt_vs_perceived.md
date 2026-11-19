# Ứng viên B: AFFEC, phân biệt cảm xúc mình cảm thấy (felt) với cảm xúc mình nhận ra ở người khác (perceived)

Mục 18 trong survey của user.
**Trạng thái (2026-11-19):** **CHƯA ĐẠT, TẠM DỪNG.** C5 đã xác minh (L3). **C2 trượt**: chưa tìm được dataset mở thứ hai có cả nhãn felt lẫn perceived trên cùng lượt. C4 mới quét bước đầu.

## Dữ kiện đã xác minh
- [SỰ THẬT] Dữ liệu (Zenodo 19408373, `core.zip`):
  - 72 người, 84 stimulus CREMA-D, 5.524 trial chính có đủ 4 nhãn trên thang 1–9.
  - Tương quan felt–perceived: 0,635 (V), 0,575 (A).
  - Tỷ lệ \|felt − perceived\| > 1 điểm: 30,5% (V), 42,7% (A).
- [SỰ THẬT] **Phân tích baseline chỉ dùng nhãn**, kiểm định chéo 5-fold theo người tham gia (held-out participants):

  | Cách dự đoán felt | R² felt V | R² felt A |
  |---|---|---|
  | Trung bình felt của stimulus | 0,347 | 0,134 |
  | Hồi quy tuyến tính từ perceived của chính người đó | 0,389 | 0,321 |
  | Stimulus + perceived của chính người đó | 0,414 | 0,322 |
  | Thêm độ lệch riêng của người (ước từ một nửa số trial của chính họ) | **0,452** | **0,431** |

- [SỰ THẬT] Baseline đa phương thức trong paper AFFEC (arXiv 2504.18969v2, §9) đạt macro-F1 0,43–0,48 trên bài toán 3 lớp.
- [SỰ THẬT] AFFEC có 11 bài trích dẫn (Semantic Scholar). Theo tiêu đề, không bài nào làm bài toán tách felt khỏi perceived.

## Đánh giá chuyên môn
- [ĐÁNH GIÁ] Câu hỏi "gán đúng vai felt và perceived" bị **một baseline tầm thường** chi phối: felt ≈ (perceived kéo về gần trung tính) + độ lệch riêng của từng người. Bài báo phải chứng minh tín hiệu sinh lý và hành vi (EEG, gaze, GSR, AU) **thêm được thông tin vượt baseline này**. Hiện chưa có bằng chứng điều đó xảy ra.
- [ĐÁNH GIÁ] Chỉ có một dataset mở (K-EmoCon thì gated), nên reviewer ICME sẽ nêu ngay câu hỏi khả năng tổng quát hóa.
- **Cần làm để mở lại:**
  1. Tìm dataset mở thứ hai có cả felt và perceived (quét có hệ thống mảng âm nhạc và phim).
  2. Pilot: tín hiệu người xem có tăng R² vượt 0,45 (V) và 0,43 (A) không.
