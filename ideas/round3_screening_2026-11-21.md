# Đợt tìm ý tưởng thứ 3 (bắt đầu 2026-11-21)

**Phạm vi data:** MER2023–2026 (user có quyền truy cập, chưa lên L3) cộng các bộ mở đã xác minh (MELD, CH-SIMS/v2, MOSI, mirror MOSEI, CREMA-D, RAVDESS, MERGE). **Không có IEMOCAP.**
**Giữ nguyên:** C1–C5 và yêu cầu phương pháp mới.

## Nguồn C1 đã đọc
- **MER 2023** (arXiv 2304.08981), **MER 2024** (2404.17113), **MER 2025** (2504.19423): đọc từng phần (protocol, kết quả, kết luận).
- **MER 2026** (2604.19417): đọc toàn văn (§1–7).
  - Bốn track: MER-Cross (cảm xúc người nghe), MER-FG, MER-Prefer, MER-PS (EEG–fNIRS).
  - Kích thước train+val của MER-SEMI: 3.373 (2023), 5.030 (2024), 7.369 (2025). MER-Cross có 9.395 mẫu "cá nhân" để train và 574 mẫu người nghe để test.
  - Metric: WAF (chính) và ACC.
  - **§2:** "considerable performance gap between individual and interlocutor emotions … multimodal fusion … does not improve performance on interlocutor emotions. Therefore, participants are encouraged to develop novel strategies to bridge this gap."
  - Bảng 2: trên test, audio chỉ đạt khoảng 30–36 WAF và text khoảng 32–34; CLIP-large (visual) đạt 58,88 WAF; fusion Top-1 đạt 57,44.

## Hướng đang xét
| # | Hướng | Trạng thái | Bằng chứng và rủi ro |
|---|---|---|---|
| R3-1 | MER-Cross: nhận diện cảm xúc người nghe từ video người nghe, cộng audio/text của người nói | **ĐANG XÉT** (cần user quyết về dual submission; cần L3 để biết nhãn test đã công bố chưa) | **C1:** bài tổng kết MER2026 §2 (trích ở trên). **C4:** RASG (arXiv 2610.03016, hạng 2 MER-Cross, đọc abstract và §1) chọn mẫu người nói có biểu cảm khớp nhãn, pseudo-label track người nghe, và chỉ gọi LLM khi visual không chắc; báo 76,25%. Đội hạng 1 chưa công bố (WebSearch 2026-11-21). **C2:** chỉ có một tập test (574 mẫu). **COLING:** ảnh hưởng của người nói lên người nghe gần với chủ đề Hi-EF của bài COLING. **C5:** `track1_test.csv` tồn tại, chưa biết có nhãn hay không. |
| R3-2 | MER-SEMI hoặc MER-NOISE (tận dụng kho không nhãn, độ bền với nhiễu) | Ưu tiên thấp | Nhiều bài dự thi (2409.04447, 2409.07078, 2409.05015, ECMF 2508.05991, MoE 2508.06036, …). |
| R3-3 | MER-Prefer: mô hình chấm sở thích với mô tả cảm xúc | Ưu tiên thấp | Đã có bài kiểm toán shortcut (2607.18508) và 2608.24730. Chỉ có một họ dataset (EmoPrefer). |
| R3-4 | Dùng MER2024/2025 làm dataset thứ hai (tiếng Trung) cho các phương pháp MER tổng quát | Hạ tầng cho C2 | Data mới chỉ giải quyết C2, không tự tạo ra novelty. |
