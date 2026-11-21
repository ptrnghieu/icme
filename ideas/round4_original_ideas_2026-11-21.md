# Đợt 4: ý tưởng tự nghĩ, xuất phát từ bản chất của dữ liệu cảm xúc và từ tài sản riêng của user (2026-11-21)

**Bối cảnh:** user chỉ ra hai lỗi:
1. Mình quanh quẩn ở các benchmark đông người làm (MOSI, SIMS, MELD).
2. Mình đọc rồi kiểm lại công trình của người khác thay vì tự nghĩ ra cái mới.

**Cách làm đợt này:**
- Ý tưởng xuất phát từ nguồn gốc của dữ liệu: MER2023–2025, CH-SIMS và MELD đều cắt từ **phim, phim truyền hình, sitcom**, tức là media đã qua hậu kỳ.
- Ý tưởng dùng tài sản riêng của user (MER2023–2026).
- Kiểm novelty **nhanh** chỉ để không đi lại đường cũ. Sau đó chạy ngay phép thử premise rẻ.

**Trạng thái:** đây là phác thảo, **chưa qua C1–C5**, chưa phải ứng viên.

## E. Hậu kỳ là "modality ẩn" → **ĐÃ LOẠI theo tiêu chí đặt trước của P6** (2026-11-21)
**Kết quả P6** (`explore/xling/p6_report_2026-11-21.md`):
- CH-SIMS: 38,7% clip có nhạc; chỉ riêng tag hậu kỳ dự đoán cực tính được bacc 59,1 (+9,1 [+4,8; +14,5]) → đạt.
- MELD: chỉ 9,6% clip có nhạc hoặc tiếng khán giả, dưới ngưỡng 15%. Tag hậu kỳ vẫn đạt +6,2 [+3,2; +9,5] → trượt vì tỉ lệ xuất hiện thấp.
- Quy tắc đòi đạt cả CH-SIMS lẫn MELD → **DOES NOT HOLD**.

**Post-hoc:**
- Split của CH-SIMS dùng chung video. Tách theo video thì bacc tụt còn 54,9.
- Mô tả thêm: khi có nhạc và tag hậu kỳ trái với nhãn, mô hình audio WavLM chỉ đúng 57,1%, so với 71,1% khi tag khớp nhãn (n = 63 và 90).
- MER2024 không có ID video nguồn, nên không kiểm soát được kiểu rò rỉ này.

### Phác thảo ban đầu (giữ lại để tham khảo)
**Insight:**
- Clip trong các dataset MER in-the-wild được cắt từ media đã biên tập. Hậu kỳ cài vào đó tín hiệu cảm xúc **của đạo diễn**, không phải **của nhân vật**:
  - nhạc nền (phim và phim truyền hình Trung Quốc: MER2023–2025, CH-SIMS);
  - tiếng cười thu sẵn (sitcom: MELD là phim Friends);
  - phụ đề in cứng vào khung hình, mà encoder visual kiểu CLIP đọc được;
  - lồng tiếng hậu kỳ, làm giọng và khuôn mặt không cùng một người hoặc không cùng thời điểm diễn.
- Mô hình học được các tín hiệu này. Kết quả là điểm in-domain bị thổi phồng, còn khi triển khai trên dữ liệu không biên tập (video call, ghi âm thật) thì hỏng. Khi nhạc nền mâu thuẫn với cảm xúc nhân vật, fusion bị kéo sai.

**Phương pháp (phác thảo): Editorial-Aware MER.**
1. Tách luồng hậu kỳ:
   - audio: tách nguồn (giọng nói và nhạc đệm), cộng tagger AudioSet cho tiếng cười và nhạc;
   - visual: phát hiện và che vùng phụ đề;
   - độ đồng bộ môi–tiếng làm tín hiệu lồng tiếng.
2. Mô hình hóa luồng hậu kỳ như một modality "ngữ cảnh cảnh quay" riêng, có head riêng.
3. Huấn luyện bằng **hoán đổi hậu kỳ phản thực tế**: ghép giọng nói của clip A với nhạc nền của clip B có nhãn khác, rồi phạt khi dự đoán cảm xúc nhân vật thay đổi.

**Đóng góp:**
- Dạng mạnh nhất: phương pháp cải thiện cả in-domain lẫn khi chuyển từ dữ liệu biên tập sang dữ liệu không biên tập, trên ít nhất 2 dataset. Đây là chủ đề multimedia đúng nghĩa (hậu kỳ), khớp ICME.
- Dạng yếu nhất còn lại: phân tích shortcut, không đạt C3. Vì vậy tầng 4 phải đo được lợi ích của phương pháp, không chỉ đo sự tồn tại của shortcut.

**Data:**
- Có ngay: CH-SIMS (raw, L2), MELD (L3), MOSI (vlog, làm đối chứng âm).
- Sau khi user chạy L3: MER2023/2024.
- Dữ liệu không biên tập để kiểm tra chuyển giao: EmotionTalk (gated auto, user chưa xác nhận), CREMA-D.

**Novelty nhanh** (`literature/gap_mining/round4_quick_novelty_2026-11-21.txt`):
- arXiv: 0 bài cho laugh track; 0 bài cho nhạc nền × emotion recognition/sentiment × speech/multimodal; 0 bài cho phụ đề × shortcut × emotion recognition.
- Lồng tiếng/đồng bộ AV × emotion recognition: 3 kết quả, không bài nào liên quan.
- WebSearch: không thấy bài nào coi nhạc nền hay tiếng cười là shortcut trong MER2023/2024 hay MELD.
- Chưa đọc full text. Phải làm đủ ở tầng 3 nếu premise đạt.

**Phép thử premise P6** (tiêu chí đặt trước: docstring `explore/xling/p6_analyze.py`):
- Gắn nhãn AudioSet bằng AST cho mọi clip.
- Premise đứng khi: tỉ lệ clip có nhạc hoặc tiếng khán giả ≥ 15%, **và** riêng các đặc trưng hậu kỳ dự đoán được nhãn cao hơn mức ngẫu nhiên ≥ 3 điểm (CI không chứa 0).
- Phải đúng **cả ở CH-SIMS lẫn MELD**. MOSI là đối chứng.

## F. Fusion EEG–fNIRS có tính độ trễ huyết động (ưu tiên 2)
**Insight:**
- Tín hiệu fNIRS (HbO/HbR) trễ hơn hoạt động thần kinh mà EEG đo được khoảng vài giây, theo hàm đáp ứng huyết động. Đây là kiến thức nền, phải trích nguồn khi viết.
- Các mô hình gộp hai tín hiệu theo cùng cửa sổ thời gian thì bỏ qua độ trễ này.

**Phương pháp:**
- Cầu nối "neurovascular coupling" học được: tích chập đường bao các dải tần EEG với một HRF có tham số, thích nghi theo từng người.
- Cross-attention có prior độ trễ.

**Data:**
- Track MER-PS của MER2026 (user có quyền; cần L3).
- Cần thêm một dataset EEG–fNIRS cảm xúc mở thứ hai cho C2. Chưa tìm.

**Novelty nhanh:**
- 4 bài EEG × fNIRS × emotion trên arXiv, trong đó có 2610.03618 ("Video-Time Priors for EEG-fNIRS Emotion Regression"). Phải đọc full text.
- Rủi ro C2 và C5.

## G. Tham chiếu theo từng người, tìm từ kho không nhãn (ưu tiên 3)
**Insight:** người xem đánh giá biểu cảm so với "mặt bình thường" và "giọng bình thường" của chính người đó. MER2024/2025 có kho clip không nhãn rất lớn, trong đó cùng một diễn viên xuất hiện nhiều lần.

**Phương pháp:**
- Gom cụm danh tính (face và voice embedding) trên kho không nhãn.
- Ước lượng trạng thái nền của từng danh tính.
- Biểu diễn mỗi clip bằng độ lệch so với trạng thái nền.

**Novelty nhanh:** 6 bài về tách danh tính trong FER (2018–2022) → trùng một phần. Phần mới là làm đa modality và tìm danh tính không cần nhãn.

**Rủi ro:** speaker normalization trong SER đã có (2202.01252).
