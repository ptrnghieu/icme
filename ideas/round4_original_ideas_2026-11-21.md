# Đợt 4: ý tưởng tự nghĩ, xuất phát từ bản chất của dữ liệu cảm xúc và từ tài sản riêng của user (2026-11-21)

**Bối cảnh:** user chỉ ra hai lỗi:
1. Mình quanh quẩn ở các benchmark đông người làm (MOSI, SIMS, MELD).
2. Mình đọc rồi kiểm lại công trình của người khác thay vì tự nghĩ ra cái mới.

**Cách làm đợt này:**
- Ý tưởng xuất phát từ nguồn gốc của dữ liệu: MER2023–2025, CH-SIMS và MELD đều cắt từ **phim, phim truyền hình, sitcom**, tức là media đã qua hậu kỳ.
- Ý tưởng dùng tài sản riêng của user (MER2023–2026).
- Kiểm novelty **nhanh** chỉ để không đi lại đường cũ. Sau đó chạy ngay phép thử premise rẻ.

**Trạng thái:** đây là phác thảo, **chưa qua C1–C5**, chưa phải ứng viên.

## E. Hậu kỳ là "modality ẩn" → **ĐÃ LOẠI, nhánh đóng hẳn** (user quyết ngày 2026-11-22)
**Lý do đóng:**
- P6 trượt tiêu chí đặt trước.
- Pilot cuối E′ không thể kiểm soát rò rỉ theo nguồn: MER2024 không có ID nguồn, và trên CH-SIMS không cách gom nhóm tự động nào đưa rò rỉ xuống dưới 0,95 (`explore/editorial/group_calibration_chsims_v2.md`).
- Thiết kế chéo dataset có xác suất thành công thấp.

Code trích feature của E′ (`explore/editorial/mer_editorial.py`) được giữ lại làm hạ tầng.

**Kết quả cũ của P6:**
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

## F. Fusion EEG–fNIRS có tính độ trễ huyết động → **BỊ CHẶN (C5)** (2026-11-22)
Repo HF `MERChallenge/MER2026` chỉ có `audio_7z/`, `openface_7z/`, `video_7z/` và CSV; không có data EEG–fNIRS cho MER-PS (`datasets/access_log.md`). Chưa có dataset EEG–fNIRS cảm xúc mở nào đã kiểm tra.

### Phác thảo ban đầu
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

## Lứa 4b (2026-11-22, song song với E′)

### I. "Peak-end": nhãn cấp clip là đánh giá hồi tưởng, không phải trung bình → **BỎ trước khi chạy pilot** (2026-11-22)
[Đánh giá] Clip của CH-SIMS, MELD và CREMA-D chỉ dài vài giây, thường là một câu nói, nên không có "đỉnh" và "đoạn cuối" tách biệt. Theo quy tắc power, không chạy phép thử khi hiệu ứng hợp lý nhỏ hơn MDE. Có thể xem lại nếu có dataset clip dài.
**Insight:**
- Người gán nhãn xem hết clip rồi cho một nhãn. Theo quy tắc peak-end trong tâm lý học, đánh giá hồi tưởng chủ yếu được quyết định bởi **khoảnh khắc mạnh nhất** và **đoạn cuối**. Đây là kiến thức nền, phải trích nguồn gốc khi viết.
- Hầu hết các mô hình MER lại pool trung bình theo thời gian.

**Phương pháp:**
- Pooling "peak-end" có thể học được: một đầu chọn đỉnh theo độ mạnh cảm xúc, một đầu lấy đoạn cuối.
- Áp được cho mọi modality và mọi dataset có nhãn cấp clip: MER2024/2025, CH-SIMS, MELD, CREMA-D.

**Novelty nhanh** (`literature/gap_mining/round4b_quick_novelty_2026-11-22.txt`):
- "peak-end" × emotion: 4 kết quả. Không bài nào dùng nó làm pooling cho nhận diện; chỉ có so sánh rating liên tục với rating hồi tưởng trong VR (2404.16487) và mô hình affect dựa trên preference.
- Apex/peak frame × multimodal/speech: 1 kết quả không liên quan.
- Chưa đọc full text.

**Phép thử premise rẻ** (trên CPU, tiêu chí sẽ đặt trước): feature theo đoạn 1 giây (WavLM), so sánh các cách gộp:
- mean, mean+std, đoạn cuối, đoạn đỉnh, đỉnh+cuối;
- trên CH-SIMS, MELD, CREMA-D.

**Rủi ro:** pooling thường chỉ đổi khoảng 1 điểm. Phải tính MDE trước khi chạy.

### R. Dùng "lý do cảm xúc" của MER2025 để giám sát việc chọn modality khi fusion → **ĐÃ LOẠI (C4, trùng một phần)** (2026-11-22)
**Ý:** phân tích 115 nghìn đoạn reason (MER-Caption) để biết mỗi clip có tín hiệu ở mặt, giọng hay lời. Dùng kết quả đó làm nhãn giám sát cho bộ định tuyến fusion. Mục tiêu là lấp chỗ P1 không học được gate từ confidence.

**Tiền lệ trong affective** (`literature/gap_mining/round4c_quick_novelty_2026-11-22.txt`, đọc abstract):
- 2508.03722: "Gemini to generate fine-grained, modality-separable reasoning traces, which are injected as priors during the fusion stage", trên đúng MER2024.
- RGL 2608.10448 (ICASSP 2026): rationale do MLLM sinh ra dùng để căn chỉnh biểu diễn nội bộ, trên IEMOCAP và MELD.

Phần còn mới chỉ là "rationale → nhãn modality → giám sát router". Phần này quá mỏng để làm đóng góp chính.

### α. "Neutral" là trạng thái vắng cảm xúc, không phải một lớp → **ĐÃ LOẠI (C1, P8 trượt 0/3)** (2026-11-22)
**Phương pháp đã thử:**
- Tách thành hai đầu: đầu "có cảm xúc hay không", huấn luyện có trọng số theo cường độ; và đầu phân lớp cảm xúc, chỉ train trên các clip có cảm xúc.
- So với softmax phẳng, và với softmax phẳng cộng bias neutral tinh chỉnh trên valid.

**Kết quả** (`explore/xling/p8_report_2026-11-22.md`): M − max(B1, B2):
- CREMA-D −1,5 [−3,2; +0,1] (MDE 2,4);
- MELD +0,1 [−1,5; +1,7] (MDE 2,2);
- CH-SIMS −4,3 [−7,1; −1,0] (MDE 4,4).

**Mô tả:** "hố hút neutral" có thật ở data tự nhiên.
- CH-SIMS: tỉ lệ clip có cảm xúc bị đoán thành neutral giảm theo cường độ, từ 37% xuống 5%.
- MELD: 30% clip có cảm xúc bị đoán thành neutral.
- CREMA-D (diễn viên đóng): chỉ 0–4%.
- Cách tách hai đầu không sửa được hiện tượng này; trên CH-SIMS nó còn làm tệ hơn.

Novelty nhanh: `literature/gap_mining/round4d_neutral_quick_2026-11-22.txt`.

### E – thảo luận lại theo yêu cầu của user (2026-11-23)
**Hai bằng chứng mới:**
- **CI của tín hiệu sau khi tách theo video:** trên CH-SIMS, chia 5-fold tách theo video (60 video, 5 seed), bootstrap theo video. Chỉ riêng tag hậu kỳ dự đoán cực tính được bacc − 50 = **+4,7 [2,6; 7,0]**. Tín hiệu có thật và không đến từ rò rỉ theo video, nhưng nhỏ. Trong 1.936 clip có 749 clip có nhạc; tỉ lệ nhãn dương là 0,40 khi có nhạc và 0,33 khi không có.
- **Chính ban tổ chức MER thừa nhận nhạc mang cảm xúc.** Bài MER2024 (2404.17113, phần MER-NOISE) viết về MUSAN: "The noise in the first two subsets [music, noise] may convey emotions and affect the emotion of the raw audio … Therefore, we only use the noise in the last subset [speech]". Tức là phép thử độ bền của MER2024 cố ý bỏ nhạc vì nhạc làm lẫn cảm xúc, trong khi bản thân các clip MER có sẵn nhạc nền. Đây là bằng chứng C1 trực tiếp từ bên dựng benchmark.

**Hướng phát triển (E2, chưa đăng ký trước):**
- Đổi câu hỏi từ "nhạc có dự đoán được nhãn không" sang **phép thử nhân quả**: hoán đổi nhạc nền giữa các clip (tách nguồn bằng Demucs rồi ghép lại) và đo tỉ lệ dự đoán bị lật.
- Phép thử này can thiệp trên chính clip đó, nên không cần ID nguồn và tránh được vấn đề rò rỉ trên MER2024.
- Data: CH-SIMS và CH-SIMS v2 có video_id, dùng cho đánh giá tách theo video; MER2024 dùng cho phép thử hoán đổi. Bỏ MELD.


**Cập nhật 2026-11-23: P9 (E2, hoán đổi nhạc nền) DOES NOT HOLD.** CH-SIMS: Δflip +1,9 [−0,7; +4,9] (MDE 3,9) < +5. Dừng M3ED theo quyết định của user vì kết luận không đổi. **Nhánh E đóng hẳn.** Chi tiết: `explore/editorial/p9_report_2026-11-23.md`.

**Cập nhật 2026-11-23: nhánh peak-end ĐÃ ĐÓNG theo quyết định của user**, trước khi P10 có kết quả (dừng ở bước trích feature, 300/2.281 clip CH-SIMS). Lý do: phản biện của user rằng attention pooling học được bao trùm peak-end về mặt biểu diễn, nên một quy tắc cố định khó thành đóng góp phương pháp (C3); clip CH-SIMS ngắn (median 3,21 s, khoảng 3 segment) nên peak, end và mean gần trùng nhau. Không có kết quả nào của P10/P10b. Script giữ lại: `explore/peakend/p10_peakend.py`, `p10b_attention.py`; screen `peakend` trong `explore/mer24/mer24_screen.py` không chạy.
