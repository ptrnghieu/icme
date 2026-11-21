# P0 / P0b: modality nào bị shift khi đổi ngôn ngữ, văn hóa, hoặc từ posed sang tự nhiên (2026-11-21)

Thí nghiệm sàng lọc cho ứng viên **D** (chéo ngôn ngữ) và **F4** (expressivity shift); xem `ideas/round2_screening_2026-11-21.md`. Chạy trên CPU của container, chỉ dùng encoder đông cứng.

## Thiết lập P0 (MOSI tiếng Anh ↔ CH-SIMS tiếng Trung)
- **Data:** raw mp4 từ mirror HF (`tamb2203579/CMU-MOSI`, `tamb2203579/CH-SIMS`); split chính thức theo cột `mode`.
- **Nhãn:** cực tính nhị phân (nhãn > 0 so với < 0, bỏ nhãn 0). Sau khi bỏ nhãn 0: MOSI còn 2.103 clip (51% dương), CH-SIMS còn 1.936 clip (36% dương).
- **Feature:**
  - text: `intfloat/multilingual-e5-base`, mean pooling;
  - audio: eGeMAPSv02 (88 chiều) và `microsoft/wavlm-base-plus` (trung bình các layer, rồi trung bình theo thời gian);
  - face: crop bằng Haar cascade trên 3 frame (25/50/75%), đưa qua `trpakov/vit-face-expression` (embedding CLS và 7 logit). Số frame có mặt trung bình: 2,94 (MOSI) và 2,57 (CH-SIMS).
- **Mô hình:** logistic regression với `class_weight=balanced`, chọn C trên valid của dữ liệu nguồn.
- **Chỉ số:**
  - TR = (bacc chéo − 50)/(bacc oracle − 50), trong đó oracle là mô hình train và test trên dữ liệu đích;
  - ATR là đại lượng tương tự trên AUC;
  - CI bootstrap ghép cặp theo video.
- **Tiêu chí cố định trước** (docstring của `p0_analyze.py`): premise của D đứng được khi TR(text) < TR(audio) và TR(text) < TR(face) ở cả hai chiều, với CI không chứa 0.
- **Translate-test:** `Helsinki-NLP/opus-mt-en-zh` và `opus-mt-zh-en`.
- **Kết quả đầy đủ:** `p0_report_2026-11-21.md`.

## Kết quả P0 (sự thật)
- **Premise của D: DOES NOT HOLD, và thực tế ngược chiều.** Ở chiều MOSI→SIMS:
  - TR(text) = +0,67, cao hơn TR(WavLM) = +0,19 (hiệu +0,50 [+0,08; +0,95]);
  - cao hơn TR(face emb) = +0,10 (hiệu +0,56 [+0,28; +0,84]).
- **SIMS→MOSI:**
  - TR(text) = +0,48; tăng lên +0,95 khi chuẩn hóa feature theo thống kê của dữ liệu đích (không cần nhãn). ATR(text) = +0,95.
  - Không tính được TR cho face, vì oracle face trên MOSI ở mức ngẫu nhiên (bacc 49,4).
- **Audio và face mất cả hướng phân biệt** (AUC chéo 52–58), không chỉ lệch ngưỡng. Riêng 7 logit của bộ FER pretrained chuyển giao tốt ở chiều MOSI→SIMS (TR +0,69).
- **Translate-test cho text:** TR +0,71 (MOSI→SIMS) và +0,83 (SIMS→MOSI).
- **Trong domain:** face trên MOSI ở mức ngẫu nhiên (49,4), còn face trên CH-SIMS đạt 71,2. Điều này khớp với nhận xét của EGMF (2601.07565) rằng dataset tiếng Trung phụ thuộc nhiều hơn vào audio và visual.

## Hệ quả
- **D (text bị shift nặng nhất, tín hiệu phi ngôn ngữ làm điểm neo): ĐÃ LOẠI** (C1: premise bị bác bỏ bằng thí nghiệm).
- **F4, phần chéo ngôn ngữ: không được ủng hộ.** Audio và face mất hướng phân biệt (ATR ≈ TR, AUC gần 50), không phải chỉ đổi độ lớn. Phần posed → tự nhiên chờ P0b (CREMA-D → MELD).
- **Phát hiện phụ:** con số thất bại chéo ngôn ngữ trong CASP (mô hình nguồn 39,17% ở MOSI→SIMS) đi kèm với việc dùng hai BERT riêng cho hai ngôn ngữ. Với encoder đa ngôn ngữ dùng chung cộng chuẩn hóa theo dữ liệu đích, text chuyển giao gần trọn vẹn (TR 0,87–0,95). Không so trực tiếp được với CASP vì metric khác nhau (ACC so với balanced accuracy).

## P0b: CREMA-D (posed, 91 diễn viên, 6/12 câu) → MELD (Friends, dev dùng làm train oracle, test 2.610 → 2.329 sau khi bỏ surprise)
- **Data:** 4.173 clip CREMA-D (train/valid/test theo diễn viên: 2.931/414/828) và 3.287 utterance MELD, 6 lớp chung.
- **Kết quả** (`p0b_report_2026-11-21.md`):
  - WavLM: TR +0,45, ATR +0,82, ATR − TR = +0,37 [+0,19; +0,54], hệ số g = −0,11 [−0,16; −0,05].
  - Face emb: oracle MELD gần mức ngẫu nhiên (21,0 so với 16,7).
  - Audio + Face: TR +0,88 (oracle MELD thấp: 29,1).
- **Chẩn đoán mô tả, làm sau khi thấy kết quả** (`p0b_diag.py`, `p0b_diag_2026-11-21.txt`):
  - Mô hình WavLM dự đoán neutral cho **5%** clip MELD, trong khi neutral thực tế chiếm **54%**.
  - Hiệu chỉnh prior bằng EM (Saerens 2002) làm bacc giảm (23,0 → 18,8). Shift vì vậy không phải lệch prior thuần túy, cũng không phải co cường độ.
  - Bên trong CREMA-D, câu IEO, biên logit tăng dần LO < MD < HI (audio 2,75/3,36/4,75; face 2,51/2,88/3,39) cùng với bacc (audio 40,0/58,9/72,2).
- **Kết luận F4: ĐÃ LOẠI.** P0 (chéo ngôn ngữ) cho thấy audio và face mất hướng phân biệt. P0b cho g âm và cho thấy hiệu chỉnh prior hay gain không gỡ được. MELD là phim diễn, nên phép thử posed → tự nhiên cũng bị nhiễu.

## P0c: khái niệm cảm xúc so với embedding (ứng viên H)
- **Khái niệm:**
  - audio: `superb/wav2vec2-base-superb-er` (IEMOCAP, 4 lớp);
  - face: 7 logit FER;
  - text: `cardiffnlp/twitter-xlm-roberta-base-sentiment`.
- **Tiêu chí cố định trước:** docstring của `p0c_analyze.py`.
- **Kết quả MOSI↔SIMS** (`p0c_report_2026-11-21.md`):
  - Face MOSI→SIMS: TR(khái niệm) − TR(emb) = +0,60 [+0,24; +1,00].
  - Audio-khái niệm gần như không mang tín hiệu sentiment kể cả trong domain (51,4 trên MOSI, 53,4 trên SIMS); hiệu −1,31 [−4,44; +0,22].
  - SIMS→MOSI: không xác định (oracle gần mức ngẫu nhiên).
  - Kết quả: 0/2 hướng ủng hộ. Tối đa chỉ đạt 1/3 dù CREMA-D→MELD ra sao, nên **H theo quy tắc: DOES NOT HOLD → ĐÃ LOẠI**. Phần P0c của CREMA-D/MELD đã dừng vì không đổi được kết luận.
- **Mô tả:** fusion trên khái niệm có bacc chéo 76,4 (MOSI→SIMS) và 78,4 (SIMS→MOSI), so với 68,6 và 63,1 của fusion trên embedding. Hiệu ứng chủ yếu đến từ bộ phân loại sentiment text đa ngôn ngữ có sẵn (TR 0,99/1,00), tức là chuyển giao zero-shot của một mô hình train cho đúng task. Không so trực tiếp được với CASP hay MIDG vì metric và cách xử lý nhãn 0 khác nhau.

## P1: gating theo mẫu học từ utility out-of-fold (ứng viên R2-2′)
- Script `p1_gating.py`, báo cáo `p1_report_2026-11-21.md`. Kết quả 1/4 dataset ủng hộ theo tiêu chí đặt trước → **R2-2′ ĐÃ LOẠI**. Chi tiết ở `ideas/round3_screening_2026-11-21.md`.

## P2: xóa thông tin corpus khỏi feature phi ngôn ngữ (quét cơ chế từ miền khác, 2026-11-21)
- **Cơ chế chuyển từ miền khác:** LEACE (xóa khái niệm tuyến tính, Belrose et al. 2023, NLP) với khái niệm = nhãn corpus; CORAL (căn chỉnh hiệp phương sai, Sun et al. 2016, vision DA). Bộ xóa fit trên feature **không nhãn** gộp từ train nguồn và phần không-test của đích; không dùng nhãn cảm xúc.
- **Tiêu chí cố định trước** (docstring `p2_erasure.py`): HELPS khi, với audio (WavLM) hoặc face (ViT-FER emb), TR(method) − TR(none) ≥ +0,10, CI bootstrap theo nhóm không chứa 0, ở ≥ 2/3 hướng {MOSI→SIMS, SIMS→MOSI, CREMA-D→MELD}, và bacc trong domain nguồn giảm ≤ 2 điểm.
- **Kết quả** (`p2_report_2026-11-21.md`): **0/3 hướng → DOES NOT HELP.**
  - MOSI→SIMS: audio LEACE −0,00 [−0,16; +0,19], CORAL −0,09; face LEACE +0,07 [−0,03; +0,17], CORAL +0,02.
  - SIMS→MOSI: audio LEACE −0,03, CORAL +0,20 [−0,01; +0,43]; face không xác định (oracle MOSI ở mức ngẫu nhiên).
  - CREMA-D→MELD: audio LEACE +0,14 [−0,07; +0,33], CORAL +0,03; face không ổn định (oracle 20,0 so với mức ngẫu nhiên 16,7).
- **Khớp với bằng chứng ngoài miền:** arXiv 2610.13027 (MRI não; mới đọc abstract) báo rằng xóa tín hiệu dataset bằng INLP, LEACE, CORAL hay adversarial "never improves" chuyển giao, vì nội dung được mã hóa theo hướng khác nhau ở mỗi dataset. arXiv 2610.08069 (mới đọc abstract) chứng minh giới hạn minimax: các bộ xóa tuyến tính (MP, SAL, LEACE) chỉ gỡ được độ lệch trung bình đo được từ dữ liệu hiệu chỉnh. P0 của mình (MOSI↔SIMS) cũng cho thấy audio và face mất hướng phân biệt khi đổi corpus (AUC chéo 52–58), nên một phép chiếu không nhãn không khôi phục được hướng đó.
