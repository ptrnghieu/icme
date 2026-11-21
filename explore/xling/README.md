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
