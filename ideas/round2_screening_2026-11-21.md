# Đợt tìm ý tưởng thứ hai: nhật ký sàng lọc (2026-11-21)

## Thanh đo
User chốt ngày 2026-11-21 (CLAUDE.md §3, C3): **đóng góp chính phải là một phương pháp mới**, cải thiện nhất quán trên ≥2 dataset mở so với baseline 2024–2026 có code.

Ràng buộc:
- Chỉ dùng data mở. Compute là Kaggle 2×T4, mỗi session tối đa 11 giờ.
- Loại mọi hướng có lõi là forecasting trong hội thoại, vì trùng bài COLING (Hi-EF, arXiv 2407.16406).

## Cách làm
- **arXiv API:** khoảng 50 truy vấn. Kết quả thô lưu ở `literature/search_logs/round2_2026-11-21/`:
  - `round1.txt`, `round2.txt`: quét theo nhánh;
  - `nov_*.txt`: sàng lọc novelty theo từng hướng;
  - `abs_*.txt`: abstract các paper gần nhất.
- **Mức đọc:**
  - Sàng lọc chủ yếu dựa trên **abstract**.
  - Đọc toàn văn: Kaffeza et al., *The Illusion of Balanced MSA* (arXiv 2609.11247, Interspeech 2026).
  - Đọc phần method: TIER-MoE (arXiv 2607.27289).
- **Ý nghĩa của "loại" trong file này:** loại ở bước sàng lọc. Hướng nào cần kết luận scoop chính thức theo §4.2 thì đã ghi rõ mức đọc.

## Bức tranh chung (sự thật từ truy vấn)
Mỗi truy vấn dưới đây sắp theo ngày và trả về 120 kết quả (giới hạn của truy vấn). Kết quả cũ nhất của từng truy vấn cho biết 120 bài đó trải dài bao lâu:

| Truy vấn (`abs:`) | 120 kết quả mới nhất trải từ khoảng |
|---|---|
| "multimodal sentiment analysis" | tháng 5/2026 |
| "speech emotion recognition" | tháng 5/2026 |
| "multimodal emotion recognition" | tháng 5/2026 |
| "emotion recognition in conversation" | tháng 10/2025 |

Các chủ đề lặp lại nhiều nhất từ tháng 5/2026:
- missing modality / incomplete;
- fusion theo độ tin cậy, MoE và routing;
- xung đột giữa modality;
- adapter LLM;
- LALM bị text lấn át.

Riêng `abs:"music emotion recognition"`: chỉ có **49 kết quả cho mọi thời điểm**.

## Các hướng đã sàng lọc
| # | Hướng | Kết luận | Lý do và bằng chứng |
|---|---|---|---|
| R2-1 | LALM bị text lấn át trong SER. Ý tưởng: tạo "góc nhìn chỉ có prosody" bằng content-masked speech (low-pass, random splicing) rồi kết hợp lúc inference, không train. | **ĐÃ LOẠI** (C3/C4, không gian bão hòa) | Đến 11/2026 đã có ≥7 benchmark: DEAF 2603.18048, ALME 2602.11488, ASPIRE 2610.20114, TWIN-SER 2609.04236, CASE 2601.04564, CREMA-ASIS 2608.28966, LISTEN-style 2610.25599. Đã có ≥10 phương pháp: FAS, DAS, ACR-Net (train); back-patching 2606.18924; steering bằng audio-specialist heads 2603.06854; can thiệp hidden-state 2608.19211; acoustic cue tokens 2606.07309; contrastive decoding 2506.07233, 2603.09232, 2604.15383, 2610.28146. Đặc biệt **2607.00247** (SLT 2026) đã khảo sát cả thư viện biến đổi audio (temporal, spectral, frequency, amplitude) làm nhánh âm. Cơ chế content-masking không bị scoop trực tiếp, nhưng reviewer sẽ coi là thêm một mẹo inference giữa hàng chục mẹo. |
| R2-2 | Học trọng số fusion theo mẫu từ hiệu năng held-out hoặc out-of-fold. Đây là hướng mở mà bài Illusion nêu ở §4.3.3, kèm headroom oracle theo mẫu +8,70 (MOSI) và +7,79 (MOSEI) điểm Acc-2. | **ĐÃ LOẠI** (C4) | **TIER-MoE** (2607.27289, đọc phần method): "cross-fitted conditional modality risk", estimator học từ Brier loss out-of-fold theo subject rồi dùng để route giữa các modality. Đây đúng là ý cốt lõi, chỉ khác miền (y sinh). **CONFER** (2608.07867, abstract) cũng dùng độ tin cậy ước lượng từ hiệu năng out-of-fold. |
| R2-3 | Hiệu chỉnh theo từng người nói cho LALM zero-shot SER, không cần nhãn. | **ĐÃ LOẠI** (C3, tính hợp lệ) | Trong corpus diễn xuất (CREMA-D, RAVDESS), mỗi diễn viên có đủ mọi lớp với số lượng đều. Hiệu chỉnh prior theo người nói vì thế khai thác thiết kế dataset, và reviewer sẽ bắt lỗi. Gần nhất: EMO-TTA 2509.25495, TTA cho SER 2601.16240 (ICASSP 2026). |
| R2-4 | Chọn frame giàu cảm xúc dựa vào audio, cho AV emotion có giới hạn số frame. | **ĐÃ LOẠI** (C3, hiệu ứng nhỏ) | Clip CREMA-D, RAVDESS và MELD dài 2–4 giây. Trong pilot, phần lớn mức +18,1 điểm của visual-only đến từ việc đổi frame đầu sang frame giữa, và lấy mẫu đều nhiều frame đã bao phủ clip. Bằng chứng gần nhất: 2604.15280 (ECCV 2026), về VLM với micro-expression trong video dài hơn. |
| R2-5 | Cá nhân hóa lúc test cho AV emotion. Giả thuyết làm sắc hướng này: kênh biểu cảm mạnh khác nhau theo từng người. | **ĐÃ LOẠI** (C1: giả thuyết bị bác bỏ bằng dữ liệu; C4: phần còn lại chỉ là gia tăng) | Phân tích của Claude (`ideas/evidence/person_channel_cremad_result.txt`): 91/91 diễn viên có r_face > r_voice; ICC(1) của d_human theo diễn viên là **0,016**; split-half của d theo diễn viên r = 0,10 (Spearman–Brown 0,19); oracle chọn modality theo diễn viên chỉ hơn **+0,5 đến +1,8 điểm**. Bỏ giả thuyết này thì hướng còn lại là mở rộng TTA-CaP 2603.21309, EB-CaP 2608.06467, CLIP-AUTT 2603.27999 (đều chỉ visual) và CASP 2412.07121 (TTA cho MSA) sang AV. |
| R2-6 | Mô hình hóa từng người chấm (perspectivist) cho cảm xúc đa phương thức, dùng CREMA-D, VEATIC, AFFEC. | **ĐÃ LOẠI** (C4/C2) | Đã có mô hình hóa từng annotator cho SER: 2408.11956 (Interspeech 2024), 2510.05934, 2611.01539. Nhãn theo người chấm *và* theo modality chỉ có ở CREMA-D. |
| R2-7 | MLLM bị mất cân bằng hoặc xung đột modality trong MER (video bị bỏ qua, phân vai theo layer). | **ĐÃ LOẠI** (bão hòa) | EmoMM/CHASE 2605.01024, LAMR 2611.11907, DCR 2605.04877, MiD-ERC 2611.08860, 2611.11987. |

## Bằng chứng giữ lại cho các vòng sau
- **Illusion** (2609.11247, đọc toàn văn):
  - Các phương pháp cân bằng dựa trên gradient hay loss không thắng late concat trên MOSI/MOSEI.
  - Chọn trọng số global theo val nằm trong khoảng 0,3–0,4 điểm so với oracle.
  - Headroom oracle theo mẫu: +8,70 (MOSI) và +7,79 (MOSEI).
  - §4.3.3 liệt kê bốn hướng mở; hướng "Robustness Profiling" chưa được sàng lọc.
- **TTA cho SER** (2601.16240, ICASSP 2026, abstract): entropy minimization và pseudo-labeling thất bại vì cảm xúc vốn mơ hồ; phương pháp không cần backprop là hứa hẹn nhất.
- **Phát hiện mới trên CREMA-D** (Claude tự chạy): mô hình audio-only thắng visual-only ở 78% diễn viên chưa thấy, trong khi với người, mặt luôn dễ nhận ra hơn giọng ở 100% diễn viên.

## Hướng còn mở, cần user quyết phạm vi
- **Music emotion recognition** (audio+lyrics, audio+symbolic, đa dataset). Nhánh này ít bài hơn hẳn: 49 kết quả arXiv cho mọi thời điểm.
- Các vấn đề mở đã được nêu:
  - khả năng tổng quát hóa giữa các dataset và nhãn không đồng nhất giữa dataset (survey 2406.08809; nghiên cứu khoảng cách phân phối 2510.04688);
  - ước lượng độ bất định của phản hồi chủ quan, "significantly more challenging" (2501.11570, ECIR 2025).
- Đã có phương pháp cho: thống nhất nhãn categorical và dimensional (2502.03979), căn nhãn bằng LLM embedding (2410.11522), train chung DEAM+PMEmo (2611.08966).
- ICME 2025 đã nhận một bài về music emotion (2409.15545).
- Ứng viên dataset mở (**chưa kiểm tra L2**): DEAM, PMEmo, EMOPIA, MERGE, MTG-Jamendo mood/theme, Memo2496.
