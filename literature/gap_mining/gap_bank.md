# Ngân hàng gap (tầng 1–2, quy trình v2; 2026-11-21)

## Nguồn đã thu hoạch
| Nguồn | Phạm vi | File |
|---|---|---|
| (b) Limitation, future work, conclusion của bài đã được nhận | **170 bài** arXiv nộp từ 01/2025 đến 21/11/2026, có comment ghi venue đã nhận: số lần tên venue xuất hiện trong comment: ICASSP 31, Interspeech 37, CVPR 15, EMNLP 12, AAAI 12, ACM MM 12, ICME 6, ICLR 7, NeurIPS 6, ACL 6, … Tải đủ 170 PDF; trích tự động bằng `tools/gap_harvest.py`; 87 bài có câu limitation hoặc future work | `accepted_2025_2026.tsv`, `limitations_condensed.md` |
| (d) Phát hiện của các bài chẩn đoán và benchmark | Đọc conclusion của 15 bài chẩn đoán trong 170 bài | Trích dẫn ở bảng dưới |
| Độ bão hòa | 18 truy vấn arXiv, nộp từ 01/2025 đến 21/11/2026, tối đa 200 kết quả mỗi truy vấn | `saturation_counts_2026-11-21.txt` |
| (c) Phê bình của reviewer trên OpenReview | Tìm được 293 forum về emotion, sentiment, affective (ICLR 2025–2026, NeurIPS 2025) qua `/notes/search` (HTTP 200) | **BỊ CHẶN:** `api2.openreview.net/notes?forum=<id>` trả **HTTP 403** từ container (thử ngày 2026-11-21), nên không đọc được review. Danh sách forum lấy được bằng `tools/openreview_reviews.py`. |

**Nhận xét [Đánh giá]:** phần limitation do tác giả tự viết chủ yếu chung chung ("mở rộng sang modality khác", "khám phá backbone lớn hơn"). Tín hiệu hữu ích nằm ở **phát hiện của các bài chẩn đoán**, và ở **những vấn đề còn ít bài giải** sau khi đếm độ bão hòa.

## Các cụm, độ bão hòa và rubric (tầng 2)
Rubric (CLAUDE.md §4.6), mỗi mục 0–2: B = bằng chứng · Đ = đánh giá được bằng tài sản hiện có · M = chưa bão hòa · P = dẫn tới phương pháp · E = hiệu ứng dự kiến lớn hơn MDE · K = vừa 6 trang, kịp hạn, không trùng COLING. **Giữ khi tổng ≥ 9 và không mục nào bằng 0.**

| # | Cụm | Số bài arXiv 2025–26 | Bằng chứng chính (trích nguyên văn) | B | Đ | M | P | E | K | Tổng | Kết luận |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 | Text hoặc ngữ nghĩa lấn át âm học; xung đột âm học–ngữ nghĩa | 28 (cộng ≥ 10 phương pháp đã ghi ở đợt 2, R2-1) | 2608.28966: "LALMs struggle with acoustic–semantic incongruence and exhibit a tendency to underweight acoustic cues when semantic content is neutral"; 2510.25054: Cramér's V 0,08 (nhãn đích) so với 0,65 (nhãn proxy theo text); 2610.20114: "traditional fusion catastrophically collapses under polarity conflicts" | 2 | 1 | 0 | 2 | 1 | 2 | 8 | Loại (M = 0) |
| G2 | Thiếu modality | 47 | Nhiều phương pháp: SemMSA, MIDAS, TLRA, TCMR, PriMD, TMDC, BALM; benchmark MissMAC-Bench | 2 | 2 | 0 | 2 | 1 | 2 | 9 | Loại (M = 0) |
| G3 | Nhãn mơ hồ, phân phối nhãn | 184 | 2606.27536: "high-entropy utterances remain challenging under hard categorical evaluation" | 2 | 1 | 0 | 2 | 1 | 2 | 8 | Loại |
| G4 | Suy luận cảm xúc bằng MLLM với RL | 84 | EMO-R3, AVERE, OPPO, EmotionThinker | 1 | 0 | 0 | 2 | 1 | 1 | 5 | Loại (cần RL trên 7B) |
| G5 | Test-time adaptation | 5 | 2601.16240: TTA "provides only marginal gains for cases such as adapting from acted to natural emotions" | 1 | 2 | 2 | 2 | 0 | 2 | 9 | Loại (E = 0: P2–P5 cho thấy thích nghi không nhãn không có headroom trên các cặp corpus của mình; nhánh này đã dừng) |
| G6 | MER gọn nhẹ, distillation | 34 | 2607.12787: mô hình dưới 1B vượt teacher 8B trên 9 benchmark | 1 | 1 | 1 | 2 | 1 | 2 | 8 | Loại |
| G7 | Fairness, bias trong SER | 14 | 2603.21050: "multimodal input does not reliably reduce bias even when it improves SER performance" | 2 | 1 | 1 | 2 | 0 | 2 | 8 | Loại (mục tiêu không phải accuracy trung bình, không khớp C3) |
| **G8** | **Xung đột giữa các modality trên dữ liệu thật** (một modality tự tin nhưng sai kéo fusion sai) | 11 | 2511.02794: "modality sabotage, a failure mode in which a high-confidence unimodal error misleads the fused prediction"; 2610.20114 (ở trên). **Tài sản riêng:** CH-SIMS có nhãn unimodal T/A/V. Mình đếm: **533/1.936 clip (27,5%)** có cực tính text khác cực tính nhãn đa modality, trong đó **357** clip có A và V đúng còn T sai | 2 | 2 | 1 | 2 | 1 | 2 | **10** | **Giữ → tầng 3** |
| G9 | Cảm xúc nhóm, nhiều người | 14 | 2609.11164: "identifying which visual cues provide reliable complementary information to audio remains an open problem" | 2 | 0 | 1 | 2 | 1 | 1 | 7 | Loại (không có data) |
| G10 | MER từ vựng mở, chi tiết | 147 | MER2025 track FG, AffectGPT | 1 | 1 | 0 | 2 | 1 | 1 | 6 | Loại |
| G11 | Valence–arousal liên tục, đa modality | 19 | 2604.07198 (RECOLA, SEWA) | 1 | 0 | 1 | 2 | 1 | 2 | 7 | Loại (data cần EULA) |
| G12 | Cá nhân hóa | 73 | 2505.16220 Meta-PerSER | 1 | 1 | 0 | 2 | 1 | 2 | 7 | Loại |
| G13 | MER với transcript ASR | 5 | 2609.20081: WER 32,38% trên MELD | 2 | 2 | 1 | 2 | 1 | 2 | 10 | Loại (C4: SWRM 2203.00257, 2406.08353 đã làm cốt lõi trong affective; xem E1 ở đợt 2) |
| G14 | Nhãn nhiễu | 18 | MUG, DGDA-Net | 1 | 1 | 1 | 2 | 0 | 2 | 7 | Loại (E6 ở đợt 2) |
| G15 | Calibration và độ bất định | 37 | EmoEUS, SURE, 2610.13013 | 1 | 2 | 0 | 2 | 1 | 2 | 8 | Loại |
| G16 | Định vị thời điểm cảm xúc đổi | 37 | 2605.25409 | 1 | 0 | 1 | 2 | 1 | 1 | 6 | Loại |
| G17 | Biểu cảm tinh tế, cường độ thấp | 48 | — | 1 | 1 | 0 | 2 | 1 | 2 | 7 | Loại |
| G18 | Data tổng hợp cho MER | 139 | 2608.28966 (TTS) | 1 | 1 | 0 | 2 | 1 | 2 | 7 | Loại |

## Kết quả tầng 2
- **Chỉ G8 qua rubric (10/12).** G13 đạt 10 điểm nhưng trượt C4 (đã có tiền lệ cốt lõi trong affective).
- **Việc tiếp theo cho G8 (tầng 3):**
  - viết claim cốt lõi và kịch bản bị scoop;
  - kiểm novelty trong affective. Tiền lệ đã biết cần đối chiếu: AV-MC của CH-SIMS v2 (2209.02604, đã ghi trong reading log: "dùng mixup A/V để chống text-predominance"), và các phương pháp xung đột modality cho LALM ở R2-1;
  - phác thảo 2–3 phương pháp;
  - **power:** split test của CH-SIMS chỉ có **100 clip xung đột** (25,8% của 388), trong đó 64 clip có A và V đúng còn T sai. Với n = 100, sai số chuẩn của accuracy khoảng 5 điểm, nên một mình CH-SIMS không đủ power. Cần thêm CH-SIMS v2 (theo reading log, 4.402 mẫu có nhãn unimodal), hiện mới ở L1+.
