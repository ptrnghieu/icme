# Phân tích lý do loại ý tưởng (2026-11-21)

Nguồn: `ideas/triage_user_survey.md` (đợt 1), hồ sơ A/B/C, `ideas/round2_screening_2026-11-21.md` (đợt 2), `ideas/round3_screening_2026-11-21.md` (đợt 3, đang làm).

Mỗi hướng được xếp theo **gate đầu tiên nó trượt**. Một hướng trượt nhiều gate được xếp vào gate chính; gate phụ ghi trong ngoặc. "COLING" là ràng buộc dual submission, nằm ngoài C1–C5.

## Đợt 1 (19 mục survey + A, B, C; ràng buộc lúc đó: chỉ data mở)
| Gate | Số hướng | Các hướng |
|---|---|---|
| **C5 (data)** | **9** | #7 CHIARO (phải tự dựng data), #8 MEmoR (gated), #9 (gated; COLING), #10 Mult²EMo (gated), #11 MECES (gated), #13 OSMO (không có link tải), #14 ParaS2S (cần nhãn mới), #15 MER-Bench (cần nhãn mới), #19 ETC (cần nhãn mới) |
| C1 (gap) | 4 | #12 DAR, #16 EEmo-Logic (C4), C/VEATIC (hiện tượng hiếm), A (H không được ủng hộ trong thí nghiệm can thiệp) |
| C2 (độ rộng) | 3 | #1 ReCANVo (1 dataset, thiếu power), #4 ASR/TESS (2 người nói), B/AFFEC (không có dataset thứ hai) |
| C3 (publish) | 3 | #2 Blackwell (phân tích, hiệu ứng nhỏ), #3 định lý n+1 (không có thực nghiệm), A′ (user loại: không phải phương pháp) |
| C4 (novelty) | 1 | #17 RESCUE-Bench |
| COLING | 1 | #6 (forecasting) |

## Đợt 2 (25 hướng; đã chốt "phương pháp mới")
| Gate | Số hướng | Các hướng | Dạng lý do |
|---|---|---|---|
| **C4 (novelty)** | **13** | R2-1 LALM prosody, R2-2 gating OOF, R2-6 perspectivist (C2), R2-7 MLLM imbalance, E1 ASR, E3 semi-sup MSA, E4 khử khẩu hình, E6 nhãn nhiễu (C3), E7 emotion cause, F1 hallucination, F3 federated, K1 TTS, K3 label shift | **6 vùng bão hòa** (≥10 bài trong 2025–2026: R2-1, R2-7, E7, F1, E3, E1). **7 bị làm cốt lõi ở chỗ khác**: TIER-MoE (y sinh, 2026), Mariooryad & Busso (TAFFC 2016), Amanda (ACL 2024), Lin et al. (S2ST cho SER, 2024), MUG/DGDA, annotator modeling cho SER, FL thiếu modality |
| C3 (publish) | 6 | R2-3 hiệu chỉnh theo người nói (khai thác thiết kế dataset), R2-4 chọn frame (hiệu ứng nhỏ), E5 circumplex (C4), F2 continual, G2 timing, K2 head motion | Hiệu ứng nhỏ, không hợp lệ, hoặc không phải phương pháp |
| **C1 (gap)** | **4** | R2-5 kênh biểu cảm theo người (ICC = 0,016), D chéo ngôn ngữ (P0: ngược chiều), F4 expressivity (P0/P0b), H khái niệm (P0c: 0/2) | **Cả 4 đều bị loại bằng thí nghiệm của chính mình**, với tiêu chí đặt trước |
| C2 | 1 | E2 fairness (C3) | |
| C5 | 1 | M1 music (0 cặp trái cảm xúc trong bản phát hành) | |

## Đợt 3 (đang làm; data: MER2023–2026 + data mở)
| Hướng | Gate đang vướng |
|---|---|
| R3-1 MER-Cross | Chưa loại. Rủi ro: COLING, C2 (1 tập test), C4 (RASG), C5 (nhãn test chưa rõ) |
| R3-2 MER-SEMI/NOISE | C4 (đông bài dự thi) |
| R3-3 MER-Prefer | C4/C2 |

## Tổng (đợt 1 + đợt 2, 47 hướng)
| Gate | Đợt 1 | Đợt 2 | Tổng |
|---|---|---|---|
| C4 | 1 | 13 | **14** |
| C5 | 9 | 1 | **10** |
| C1 | 4 | 4 | **8** |
| C3 | 3 | 6 | **9** |
| C2 | 3 | 1 | 4 |
| COLING | 1 | 0 | 1 |

## Đánh giá: cái phễu
1. **Đợt 1 chết chủ yếu ở C5**, vì ràng buộc "chỉ data mở" và vì phần lớn ý trong survey cần data gated hoặc nhãn mới. Ràng buộc này nay đã được nới một phần (có MER2023–2026).
2. **Đợt 2 chết chủ yếu ở C4** khi đi theo hướng **cơ chế trước** (lấy một cơ chế ML tổng quát rồi áp vào cảm xúc). Lý do là cơ chế đó gần như luôn đã có người làm, thường ở miền lân cận.
3. Để tránh C4, mình chuyển sang hướng **hiện tượng trước** (tìm một hiện tượng riêng của cảm xúc rồi dựng phương pháp quanh nó). Cả 4 hiện tượng đem ra thí nghiệm đều **không tồn tại trong dữ liệu**, nên chết ở C1.
4. Những hiện tượng **đứng vững** (Q1 của A: cân bằng làm mức khớp với người tăng theo liều; ba lỗi protocol; text chuyển giao tốt nhất qua ngôn ngữ) đều chỉ mang tính **mô tả**, nên chết ở C3, vì không phải phương pháp.
5. **Chuẩn C4 đang áp dụng chặt:** một cơ chế đã có ở miền khác (ví dụ TIER-MoE ở y sinh) bị tính là scoop. Đây là một nguyên nhân khiến C4 chiếm tỷ trọng lớn nhất.
