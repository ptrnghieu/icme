# Quét cơ chế từ miền khác (đợt 3, 2026-11-21)

User yêu cầu (2026-11-21): loại MER-Cross, quét các cơ chế đã có ở miền khác (C4 đã nới: chỉ tính scoop nếu ý tưởng cốt lõi đã có trong MER/affective computing).

**Cách làm:** ghép từng cơ chế với một **thất bại đã đo được** trong MER (từ pilot của mình), kiểm tra tiền lệ trong affective computing trên arXiv (truy vấn ở `literature/search_logs/round3_2026-11-21/r3_mech.txt`, `r3_erasure.txt`, `r3_selection.txt`), rồi chạy **phép thử premise rẻ** với tiêu chí cố định trước trên harness feature đông cứng (`explore/xling/`).

## Các thất bại đã đo được (đích của cơ chế)
| Mã | Thất bại | Bằng chứng |
|---|---|---|
| F1 | Audio và face mất hướng phân biệt khi đổi corpus | P0: AUC chéo 52–58 (MOSI↔SIMS); P0b: TR WavLM +0,45 (CREMA-D→MELD) |
| F2 | Chọn mô hình không ổn định khi valid nhỏ hoặc có shift | Pilot A: chọn checkpoint theo test thổi phồng +1,65 điểm; MELD valid 184 mẫu (P1: B2 tệ hơn early concat) |
| F3 | Headroom bổ trợ theo mẫu lớn nhưng gate không lấy lại được | P1: oracle +10 đến +35, gate OOF lấy lại 0–1 điểm |
| F4 | Lệch lớp neutral giữa corpus | P0b: mô hình dự đoán neutral 5% so với 54% thực tế; hiệu chỉnh prior EM làm tệ hơn |

## Bảng cơ chế
| Cơ chế (miền gốc) | Thất bại đích | Tiền lệ trong affective computing | Phép thử premise | Kết luận |
|---|---|---|---|---|
| Xóa khái niệm tuyến tính LEACE/INLP (NLP fairness); CORAL (vision DA) | F1 | Không thấy bài xóa nhãn corpus bằng LEACE/INLP trong affective (truy vấn `r3_erasure.txt`). Căn chỉnh thống kê kiểu CORAL đã phổ biến trong cross-corpus SER. | **P2: 0/3 hướng** (`explore/xling/p2_report_2026-11-21.md`) | **ĐÃ LOẠI (C1).** Khớp với 2610.13027 (MRI não, abstract): xóa tín hiệu dataset "never improves" chuyển giao |
| Agreement-on-the-Line, ALine-D (Baek et al. 2022); ATC (Garg et al. 2022): chọn mô hình không cần nhãn đích | F2 | 0 kết quả cho model selection hay accuracy estimation không nhãn trong emotion (`r3_selection.txt`). 2607.27655 (EEG) bàn về chọn checkpoint nhưng **có** dùng nhãn đích → không trùng | **P3** (`explore/xling/p3_selection.py`) | Xem mục P3 bên dưới |
| Gating/stacking học từ utility out-of-fold (TIER-MoE, y sinh) | F3 | CONFER, PDF, QMF: trùng một phần. Atsuko (2603.13340, abstract): router của MER được giám sát bằng "marginal complementarity" | P1: 1/4 | **ĐÃ LOẠI** (đợt 3, R2-2′) |
| Model merging / task arithmetic (NLP, vision) | Đa corpus | AdaLTM (2603.25041, SER): merge task vector theo layer | Không chạy | **ĐÃ LOẠI (C4):** cơ chế đã có trong SER |
| DRO (Sagawa et al. 2020) theo nhóm người nói | Chênh lệch giữa người nói | HDF (2507.15765, ACM MM 2025, DFER, abstract) dùng khung "distributional" để cân bằng mẫu khó, không phải group DRO theo người | Không chạy | Ưu tiên thấp (C3): group DRO cải thiện nhóm tệ nhất, thường đổi bằng accuracy trung bình, nên không đáp ứng yêu cầu cải thiện nhất quán |
| WiSE-FT, LP-FT (robust fine-tuning; vision) | F1 | 0 kết quả trong abstract affective. CLAIP-Emo (2509.14527, abstract): LoRA trên CLIP/CLAP đông cứng, không nội suy trọng số | Không chạy | Ưu tiên thấp: đầu zero-shot nonverbal (logit FER, superb-er) đã được thử ở P0c (H, 0/2) |
| MixStyle / nhiễu thống kê feature (vision DG) | F1 | 0 kết quả trong abstract affective | Không chạy | Ưu tiên thấp: cùng họ với CORAL (P2 không giúp) và có cùng giới hạn khi hướng phân biệt khác nhau giữa corpus |
| IRM (Arjovsky et al. 2019) | F1 | 0 kết quả trong abstract affective | Không chạy | Ưu tiên thấp, cùng lý do |
| Learning to defer | F3 | 0 kết quả trong abstract affective | Không chạy | Ưu tiên thấp: P1 cho thấy tín hiệu confidence không dự đoán được modality nào đúng |

## P3: chọn mô hình không cần nhãn đích
(đang chạy; điền sau khi có kết quả)
