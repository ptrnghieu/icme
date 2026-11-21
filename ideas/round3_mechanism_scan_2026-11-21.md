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
| Agreement-on-the-Line, ALine-D (Baek et al. 2022); ATC (Garg et al. 2022): chọn mô hình không cần nhãn đích | F2 | 0 kết quả cho model selection hay accuracy estimation không nhãn trong emotion (`r3_selection.txt`). 2607.27655 (EEG) bàn về chọn checkpoint nhưng **có** dùng nhãn đích → không trùng | **P3: method 0/3** (premise headroom 2/3) | **ĐÃ LOẠI (C1)** |
| Gating/stacking học từ utility out-of-fold (TIER-MoE, y sinh) | F3 | CONFER, PDF, QMF: trùng một phần. Atsuko (2603.13340, abstract): router của MER được giám sát bằng "marginal complementarity" | P1: 1/4 | **ĐÃ LOẠI** (đợt 3, R2-2′) |
| Model merging / task arithmetic (NLP, vision) | Đa corpus | AdaLTM (2603.25041, SER): merge task vector theo layer | Không chạy | **ĐÃ LOẠI (C4):** cơ chế đã có trong SER |
| DRO (Sagawa et al. 2020) theo nhóm người nói | Chênh lệch giữa người nói | HDF (2507.15765, ACM MM 2025, DFER, abstract) dùng khung "distributional" để cân bằng mẫu khó, không phải group DRO theo người | Không chạy | Ưu tiên thấp (C3): group DRO cải thiện nhóm tệ nhất, thường đổi bằng accuracy trung bình, nên không đáp ứng yêu cầu cải thiện nhất quán |
| WiSE-FT, LP-FT (robust fine-tuning; vision) | F1 | 0 kết quả trong abstract affective. CLAIP-Emo (2509.14527, abstract): LoRA trên CLIP/CLAP đông cứng, không nội suy trọng số | Không chạy | Ưu tiên thấp: đầu zero-shot nonverbal (logit FER, superb-er) đã được thử ở P0c (H, 0/2) |
| MixStyle / nhiễu thống kê feature (vision DG) | F1 | 0 kết quả trong abstract affective | Không chạy | Ưu tiên thấp: cùng họ với CORAL (P2 không giúp) và có cùng giới hạn khi hướng phân biệt khác nhau giữa corpus |
| IRM (Arjovsky et al. 2019) | F1 | 0 kết quả trong abstract affective | Không chạy | Ưu tiên thấp, cùng lý do |
| Learning to defer | F3 | 0 kết quả trong abstract affective | Không chạy | Ưu tiên thấp: P1 cho thấy tín hiệu confidence không dự đoán được modality nào đúng |
| Gộp nhãn kiểu Dawid–Skene (crowdsourcing) và ước lượng accuracy từ agreement (Platanios et al. 2014): coi mỗi head unimodal là một "annotator" trên corpus đích không nhãn | F1, F4 | Trong affective, Dawid–Skene chỉ thấy dùng để gộp nhãn của người gán, không dùng cho head modality (chưa chạy truy vấn riêng vì P4 trượt) | **P4: 0/3** | **ĐÃ LOẠI (C1)** |
| Co-training / dạy chéo modality (Blum & Mitchell 1998; cross-modal self-training trong video DA): text (chuyển giao tốt) gán pseudo-label để train head phi ngôn ngữ **riêng cho corpus đích** | F1 | CASP (AAAI 2025) dùng pseudo-label cho mọi modality cùng lúc; S²LIF học text trước trong DG. Chưa kiểm novelty sâu vì P5 trượt | **P5: trượt** ở hướng duy nhất kiểm được | **ĐÃ LOẠI (C1)** |

## P3: chọn mô hình không cần nhãn đích (`explore/xling/p3_selection.py`, `p3_report_2026-11-21.md`)
- **Zoo:** mọi tập con khác rỗng của các khối feature (early concat) × C ∈ {0,001; 0,01; 0,1; 1}: 124 mô hình (MOSI↔SIMS) và 60 mô hình (CREMA-D→MELD). Pool không nhãn = phần không-test của corpus đích.
- **Tiêu chí cố định trước:** (P) headroom = mô hình tốt nhất trên test đích − S0 (chọn theo valid nguồn) ≥ 3 điểm ở ≥ 2/3 hướng; (M) ALine-D − S0 ≥ +2 với CI không chứa 0 ở ≥ 2/3 hướng.
- **Kết quả:**

| Hướng | Spearman(valid nguồn, test đích) | S0 | ALine-D | Agreement với đa số zoo (mô tả) | ATC (mô tả) | Tốt nhất trên test | Headroom | ALine-D − S0 |
|---|---|---|---|---|---|---|---|---|
| MOSI→SIMS | +0,86 | 65,6 | 66,7 | 68,4 | 61,0 | 69,4 | +3,7 | +1,0 [−2,5; +5,3] |
| SIMS→MOSI | +0,53 | 63,1 | 65,7 | 64,7 | 62,9 | 72,8 | +9,7 | +2,6 [−0,6; +5,7] |
| CREMA-D→MELD | +0,69 | 27,0 | 27,0 | 27,0 | 18,3 | 27,7 | +0,7 | +0,0 |

- **Premise 2/3 → HOLDS; method 0/3 → DOES NOT HOLD.**
- **Lưu ý [Đánh giá]:** headroom lấy max trên 124 ước lượng có nhiễu trên cùng một tập test, nên bị thổi phồng do chọn lọc. Headroom thật nhỏ hơn con số trong bảng.

## P4: modality như annotator độc lập, Dawid–Skene trên corpus đích (`p4_ds.py`, `p4_report_2026-11-21.md`)
- **Tiêu chí cố định trước:** posterior DS − S0 (late fusion, trọng số global chọn theo valid nguồn) ≥ +2 với CI không chứa 0 ở ≥ 2/3 hướng.
- **Kết quả: 0/3 → DOES NOT HOLD.** MOSI→SIMS −2,7 [−7,4; +1,7]; SIMS→MOSI −9,2 [−15,4; −3,1]; CREMA-D→MELD +0,6 [−3,0; +4,3].
- **Chẩn đoán (mô tả):** giả định độc lập có điều kiện bị vi phạm. Hai head face (embedding và logit) đồng thuận với nhau, nên DS ước lượng accuracy của chúng là 97,0/96,1 (SIMS→MOSI), trong khi accuracy thật chỉ 52,4/52,0. Trên MELD, mọi head đều dự đoán thiếu neutral: DS ước lượng prior neutral 0,013, trong khi thật là 0,539. **Lỗi do shift corpus tương quan giữa các modality**, vì mọi head đều học khái niệm nhãn của corpus nguồn.
- **Phát hiện phụ (mô tả):** ở cả hai hướng chéo ngôn ngữ, head text đơn lẻ tốt hơn late fusion có trọng số chọn theo valid nguồn: 67,3 so với 64,2 và 65,7 so với 61,8.

## P5: text làm neo, train head phi ngôn ngữ riêng cho corpus đích (`p5_anchor.py`, `p5_report_2026-11-21.md`)
- **Động cơ:** P0 cho thấy text chuyển giao tốt, còn audio và face mất hướng phân biệt, nên head phi ngôn ngữ phải được học trên chính corpus đích.
- **Tiêu chí cố định trước** (MOSI→SIMS là hướng duy nhất kiểm được, vì trên MOSI tín hiệu phi ngôn ngữ không thêm gì so với text ngay cả trong domain): M − T ≥ +2 với CI không chứa 0, M > B0 và M ≥ B1.
- **Kết quả MOSI→SIMS:** T (text, chuẩn hóa theo đích) 69,3; M 67,7; B0 67,6; B1 (pseudo-label từ fusion) 71,2; cận trên U (head phi ngôn ngữ train bằng nhãn thật của đích) 75,4. M − T = −1,6 [−5,4; +2,0] → **DOES NOT HOLD**.
- **SIMS→MOSI (mô tả):** M − T −2,2; cận trên U − T −3,9.
- **Chẩn đoán (mô tả):** pseudo-label giữ lại có bacc thật 73,8. Head phi ngôn ngữ học theo pseudo-label đạt 64,7 khi đứng riêng, nhưng nó bắt chước cả lỗi của text, nên ghép với text không bổ sung được gì. Khoảng cách +6,1 điểm đến U chỉ lấp được bằng **nhãn thật của corpus đích**.

## Kết luận của đợt quét
- **[Sự thật]** Bốn họ cơ chế **không dùng nhãn đích** đều trượt tiêu chí đặt trước trên cùng harness:
  - bất biến (P2, 0/3);
  - agreement để chọn mô hình (P3, 0/3);
  - độc lập có điều kiện giữa các modality (P4, 0/3);
  - dạy chéo từ modality chuyển giao tốt (P5, trượt).
  Cộng thêm P1 (gating OOF, 1/4) ở bài toán trong domain.
- **[Đánh giá]** Các lần trượt có chung một nguyên nhân đo được:
  - khi đổi corpus, hướng phân biệt của audio và face thay đổi (P0, P2);
  - lỗi tương quan giữa các modality (P4);
  - pseudo-label kéo theo lỗi của modality làm thầy (P5).
  Vì vậy cả nhóm cơ chế "sửa shift mà không có nhãn đích" khó có headroom thật trên các corpus này. Các cơ chế còn lại trong bảng (MixStyle, IRM, WiSE-FT, learning to defer) dựa trên cùng giả định nên được xếp ưu tiên thấp, không chạy tiếp.
