# G8: xung đột giữa các modality trên dữ liệu thật (tầng 3, 2026-11-21) → **ĐÃ LOẠI (C4)**

## Claim cốt lõi (viết trước khi tìm)
Trên dữ liệu thật, một modality tự tin nhưng sai (thường là text) kéo fusion sai. Có thể dùng nhãn unimodal của người (CH-SIMS, CH-SIMS v2, phiếu cảm nhận theo modality của CREMA-D) để đo hiện tượng này và để huấn luyện một cơ chế phân xử: lúc nào gộp các modality, lúc nào tin modality nào.

## Kịch bản bị scoop
| # | Kịch bản | Kết quả kiểm tra |
|---|---|---|
| S1 | Đã có tập test "xung đột" dựng từ nhãn unimodal của CH-SIMS/v2, cho thấy mô hình MSA suy giảm | **Có.** DiffEmo (arXiv 2406.03004): "we resample CH-SIMS v2.0 dataset and make a DiffEmo Dataset, consisting of three different test settings, namely Mixed Set, Conflicting Set and Aligned Set"; "all models have a severe degradation compared between Conflicting Set and Aligned Set" |
| S2 | Đã có phương pháp fusion theo độ tin cậy từng mẫu, dựa trên bất đồng giữa các modality, đánh giá trên tập con xung đột thật | **Có.** UMC (2510.20256, 13 trang): khôi phục pseudo-label unimodal liên tục và dựng consensus theo độ tin cậy; Phụ lục H đánh giá trên D_msc/D_msi của CH-SIMS và CH-SIMS v2, "These test subsets distinguish whether the unimodal sentiment annotations agree or disagree" |
| S3 | Đã có phương pháp học khi nào gộp, khi nào bỏ một modality xung đột | **Có.** DCR (2605.04877, abstract): "learns when to fuse and when to drop modalities", contextual bandit chọn giữa fusion và dự đoán unimodal, 5 benchmark. MiD-ERC (2611.08860, abstract): gating, trọng số theo độ tin cậy và prompt LLM, đều dựa trên mức bất đồng giữa các modality |
| S4 | Đã có benchmark và phương pháp xung đột cho MLLM cảm xúc | **Có.** MoSEAR / CA-MER (2508.01181, ACM MM 2025 Oral): tập con video-aligned / audio-aligned / consistent; MLLM "systematically over-rely on audio signal during emotion conflicts" |
| S5 | Đã có phương pháp dùng nhãn unimodal (thật hoặc học được) để giám sát nhánh unimodal | **Có.** MUG (2408.16029); nhãn unimodal của CH-SIMS và AV-MC của CH-SIMS v2 (2209.02604) |

Phần còn lại chưa ai làm chỉ là "giám sát module phân xử bằng nhãn unimodal thật". DCR và MiD-ERC nói rõ không cần nhãn này. [Đánh giá] Phần này **không đủ làm đóng góp chính**, vì ba lý do:
1. Chỉ áp dụng được cho dataset có nhãn unimodal, mà CH-SIMS ⊂ CH-SIMS v2 (xem dưới).
2. Thí nghiệm can thiệp của ứng viên A trên CREMA-D (dùng nhãn unimodal của người làm trọng số theo clip) không làm tăng accuracy: Δacc +0,18 [−1,56; +1,92].
3. Reviewer sẽ so ngay với UMC và DCR.

## Data và power (đã đo, ghi lại để dùng về sau)
- **CH-SIMS v2(s)** (mirror HF `tamb2203579/CH-SIMSv2`, `CH-SIMS v2(s)/meta.csv`, tải HTTP 200, 440.396 byte):
  - 4.403 clip (train 2.722 / valid 647 / test 1.034); có cột `label_T`, `label_A`, `label_V`.
  - Test có 911 clip khác 0; trong đó 172 (18,9%) có cực tính text khác nhãn đa modality, và 104 clip có text sai nhưng A và V đúng.
- **CH-SIMS nằm trọn trong CH-SIMS v2:** 2.281/2.281 clip của CH-SIMS có trong v2, cùng text, cùng nhãn, cùng split. Hai bộ này **không phải hai dataset độc lập** khi tính C2.

## Kết luận
- C1: PASS (bằng chứng ở S1 và ở số đếm trên).
- C4: **FAIL** (S1–S4 đều đã có trong affective computing).
- → **ĐÃ LOẠI** ở tầng 3.
