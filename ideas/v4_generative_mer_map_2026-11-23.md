# Quy trình v4, bước 1–4: MER sinh bằng MLLM (2026-11-23)

## 1. Nhận định về lĩnh vực
- [Sự thật] MER2026 (2604.19417, ACM MM 2026): "the focus of MER has shifted from discriminative emotion recognition to generative emotion understanding"; các track 2026 là MER-Cross, MER-FG (từ vựng mở), MER-Prefer, MER-PS.
- [Sự thật] Survey IJCAI 2026 (2605.21239), Future Directions: "it remains underexplored how to … evaluate open-ended interpretations"; "how subjectivity should be incorporated into models"; "which model parameters critically influence emotion perception".
- [Đánh giá] Vùng con ở biên chuyển đổi: **các luật quyết định và cách đánh giá cho MLLM cảm xúc**. Phương pháp hiện tại gần như đều là huấn luyện thêm (SFT, DPO, GRPO trên model 7B); phần "dùng model đã có cho đúng" (giải mã, hiệu chỉnh, chọn câu trả lời) còn mỏng, và hợp với lợi thế 2×T4.

## 2. Bản đồ claim và giả định (full text, đọc các phần abstract, method, kết quả, limitation)
| Bài | Claim chính | Giả định ngầm | Điều chưa kiểm / mâu thuẫn |
|---|---|---|---|
| OV-MER 2410.01495 (ICML 2025) | Định nghĩa task từ vựng mở; metric EW | Gom nhãn về nhóm trên 5 wheel là đủ để chấm | Limitations: chưa xét khác biệt văn hóa (data và annotator đều người Trung Quốc) |
| AffectGPT 2501.16566 (ICML 2025) | MER-Caption, pre-fusion | Học token-level là đủ | README của repo: token-level loss "misaligned with the metrics actually used in OV-MER" |
| AffectGPT-RL 2605.06126 (NeurIPS 2026, theo README repo) | RL với phần thưởng là metric EW | Phải huấn luyện để khớp metric | Limitations: "adding a thinking process leads to performance degradation"; RL "slight degradation in sentiment analysis" |
| R1-Omni 2503.05379; EmotionThinker 2601.15668 (ICLR 2026); OPPO 2606.25325; HyDRA 2603.16463 | Suy luận (CoT, RLVR) giúp nhận diện cảm xúc | Suy luận luôn có ích | **Mâu thuẫn** với AffectGPT-RL (suy luận làm giảm điểm OV-MER) và 2506.19467 (RLVR làm kém mô hình hóa bất đồng của annotator) |
| CIAR (ACM MM 2026); ERCThinker (ACL 2026) | Suy luận thích nghi theo độ khó (có huấn luyện) | Độ khó học được từ nhãn | Đã có → hướng "khi nào nên suy luận" bị chiếm |
| LAMR 2611.11907 (ACM MM 2026) | Phân bổ lại attention theo vai trò lớp, **không huấn luyện** | Mất cân bằng modality nằm ở attention | Tiền lệ trực tiếp cho phương pháp lúc test trong MLLM cảm xúc (can thiệp bên trong model) |
| When Audio and Video Mislead 2611.11987 | Thay audio/video bằng đoạn không liên quan làm giảm accuracy ở 18/18 so sánh | — | Kiểm toán, không có phương pháp |
| CA-MER (ACM MM 2025), EmoReAlM 2602.07054 (ICLR 2026), OPPO | MLLM dựa quá nhiều vào audio hoặc prior của text; bịa cue | — | Vùng đã đông (≥ 8 bài 2026) |
| Style over Substance 2607.18508 | EmoPrefer giải được bằng độ dài và danh tính model sinh (65,8 so với 66,8 WAF) | — | Benchmark MER-Prefer có shortcut |
| 2604.27345; Scaling Ambiguity 2601.14620 (ICASSP 2026) | LLM bắt được nhãn chứ không bắt được độ bất định; mẫu sinh ra ít đa dạng hơn người | — | Hiệu chỉnh hậu kỳ cho text đã có; đa phương thức chưa |
| MultiMER, Agent-MER, ZeroES (ACM MM / workshop) | Bỏ phiếu giữa model hoặc giữa mẫu; chuẩn hóa từ vựng bằng tay | Đồng thuận theo từ là đủ | Chưa đọc full text (ACM DL chặn) |

## 3. Phiên sinh ý tưởng (theo lăng kính; ★ = vào bước 4)
- Lệch mục tiêu và metric: ★F1 MBR theo metric EW; ★F1+ dùng chung một luật quyết định cho task mở và closed-set (chiếu nhãn mở về lớp qua emotion wheel); F1-k chọn số nhãn tối ưu theo metric.
- Giả định gãy: ★N4 prior nhãn lệch khi MLLM chọn từ danh sách lớp (closed-set zero-shot); "không đưa danh sách lớp vào prompt" (sinh mở rồi chiếu về lớp) như một cách tránh lệch prior; suy luận không luôn có ích (đã có CIAR, ERCThinker → loại).
- Nguyên lý lân cận: ★N4 hiệu chỉnh kiểu batch/contextual calibration có cấu trúc cảm xúc; conformal prediction cho tập nhãn cảm xúc (vùng calibration G15 đông → loại); MBR từ dịch máy (= F1).
- Nối hai nhánh: nối OV-MER với dataset closed-set (= F1+); nối nhãn mềm EmoS với giải mã (= A, đã loại vì C4).
- Lợi thế riêng: phương pháp lúc test, không huấn luyện, chạy T4; data MER đã có; EmoS mở rộng ra ngoài họ MER.
- Đã loại trong phiên: MER-Prefer (shortcut); xung đột modality (đông); suy luận thích nghi (CIAR, ERCThinker); A (Scaling Ambiguity, AER-LLM).

## 4. Abstract viết trước cho các ý tưởng ★

**F1. Metric-aligned decoding for open-vocabulary emotion recognition.** Multimodal LLMs for open-vocabulary emotion recognition are trained with token-level likelihood but evaluated with set-level, emotion-wheel-grouped F-scores, and closing this gap so far requires reinforcement learning on 7B models. We show that the gap can be closed at inference time: sampling a handful of label lists and selecting the one with maximal expected emotion-wheel F-score (minimum Bayes risk decoding) consistently improves zero-shot and fine-tuned MLLMs. On OV-MERD and an out-of-family English benchmark built from EmoS, the decoder improves EW-F by X points over greedy decoding and Y over self-consistency voting, with no training. Analyses show where the gains come from (label cardinality, vocabulary coverage, ambiguous clips).
- Bảng chính: backbone × {greedy, vote, MBR} × {OV-MERD, EmoS-MELD}; so với AffectGPT-RL.
- Phản biện mạnh nhất: "MBR là kỹ thuật có sẵn; đây là ứng dụng". Đối đáp: utility theo cấu trúc emotion wheel, phân tích theo độ mơ hồ, và bằng chứng không cần huấn luyện mà gần bằng RL.
- Cược hai chiều: đạt → bài; trượt → không còn gì (không có dạng phân tích đủ cho C3).

**F1+. One decision rule for open and closed emotion tasks.** Emotion MLLMs are evaluated on open-vocabulary and closed-set benchmarks with separate prompting conventions; closed-set prompting exposes the class list and induces label-prior bias. We propose to always generate open labels, aggregate samples under an emotion-wheel utility, and project to the closed label set only at decision time. Across OV-MERD, EmoS-MELD, MER2024, MELD and M3ED, the same training-free rule improves both open-vocabulary EW-F and closed-set WAF over standard prompting.
- Điều kiện: F1 đạt và phép chiếu về lớp không làm giảm WAF so với prompt có danh sách lớp.
- Ưu điểm: C2 mạnh (2 loại task, 5 dataset). Phản biện: "chiếu qua wheel làm mất lớp neutral hoặc surprise" → cần quy tắc rõ cho các lớp không có trên wheel.

**N4. Structure-aware test-time prior correction for zero-shot emotion MLLMs.** When multimodal LLMs classify emotions from a given label list, their predictions are dominated by a few classes, most often neutral. We estimate the model's label prior from unlabeled test predictions and correct it using the structure of emotions (valence/arousal groups, dialogue context) rather than independent per-class offsets. On MER2024, MELD and M3ED, the correction improves zero-shot WAF over standard prompting and over generic contextual and batch calibration.
- Điều kiện premise: MLLM zero-shot thật sự lệch prior trên các dataset này (chưa đo).
- Phản biện mạnh nhất: "batch calibration đã có ở NLP" → bắt buộc so sánh, và phần cấu trúc cảm xúc phải tạo khác biệt đo được.
- Cược hai chiều: nếu lệch prior không đáng kể → bỏ ngay, chi phí chỉ một pilot.

## Khuyến nghị [Đánh giá]
Thứ tự: F1 (đang chạy P13) → nếu đạt, F1+ là dạng bài mạnh nhất; N4 là phương án song song, cần một pilot đo lệch prior trên MLLM (MER2024, MELD; khoảng 1.000 clip mỗi bộ) có thể chạy ngay sau P13 trên cùng pipeline.
