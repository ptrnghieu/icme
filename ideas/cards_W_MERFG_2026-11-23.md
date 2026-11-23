# Thẻ ý tưởng: W (ERC giám sát yếu, đa modality) và MER-FG (cảm xúc từ vựng mở), 2026-11-23

Đọc full text (pdftotext, các phần abstract, method, kết quả, limitation/conclusion):
- W: DERC (AAAI 2025), HAN/AVVP 2007.10558 (ECCV 2020), JoMoLD 2204.11573 (ECCV 2022), VAPLAN 2406.00919, MIL-SER 2008.06667, MUG 2408.16029.
- MER-FG: OV-MER 2410.01495, AffectGPT 2501.16566 (ICML 2025), AffectGPT-R1 2508.01318, HyDRA 2603.16463, APCF 2609.16962, MER2026 2604.19417.
- Code đã kiểm bằng `git ls-remote` (2026-11-23): zeroQiaoba/AffectGPT (gồm AffectGPT, AffectGPT-RL, OV-MER, EmoPrefer), YapengTian/AVVP-ECCV20, MCG-NJU/JoMoLD, Lvcrezia77/PRODEN.

Bằng chứng chung từ bản đồ lỗi (d1, harness LR): trên M3ED, nhóm câu có annotator bất đồng chứa 66,8% lỗi (tỉ lệ lỗi 57,1% so với 35,1%); "cảm xúc → đoán neutral" chiếm 42,6% lỗi. Trên CH-SIMS, nhóm clip có nhãn unimodal mâu thuẫn chứa 73% lỗi.

## W1. Gán nhãn câu theo ràng buộc phủ tập (coverage-constrained assignment)
- **Cơ chế:** trong mỗi hội thoại, mỗi cảm xúc trong tập S phải xuất hiện ở ít nhất một câu. PRODEN và DERC-PL coi từng câu độc lập (tập ứng viên = S), nên bỏ qua ràng buộc này. Đề xuất: mỗi vòng EM, giải một bài toán gán (optimal transport có entropy, biên dưới ≥ 1 câu cho mỗi cảm xúc trong S, prior tần suất lớp) giữa các câu và S; dùng nghiệm làm pseudo-label.
- **Vì sao nên hiệu quả:** chặn nghiệm suy biến "mọi câu đều neutral", vốn là kiểu lỗi lớn nhất đo được trên M3ED.
- **Baseline có code:** PRODEN; DERC-PL (không có code, phải cài lại theo bài); cận trên là học với nhãn đầy đủ.
- **Data:** MELD, M3ED, EmoryNLP (text); nhãn tập dẫn xuất tự động như DERC.
- **Phản biện dự kiến:** "OT cho partial-label learning đã có ở miền khác" → phải trích dẫn, so sánh; novelty trong affective chưa kiểm.

## W2. Khử nhiễu ứng viên theo từng modality (chuyển từ JoMoLD/AVVP)
- **Cơ chế:** head riêng cho mỗi modality; một cảm xúc trong S phải xuất hiện ở ít nhất một câu và ít nhất một modality (giống "a labeled event would appear in at least one modality" của JoMoLD). Dùng quan hệ giữa loss trong modality và giữa các modality để loại ứng viên sai ở từng câu, từng modality.
- **Vì sao nên hiệu quả:** cảm xúc có thể chỉ lộ ở lời nói, hoặc chỉ lộ ở giọng; bằng chứng theo modality giúp định vị.
- **Baseline có code:** JoMoLD (AVVP), PRODEN, DERC-PL (cài lại).
- **Điều kiện:** phụ thuộc kết quả P12 (đa modality có thu hẹp khoảng cách giám sát yếu không).

## W3. Prior từ foundation model, giới hạn trong tập S
- **Cơ chế:** xác suất zero-shot từ model cảm xúc pretrained (speech, face, text), che bởi S, dùng để khởi tạo và điều chuẩn trọng số PLL (giống VAPLAN dùng CLIP/CLAP làm pseudo-label đoạn).
- **Vì sao nên hiệu quả:** phá thế hòa giữa các cảm xúc trong S với chi phí thấp, chạy được trên T4.
- **Rủi ro:** model pretrained có thể đã huấn luyện trên chính MELD hoặc dataset đích → phải kiểm rò rỉ trước khi dùng.

## F1. Giải mã tập nhãn khớp với metric, không cần huấn luyện (ưu tiên 1)
- **Gap [Sự thật]:** README của AffectGPT ghi: "Existing approaches are trained with a token-level loss, which is misaligned with the metrics actually used in OV-MER … EW-based metrics are non-differentiable"; AffectGPT-RL giải bằng RL, tức phải huấn luyện lại model 7B.
- **Cơ chế:** chấm log-likelihood của MLLM cho một từ điển cảm xúc cố định (từ vựng nhãn của OV-MERD/Human-OV), gom theo emotion wheel, rồi chọn tập nhãn tối đa hóa *kỳ vọng* EW-F dưới posterior của model (tối đa hóa F-measure theo lý thuyết quyết định, đã có ở multi-label learning, miền khác). Áp dụng cho mọi backbone (AffectGPT, Qwen2.5-Omni zero-shot).
- **Vì sao nên hiệu quả:** sinh văn bản tham lam không chọn số nhãn tối ưu cho metric dạng tập; bộ giải mã kỳ vọng-F chọn đúng số lượng và nội dung theo metric.
- **Data:** OV-MERD (332) và MER2025-FG test (1.200), cả hai nằm trong Human-OV của MER2026 mà user có quyền. **Rủi ro C2:** cùng họ MER; cần thêm một tập ngoài họ MER có nhãn chi tiết.
- **Baseline có code:** AffectGPT, AffectGPT-RL (cùng repo); MLLM zero-shot.
- **Compute:** chấm điểm log-prob với model 7B lượng tử 4-bit trên T4, khoảng 1.500 mẫu × kích thước từ điển; ước lượng cụ thể khi pilot.
- **Phản biện dự kiến:** "chỉ là mẹo giải mã" → cần ablation theo backbone, theo kích thước từ điển, và so với AffectGPT-RL.

## F2. Mục tiêu mềm theo mức mơ hồ của nhãn
- **Cơ chế:** LoRA nhỏ (≤ 3B) với mục tiêu mềm thay cho chuỗi nhãn cứng. Bằng chứng: lỗi tập trung ở nhãn mơ hồ (d1); MER2026 ghi Human-OV và MER-Caption+ cho kết quả gần như nhau ("trade-off between data quality and quantity").
- **Điểm yếu:** chưa biết Human-OV có nhãn từng annotator không; nếu chỉ có nhãn đa số thì không có tín hiệu mơ hồ để học.

## F3. Đối chiếu bằng chứng giữa các modality khi giải mã từ vựng mở
- **Cơ chế:** so log-likelihood của từng cảm xúc ứng viên khi có đủ modality với khi chỉ có text; giảm điểm các cảm xúc chỉ được prior của text ủng hộ (HyDRA nêu "premature commitment to dominant data priors"; EmoReAlM nêu hallucination do prior của text). Ghép được vào khung chấm điểm của F1.
- **Rủi ro C4:** gần vùng xung đột modality trong MLLM đã bão hòa (CA-MER và ≥ 8 bài tiếp nối).
