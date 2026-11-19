# Ứng viên A: Kiểm toán modality imbalance bằng tri giác người theo từng modality

**Tên làm việc:** *Balancing the Wrong Modality? Human-Grounded Evaluation of Modality Imbalance in Audio-Visual Emotion Recognition*
**Trạng thái (2026-11-19):** **ỨNG VIÊN CHÍNH.** C1, C2, C5 đã xác minh. C4: chưa tìm thấy công trình trùng, nhưng còn 1 bài paywall chưa đọc được (xem §5). C3: có kế hoạch kèm điểm go/no-go bằng pilot.
Ký hiệu: **[SỰ THẬT]** = đã tự kiểm chứng, có nguồn · **[ĐÁNH GIÁ]** = nhận định chuyên môn.

---

## 0. Novelty chính xác (bổ sung 2026-11-19, theo câu hỏi của user)

**Một câu:** Các phương pháp chống modality imbalance hiện nay đều tự đo "modality nào đóng góp bao nhiêu" từ chính mô hình. Bài này lần đầu dùng **tri giác người theo từng modality, đo trên từng mẫu**, làm **chuẩn đối chiếu bên ngoài** cho đại lượng đó, rồi kiểm định giả thuyết H:

> **H:** Cân bằng modality theo đúng mức người dựa vào từng modality trên từng mẫu thì **tổng quát hóa sang diễn viên chưa thấy** tốt hơn cân bằng dựa trên tín hiệu nội bộ của mô hình.

| Đóng góp | Công trình gần nhất đã làm gì | Phần mới của bài này | Độ mạnh novelty |
|---|---|---|---|
| **N1 (lõi).** Chuẩn đối chiếu đóng góp modality cấp mẫu từ người: độ tin cậy theo modality r_i^voice, r_i^face, và chỉ số căn chỉnh giữa đóng góp unimodal của mô hình với r_i | Wei'24: định giá bằng Shapley của mô hình; phải tạo dataset tổng hợp MM-Debiased vì **không có chuẩn thật**. BalanceBenchmark và IIBalance cũng đo từ mô hình. | Chuẩn thật do người đo, ở cấp từng mẫu, ngay trên benchmark chính của cộng đồng | [ĐÁNH GIÁ] Mạnh nhất. Không tìm thấy công trình trùng trong phạm vi đã quét (còn 1 bài paywall). |
| **N2 (hỗ trợ).** Kiểm toán protocol CREMA-D: split trùng 91/91 diễn viên; đánh giá lại theo split độc lập diễn viên | Bài SER trên PLOS ONE 2026 cho thấy split random dễ hơn split theo actor, nhưng không thuộc mảng imbalance | Kiểm tra xem các kết luận và thứ hạng của phương pháp imbalance có còn đứng khi bỏ rò rỉ diễn viên | [ĐÁNH GIÁ] Yếu nếu đứng riêng; chỉ có giá trị nếu kết luận thay đổi |
| **N3 (phụ).** Điều biến theo prior độ tin cậy của người, kèm head dự đoán để inference không cần nhãn người | Wei'24: tái lấy mẫu cấp mẫu theo Shapley. IIBalance: prior cấp dataset từ entropy. | Prior lấy từ người, ở cấp từng mẫu | [ĐÁNH GIÁ] Thấp–vừa; vai trò là chứng minh chuẩn N1 dùng được trong thực tế |

**Những gì KHÔNG phải là novelty (phải trích dẫn và phân biệt trong bài):**
- Dùng nhãn voice-only/face-only của CREMA-D làm nhãn huấn luyện: đã có ở Lei & Cao 2023, Goncalves et al. 2025, Chou et al. 2025.
- Dùng nhãn người cho từng modality để chống một modality áp đảo: CH-SIMS v2 AV-MC (2022) đã làm, nhưng dùng làm target cho mixup chứ không làm chuẩn đánh giá.

**Rủi ro khái niệm lớn nhất** [ĐÁNH GIÁ]: tri giác người khác với lượng thông tin máy khai thác được. Mô hình có thể dùng hợp lệ những đặc trưng âm học mà người không nghe ra. Vì vậy bài **không được** viết theo kiểu "người là ground truth của thông tin". Claim phải là H, tức là gắn việc căn chỉnh theo người với một hệ quả đo được: tổng quát hóa sang diễn viên mới, hoặc độ bền khi một modality bị nhiễu. Nếu pilot không thấy liên hệ này thì N1 chỉ còn là một phép đo mô tả, không đủ cho ICME.

## 1. Phát biểu gap
Cộng đồng *modality imbalance* coi CREMA-D là benchmark audio-visual chủ lực, và coi visual là modality "yếu" cần được kéo lên. Nhưng mọi phương pháp đều ước lượng đóng góp của từng modality **từ chính mô hình**, không có chuẩn đối chiếu bên ngoài. Trong khi đó, CREMA-D có sẵn khoảng 220 nghìn phiếu chấm của người theo **giọng / mặt / AV**: đây là một chuẩn tham chiếu do người đo, ở cấp từng mẫu, về việc modality nào thực sự mang thông tin cảm xúc. Thêm vào đó, split chuẩn của cộng đồng này để **cùng diễn viên ở cả train và test**.

## 2. C1: Bằng chứng gap
| # | Bằng chứng | Loại | Nguồn |
|---|---|---|---|
| 1 | Split CREMA-D 6.698/744 dùng trong OGM-GE (CVPR'22), PMR (CVPR'23) và InfoReg (CVPR'25) có **test set giống hệt nhau**, và **cả 91/91 diễn viên** xuất hiện ở cả train lẫn test. Diagnosing & Re-learning (ECCV'24) dùng split khác nhưng cũng trùng toàn bộ diễn viên. | [SỰ THẬT] | Đã clone các repo GeWu-Lab/OGM-GE_CVPR2022, fanyunfeng-bit/Modal-Imbalance-PMR, GeWu-Lab/InfoReg_CVPR2025, GeWu-Lab/Diagnosing_Relearning_ECCV2024 và đếm ID diễn viên (4 ký tự đầu tên file). |
| 2 | OGM-GE chỉ dùng **1 frame** cho visual của CREMA-D. Kết quả: audio-only 52,5, visual-only 41,9. | [SỰ THẬT] | arXiv 2203.15332, §4.2 và Bảng 1 |
| 3 | Theo người chấm: accuracy so với nhãn diễn là **mặt 56,9% > giọng 39,2%**, AV 62,5%. Nhãn đa số của giọng và mặt khác nhau ở 50,2% số clip. 34,9% clip có acc(mặt) > acc(giọng) + 0,3; 8,7% clip theo chiều ngược lại. | [SỰ THẬT] | Tự tính từ `finishedResponses.csv` (219.688 phiếu, 2.443 người chấm); xem `datasets/access_log.md` |
| 4 | Các thước đo đóng góp modality hiện có đều xuất phát từ mô hình: Shapley cấp mẫu (Wei et al., CVPR'24); chỉ số imbalance dựa trên Shapley (BalanceBenchmark); "Intrinsic Information Budget" ước lượng từ entropy dự đoán của mô hình unimodal (IIBalance, arXiv 2603.17347). | [SỰ THẬT] | Đã đọc phần phương pháp của cả ba |
| 5 | Interspeech 2026 ("The Illusion of Balanced MSA") kết luận các phương pháp dựa trên tối ưu hóa *"conflate fitting speed with discriminative contribution"* và kêu gọi định giá modality bằng dữ liệu held-out. | [SỰ THẬT] | arXiv 2609.11247, Abstract và §1 |
| 6 | Nhãn người theo modality của CREMA-D **đã được dùng làm nhãn huấn luyện** (Lei & Cao, TAFFC 2023; Goncalves et al., IEEE OJSP 2025; Chou et al., arXiv 2409.10762), **nhưng chưa ai dùng làm chuẩn đối chiếu cho bài toán imbalance**. | [SỰ THẬT] cho phần "đã dùng"; [SỰ THẬT trong phạm vi đã quét] cho phần "chưa ai" | Xem §5 |

## 3. C2: Độ rộng
- **CREMA-D:** audio + visual, 6 cảm xúc, tiếng Anh, có phiếu chấm theo giọng / mặt / AV.
- **CH-SIMS v2(s):** text + audio + visual, sentiment, tiếng Trung. 4.403 clip, có nhãn người riêng cho từng modality `label_T/A/V`, split chính thức 2.722/647/1.034. Tương quan: label_T–label 0,776; label_A–label 0,876; label_V–label 0,841. Đây là bối cảnh imbalance kiểu "text chiếm ưu thế".
- (Tùy chọn) **CH-SIMS v1:** 2.280 dòng nhãn.

[ĐÁNH GIÁ] Như vậy có 2 task, 2 ngôn ngữ và 2 tổ hợp modality, đủ để trả lời câu hỏi "chỉ đúng trên một dataset".

## 4. C3: Kế hoạch thực nghiệm (khớp topic ICME "Multi-modal media computing…" và "AI/ML-based multimedia")
- **E1. Kiểm toán protocol.** Chạy 6–8 phương pháp (concat baseline, OGM-GE, AGM, PMR, MMPareto, Diagnosing & Re-learning, InfoReg, ReconBoost hoặc MLA) trên CREMA-D theo hai protocol: (i) split chuẩn 6.698/744, (ii) split độc lập diễn viên (5-fold theo actor). Mỗi cấu hình 3 seed. Báo cáo mức giảm và sự đảo thứ hạng.
  [SỰ THẬT] Toolkit BalanceBenchmark (GeWu-Lab, commit 4c1482f) đã cài 17 phương pháp và có loader CREMA-D, nên giảm mạnh công sức triển khai.
- **E2. Chỉ số căn chỉnh theo người.** Ở cấp từng mẫu, so bằng chứng unimodal của mô hình (xác suất lớp đúng ở từng nhánh, hoặc Shapley) với độ tin cậy theo modality của người (acc giọng và acc mặt). Câu hỏi chính: các phương pháp cân bằng có đẩy mô hình về phía cách người dựa vào modality, hay chỉ tăng visual một cách đồng loạt?
- **E3. Phương pháp.** Điều biến gradient hoặc trọng số cấp mẫu theo prior độ tin cậy của người, chỉ dùng nhãn người ở pha train. Train thêm một head dự đoán độ tin cậy để inference không cần nhãn người. So sánh với OGM-GE, Wei'24 và IIBalance.
- **E4. Chuyển sang CH-SIMS v2.** Dùng feature có sẵn (`Processed/unaligned.pkl`) và nhãn `label_T/A/V`.
- **Compute** [ĐÁNH GIÁ, cần pilot đo thật]: CREMA-D với ResNet-18, khoảng 6,7 nghìn mẫu, ước 0,5–1,5 giờ T4 mỗi run. E1 khoảng 48 run, tức 24–72 giờ GPU, vừa trong 1 tuần với 2×T4. CH-SIMS dùng feature trích sẵn nên rất nhẹ.
- **Go/no-go trước 28/11/2026:** pilot baseline và OGM-GE trên hai protocol, cộng chỉ số E2. Nếu đổi protocol không làm thay đổi kết luận, và mô hình đã căn chỉnh sẵn với người, thì câu chuyện chính yếu đi. Khi đó chỉ giữ E3 hoặc loại ý tưởng.
- **Phản biện dự kiến và cách đối phó** [ĐÁNH GIÁ]:
  1. *"Nhãn người theo modality hiếm, phương pháp khó tổng quát."* Đóng khung đóng góp chính là **chuẩn đánh giá**. Phương pháp chỉ cần nhãn người ở pha train, kèm head dự đoán để inference không cần nhãn.
  2. *"Chỉ là bài phân tích."* E3 phải thắng các baseline imbalance mạnh trên cả hai dataset.
  3. *"Trùng Goncalves'25 / CH-SIMS v2."* Hai bài đó dùng nhãn unimodal làm **mục tiêu huấn luyện**; bài này dùng chúng làm **chuẩn đối chiếu cho imbalance** và kiểm toán protocol.
  4. *"Accuracy theo nhãn diễn khác accuracy theo nhãn perceived."* Báo cáo cả hai loại nhãn.

## 5. C4: Nhật ký novelty (2026-11-19)
**Truy vấn đã chạy** (chi tiết trong `literature/reading_log.md`): arXiv API 20 truy vấn kết hợp {modality imbalance, imbalanced multimodal, modality competition, balanced multimodal} × {human, annotat\*, unimodal label, perception, speaker, CREMA-D, face-only}. WebSearch extended 4 truy vấn. Citation (Semantic Scholar): Wei'24 (96 bài trích dẫn, sàng theo tiêu đề), Goncalves'25 (15 bài), Chou'25 (5 bài).

| Công trình gần nhất | Mức đọc | Verdict |
|---|---|---|
| Goncalves et al., *Jointly Learning From Unimodal and Multimodal-Rated Labels in AVER*, IEEE OJSP 2025 | Toàn văn | **Trùng một phần.** Dùng nhãn voice-only, face-only, AV của CREMA-D làm nhãn huấn luyện hai giai đoạn, split độc lập người nói. Không bàn imbalance, không kiểm toán protocol. |
| Chou et al., *Stimulus Modality Matters*, arXiv 2409.10762 | Toàn văn (trừ bảng kết quả chi tiết) | **Không trùng.** Chỉ SER audio, so sánh nhãn theo modality. |
| Lei & Cao, *AVER with preference learning…intended and multi-modal perceived labels*, TAFFC 2023 (bản NSF PAR) | Đọc phần dữ liệu và phương pháp qua grep | **Không trùng.** Dùng nhãn perceived theo modality cho ranking, không bàn imbalance. |
| Liu et al., *CH-SIMS v2.0 & AV-MC*, ICMI 2022 | Abstract và §1 | **Trùng một phần ở ý tưởng chung.** Dùng nhãn unimodal để "làm A/V có trọng lượng" bằng mixup, không phải đánh giá căn chỉnh hay gradient modulation. |
| Wei et al., *Sample-level Modality Valuation*, CVPR 2024 | Abstract, §1, phương pháp | **Không trùng.** Định giá bằng Shapley của mô hình; phải tự tạo MM-Debiased vì *không có chuẩn cấp mẫu*. Đây chính là chỗ dữ liệu người lấp vào. |
| BalanceBenchmark (arXiv 2502.10816) | §1, §4–6 | **Không trùng.** Không dùng dữ liệu tri giác người. |
| IIBalance (arXiv 2603.17347) | Abstract, §1, §3 (cách ước lượng IIB), §5 | **Trùng một phần ở động cơ** ("không ép cân bằng"). Budget được ước lượng từ entropy của mô hình, không từ người. |
| Kaffeza et al., *The Illusion of Balanced MSA*, Interspeech 2026 | Abstract, §1 | **Không trùng.** Ủng hộ động cơ của bài. |
| Probabilistic Separation (arXiv 2510.21797), Three-Channel Score Conflicts (arXiv 2610.10808), Disentangling Reasoning in LALMs (arXiv 2603.08230) | Abstract | **Không trùng.** |
| *Reference-guided teacher-student learning for sample-level modality imbalance*, Information Sciences 2026, DOI 10.1016/j.ins.2026.123797 | **Không lấy được toàn văn**: paywall, OpenAlex không có abstract, không có bản OA | Chưa thể kết luận. User cần tải qua thư viện trường để mình đọc. |

**Kiểm tra lại toàn bộ** vào khoảng 20/12/2026 (CLAUDE.md §4.2).

## 6. C5: Dữ liệu
- **CREMA-D:** L2 cho file media (wav, flv tải thật), L3 cho phiếu chấm. License ODbL / DbCL.
- **CH-SIMS v2(s):**
  - Kênh chính thức (Google Drive của thuiar/ch-sims-v2): ngày 19/11, `meta.csv` trả về "Quota exceeded". `Raw.zip` (11,65 GB) ban đầu trả HTTP 206, sau đó cũng bị quota.
  - Mirror Hugging Face `tamb2203579/CH-SIMSv2` (không gated): đã tải `meta.csv` (L3); `Processed/unaligned.pkl` (3,6 GB) trả HTTP 206 (L2).
  - License: README chính thức không ghi license dataset. Thẻ của mirror ghi apache-2.0, nhưng đó là do người upload mirror tự đặt.
