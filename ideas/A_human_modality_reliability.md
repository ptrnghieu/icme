# Ứng viên A: Nhãn unimodal của người làm *thước đo* cho modality imbalance

**Tên làm việc:** *Balancing the Wrong Modality? Human Unimodal Perception as a Yardstick for Modality Imbalance in Audio-Visual Emotion Recognition*
**Trạng thái (2026-11-21):** **H KHÔNG ĐƯỢC ỦNG HỘ** theo thí nghiệm can thiệp đã đăng ký trước (`pilot/results/summary_2026-11-21.md`). Q1 và các phát hiện protocol vẫn đứng được. Cách đóng khung ban đầu (cần H) **không dùng được nữa**; chờ user chọn hướng đóng khung lại (xem tổng kết).
Ký hiệu: **[SỰ THẬT]** = đã tự kiểm chứng, có nguồn · **[ĐÁNH GIÁ]** = nhận định chuyên môn.

---

## 0. Đóng khung contribution (chốt với user ngày 2026-11-19)

**Không claim:** "Imbalance trong MER được giải quyết bằng nhãn unimodal của người." Hai lý do:
- Đã có người làm: CH-SIMS v2 AV-MC 2022; Goncalves 2025 dùng nhãn unimodal làm target.
- Không mở rộng được: chỉ CREMA-D và CH-SIMS có loại nhãn này.

**Claim:** Nhãn unimodal của người là **thước đo** để chẩn đoán modality imbalance. Bài trả lời hai câu hỏi:
- **Q1.** Các phương pháp cân bằng hiện nay có cân bằng *đúng hướng* không? Tức là có tăng mức dựa vào visual **đúng ở những mẫu mà người thấy khuôn mặt mang thông tin**, hay chỉ tăng đồng loạt?
- **Q2 (giả thuyết H).** Cân bằng đúng hướng có quan trọng không? Mô hình căn chỉnh tốt với người có **tổng quát hóa tốt hơn sang diễn viên chưa thấy** không?

**Đầu ra dùng được cho cả cộng đồng:** dùng nhãn người *một lần* trên CREMA-D để xếp hạng các tín hiệu nội bộ (tỷ lệ score của OGM-GE, Shapley, entropy…) theo mức chúng phản ánh đúng mức người dựa vào modality. Tín hiệu tốt nhất sau đó được áp lên dataset **không có nhãn người** (MELD, CMU-MOSEI). Nhãn người đóng vai thước chuẩn dùng một lần, không phải điều kiện bắt buộc của phương pháp.

**Tiền lệ cho kiểu đóng góp này:** Das, Agrawal, Zitnick, Parikh, Batra. *Human Attention in VQA: Do Humans and Deep Networks Look at the Same Regions?* arXiv 1606.03556 (nộp EMNLP 2016). Bài dùng bản đồ chú ý của người làm chuẩn đánh giá attention của mô hình.

| Đóng góp | Công trình gần nhất | Phần mới | Độ mạnh novelty |
|---|---|---|---|
| **N1 (lõi).** Thước đo cấp mẫu từ người (r_voice, r_face) và chỉ số căn chỉnh với đóng góp unimodal của mô hình; trả lời Q1 | Wei'24 định giá bằng Shapley của mô hình, phải tạo dữ liệu tổng hợp vì **không có chuẩn thật**; BalanceBenchmark và IIBalance cũng đo từ mô hình | Chuẩn thật do người đo, cấp mẫu, ngay trên benchmark chủ lực | [ĐÁNH GIÁ] Mạnh nhất |
| **N2.** Kiểm định H qua protocol độc lập diễn viên. Kèm theo là kiểm toán 2 lỗi protocol: (a) split trùng 91/91 diễn viên; (b) OGM-GE chọn checkpoint theo test accuracy | PLOS ONE 2026 (SER): split random dễ hơn split theo actor | Mới với mảng imbalance; vừa là phép thử cho H vừa là phát hiện phụ | [ĐÁNH GIÁ] Vừa; có giá trị nếu kết luận thay đổi |
| **N3.** Xếp hạng tín hiệu nội bộ theo độ khớp với người, rồi chuyển tín hiệu tốt nhất sang dataset không có nhãn người | Chưa thấy công trình nào làm | Biến thước đo thành công cụ dùng được rộng rãi | [ĐÁNH GIÁ] Vừa; phụ thuộc kết quả của N1 |

**Phải trích dẫn và phân biệt rõ:** Lei & Cao 2023, Goncalves 2025 và Chou 2025 dùng nhãn voice/face của CREMA-D làm **target huấn luyện**. CH-SIMS v2 AV-MC dùng nhãn unimodal làm **target cho mixup**.

**Rủi ro khái niệm** [ĐÁNH GIÁ]: tri giác người khác với lượng thông tin máy khai thác được, vì mô hình có thể dùng hợp lệ những đặc trưng âm học mà người không nghe ra. Không viết bài theo kiểu "người là ground truth của thông tin". Mức căn chỉnh theo người chỉ có giá trị khi gắn được với hệ quả đo được (H). Nếu pilot không thấy liên hệ đó thì N1 chỉ còn là một phép đo mô tả, không đủ cho ICME.

## 1. Phát biểu gap
Cộng đồng *modality imbalance* coi CREMA-D là benchmark audio-visual chủ lực, và coi visual là modality "yếu" cần được kéo lên. Nhưng mọi phương pháp đều ước lượng đóng góp của từng modality **từ chính mô hình**, không có chuẩn đối chiếu bên ngoài. Trong khi đó, CREMA-D có sẵn khoảng 220 nghìn phiếu chấm của người theo **giọng / mặt / AV**, ở cấp từng mẫu. Ngoài ra, protocol chuẩn của cộng đồng này có hai lỗi: để cùng diễn viên ở cả train và test, và chọn checkpoint theo test set.

## 2. C1: Bằng chứng gap
| # | Bằng chứng | Loại | Nguồn |
|---|---|---|---|
| 1 | Split CREMA-D 6.698/744 dùng trong OGM-GE (CVPR'22), PMR (CVPR'23) và InfoReg (CVPR'25) có **test set giống hệt nhau**, và **cả 91/91 diễn viên** xuất hiện ở cả train lẫn test. Diagnosing & Re-learning (ECCV'24) dùng split khác nhưng cũng trùng toàn bộ diễn viên. | [SỰ THẬT] | Đã clone các repo GeWu-Lab/OGM-GE_CVPR2022, fanyunfeng-bit/Modal-Imbalance-PMR, GeWu-Lab/InfoReg_CVPR2025, GeWu-Lab/Diagnosing_Relearning_ECCV2024 và đếm ID diễn viên (4 ký tự đầu tên file). |
| 2 | OGM-GE chỉ dùng **1 frame** cho visual của CREMA-D. Kết quả: audio-only 52,5, visual-only 41,9. | [SỰ THẬT] | arXiv 2203.15332, §4.2 và Bảng 1; code mặc định `--fps 1` |
| 3 | Theo người chấm: accuracy so với nhãn diễn là **mặt 56,9% > giọng 39,2%**, AV 62,5%. Nhãn đa số của giọng và mặt khác nhau ở 50,2% số clip. 34,9% clip có acc(mặt) > acc(giọng) + 0,3; 8,7% clip theo chiều ngược lại. | [SỰ THẬT] | Tự tính từ `finishedResponses.csv` (219.688 phiếu, 2.443 người chấm); xem `datasets/access_log.md` |
| 4 | Các thước đo đóng góp modality hiện có đều xuất phát từ mô hình: Shapley cấp mẫu (Wei et al., CVPR'24); chỉ số imbalance dựa trên Shapley (BalanceBenchmark); "Intrinsic Information Budget" ước lượng từ entropy dự đoán của mô hình unimodal (IIBalance, arXiv 2603.17347). | [SỰ THẬT] | Đã đọc phần phương pháp của cả ba |
| 5 | Interspeech 2026 ("The Illusion of Balanced MSA") kết luận các phương pháp dựa trên tối ưu hóa *"conflate fitting speed with discriminative contribution"* và kêu gọi định giá modality bằng dữ liệu held-out. | [SỰ THẬT] | arXiv 2609.11247, Abstract và §1 |
| 6 | Nhãn người theo modality của CREMA-D **đã được dùng làm nhãn huấn luyện** (Lei & Cao, TAFFC 2023; Goncalves et al., IEEE OJSP 2025; Chou et al., arXiv 2409.10762), **nhưng chưa ai dùng làm chuẩn đối chiếu cho bài toán imbalance**. | [SỰ THẬT] cho phần "đã dùng"; [SỰ THẬT trong phạm vi đã quét] cho phần "chưa ai" | Xem §5 |
| 7 | Code OGM-GE gọi `valid()` trên **test set** sau mỗi epoch, rồi lưu checkpoint có `acc > best_acc`. Nghĩa là con số báo cáo được chọn theo test accuracy; repo không có validation set. | [SỰ THẬT] | `main.py` của GeWu-Lab/OGM-GE_CVPR2022 (hàm `main`, vòng lặp epoch) |

## 3. C2: Độ rộng
- **CREMA-D:** audio + visual, 6 cảm xúc, tiếng Anh, có phiếu chấm theo giọng / mặt / AV.
- **CH-SIMS v2(s):** text + audio + visual, sentiment, tiếng Trung. 4.403 clip, có nhãn người riêng cho từng modality `label_T/A/V`, split chính thức 2.722/647/1.034. Tương quan: label_T–label 0,776; label_A–label 0,876; label_V–label 0,841. Đây là bối cảnh imbalance kiểu "text chiếm ưu thế".
- (Tùy chọn) **CH-SIMS v1:** 2.280 dòng nhãn.
- **Dataset không có nhãn người** để thử chuyển tín hiệu (N3): MELD và CMU-MOSEI. Chưa kiểm tra truy cập; phải kiểm tra trước khi đưa vào E4.

[ĐÁNH GIÁ] Như vậy có 2 task, 2 ngôn ngữ và 2 tổ hợp modality.

## 4. C3: Kế hoạch thực nghiệm (khớp topic ICME "Multi-modal media computing…" và "AI/ML-based multimedia")
- **Pilot (trước 28/11/2026, code ở `pilot/`):** joint baseline, OGM-GE (α ∈ {0,1; 0,3; 0,5; 0,8}), audio-only, visual-only. Chạy trên (S) split chuẩn 6.698/744 và (A) split độc lập diễn viên (2/5 fold). Đầu ra:
  - G1: accuracy S và A; checkpoint chọn theo val so với chọn theo test.
  - G2: căn chỉnh theo người (Q1).
  - G3: căn chỉnh trên val (diễn viên chưa thấy) có dự báo được test accuracy trên diễn viên mới không (Q2/H).
  Quy tắc go/no-go nằm trong `pilot/README.md`.
- **Thí nghiệm can thiệp cho H (2026-11-20):** pilot cho thấy mức khớp với người gần như chỉ do α quyết định (ρ ≈ 0,96), nên G3a không tách được H khỏi cường độ cân bằng. Thí nghiệm can thiệp đã đăng ký trước ở `pilot/INTERVENTION.md`: trọng số theo clip lấy từ người (`human`), so với cùng tập trọng số bị xáo trộn (`shuffled`), cộng thêm `anti` và `uniform`; 32 run.
- **E1 (nếu GO):** mở rộng lên 6–8 phương pháp (AGM, PMR, MMPareto, Diagnosing & Re-learning, InfoReg, Wei'24) qua BalanceBenchmark; 5 fold độc lập diễn viên; 3 seed.
- **E2:** xếp hạng tín hiệu nội bộ theo độ khớp với người (N3).
- **E3:** lặp lại Q1 trên CH-SIMS v2 (`label_T/A/V`) bằng feature có sẵn.
- **E4:** áp tín hiệu tốt nhất lên MELD hoặc CMU-MOSEI (sau khi kiểm tra truy cập data).
- **Compute** [ĐÁNH GIÁ; pilot sẽ đo thật]: pilot có 49 run, mỗi run khoảng 0,5–1,5 giờ T4, tức 25–75 giờ GPU. Với 2×T4 chạy song song là 12–37 giờ thực.
- **Phản biện dự kiến và cách đối phó** [ĐÁNH GIÁ]:
  1. *"Nhãn người hiếm."* Nhãn người chỉ là thước đo dùng một lần; N3 chuyển tín hiệu sang dataset không có nhãn người.
  2. *"Chỉ là bài phân tích."* Có tiền lệ ở mảng VQA (Das et al. 2016). Ngoài ra N3 tạo ra một công cụ cụ thể.
  3. *"Trùng Goncalves'25 / CH-SIMS v2."* Hai bài đó dùng nhãn unimodal làm target huấn luyện; bài này dùng làm thước đo.
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
- **CREMA-D:** L2 cho file media (wav, flv tải thật), L3 cho phiếu chấm. License ODbL / DbCL. Tổng dung lượng theo con trỏ LFS: AudioWAV 0,61 GB, VideoFlash 2,44 GB.
- **CH-SIMS v2(s):**
  - Kênh chính thức (Google Drive của thuiar/ch-sims-v2): ngày 19/11, `meta.csv` trả về "Quota exceeded". `Raw.zip` (11,65 GB) ban đầu trả HTTP 206, sau đó cũng bị quota.
  - Mirror Hugging Face `tamb2203579/CH-SIMSv2` (không gated): đã tải `meta.csv` (L3); `Processed/unaligned.pkl` (3,6 GB) trả HTTP 206 (L2).
  - License: README chính thức không ghi license dataset. Thẻ của mirror ghi apache-2.0, nhưng đó là do người upload mirror tự đặt.
