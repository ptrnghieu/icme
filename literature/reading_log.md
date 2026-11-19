# Nhật ký đọc tài liệu

Mức đọc: **Toàn văn** = đọc hết phần nội dung (không tính references) · **Từng phần** = ghi rõ đã đọc mục nào · **Abstract** = chỉ abstract.
Mọi bản PDF tải từ nguồn mở ngày 2026-11-19.

| Ngày | Công trình | Nguồn | Mức đọc | Nội dung chính liên quan | Dùng cho |
|---|---|---|---|---|---|
| 11-19 | Sekiavandi et al., *AFFEC dataset* (arXiv 2504.18969v2) | arxiv.org/pdf/2504.18969v2 | Toàn văn | Felt và perceived thang 1–9, chia 3 lớp bằng optimal binning; split stratified 60/20/20 theo trial, **không theo người tham gia**; macro-F1 đa phương thức 0,43–0,46, thêm personality 0,46–0,48. | B |
| 11-19 | Goncalves, Chou, Salman, Lee, Busso, *Jointly Learning From Unimodal and Multimodal-Rated Labels in AVER*, IEEE OJSP vol. 6, 2025, DOI 10.1109/OJSP.2025.3530274 | lab-msp.com/MSP/publications/Goncalves_2025.pdf | Toàn văn | Huấn luyện hai giai đoạn: lớp unimodal học bằng nhãn voice-only/face-only cộng nhãn AV, lớp chung học bằng nhãn AV. CREMA-D, multi-label, split độc lập người nói; WavLM + MobileNetV2; khoảng 500 giờ A100. Tự nhận *"the only corpus"* có nhãn theo cả 3 điều kiện kích thích. | A (gần nhất) |
| 11-19 | Chou, Wu, Lee, Lee, *Stimulus Modality Matters* (arXiv 2409.10762v4) | arxiv.org/pdf/2409.10762 | Toàn văn (bảng kết quả xem lướt) | SER audio-only huấn luyện bằng nhãn Voice/Face/AV/All; CCC Voice–AV 0,573, Face–AV 0,805; dùng partition của EMO-SUPERB (5 session). | A |
| 11-19 | Lei & Cao, *AVER with Preference Learning Based on Intended and Multi-Modal Perceived Labels*, TAFFC 14(4) 2023 | par.nsf.gov/servlets/purl/10476345 | Từng phần (grep phần dữ liệu và protocol) | Dùng nhãn intended và nhãn perceived theo audio-only, visual-only, AV cho ranking; kiểm định 5-fold độc lập chủ thể. | A |
| 11-19 | Peng et al., *OGM-GE*, CVPR 2022 (arXiv 2203.15332) | arxiv.org/pdf/2203.15332 | Từng phần (§3 modulation, §4.1–4.3) | CREMA-D: 1 frame cho visual; audio-only 52,5 / visual-only 41,9 / OGM-GE 61,9. Split 6.698/744 (đối chiếu với repo: trùng 91/91 diễn viên). | A |
| 11-19 | Wei et al., *Sample-level Modality Valuation*, CVPR 2024 (arXiv 2309.06255) | arxiv.org/pdf/2309.06255 | Từng phần (Abstract, §1, §3) | Shapley cấp mẫu; phải tự tạo MM-Debiased vì không có chuẩn sự khác biệt modality cấp mẫu. | A |
| 11-19 | Xu et al., *BalanceBenchmark* (arXiv 2502.10816v4) | arxiv.org/pdf/2502.10816 | Từng phần (§1, §4.4–6) | 17 phương pháp, 7 dataset; CREMA-D: unimodal audio 59,38 / video 58,10, baseline 65,50, MMPareto 79,97; chỉ số imbalance dựa trên Shapley; không dùng dữ liệu tri giác người. | A |
| 11-19 | Kaffeza, Georgiou, Potamianos, *The Illusion of Balanced MSA*, Interspeech 2026 (arXiv 2609.11247) | arxiv.org/pdf/2609.11247 | Từng phần (Abstract, §1–2) | Gradient và loss balancing không thắng late concat trên MOSI/MOSEI; "loss is not utility, and gradients are not importance". | A (động cơ) |
| 11-19 | Xiong et al., *IIBalance* (arXiv 2603.17347v2) | arxiv.org/pdf/2603.17347 | Từng phần (Abstract, §1–3, §5.A–B) | Intrinsic Information Budget = 1 − entropy trung bình của mô hình unimodal; prior ở cấp dataset; có CREMA-D. | A |
| 11-19 | Liu et al., *CH-SIMS v2.0 & AV-MC*, ICMI 2022 (arXiv 2209.02604) | arxiv.org/pdf/2209.02604 | Từng phần (Abstract, §1, grep "unimodal") | 4.402 mẫu có nhãn unimodal; AV-MC dùng mixup A/V để chống text-predominance. | A |
| 11-19 | arXiv 2510.21797, 2610.10808, 2603.08230, 2606.21657 | arXiv API | Abstract | Sample-level imbalance (GMM), three-score conflicts, LALM ambiguous SER, Chehre (nhận diện biểu cảm dạng phân phối) | A, C |
| 11-19 | *Integration of affective cues in context-rich and dynamic scenes varies across individuals*, Nat. Commun. 2025, DOI 10.1038/s41467-025-67466-1 | PMC12824156 (WebFetch) | Tóm tắt nội dung | Mô hình Bayes tích hợp tín hiệu cho từng cá nhân; không dùng VEATIC, không mô hình phân phối hai đỉnh hay độ trễ. | C |

## Không lấy được toàn văn
- *Reference-guided teacher-student learning for sample-level modality imbalance in multimodal learning*, Information Sciences 2026, DOI 10.1016/j.ins.2026.123797. Không có bản OA, OpenAlex không có abstract. User cần tải qua thư viện.

## Truy vấn đã chạy
Kết quả thô lưu trong scratchpad của phiên:
- arXiv API, đợt 1: 8 truy vấn ("modality imbalance" × human / unimodal label / annotat\*; "imbalanced multimodal" × human; "modality competition" × human; CREMA-D × perceptual / face-only; "unimodal annotations").
- arXiv API, đợt 2: 6 truy vấn (sample-level modality valuation, BalanceBenchmark, imbalance survey, sample-level CREMA-D, modality reliance × human × emotion, human-like × audio-visual × emotion).
- arXiv API, đợt 3: 6 truy vấn (CREMA-D speaker-/actor-independent × imbalance, imbalance × speaker / human perception / perception / human).
- WebSearch (extended) 4 truy vấn.
- Semantic Scholar citations: Wei'24 (96), Goncalves'25 (15), Chou'25 (5), AFFEC (11), VEATIC (25).
