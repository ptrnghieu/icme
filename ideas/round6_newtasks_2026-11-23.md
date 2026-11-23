# Đợt 6 (2026-11-23): kiểm nhanh 3 hướng từ các task mới ở venue chính

Nguồn: bộ quét `tools/new_task_scan.py` (1.442 bài, 19 venue, 2025–2026). Trích dẫn (cited-by) lấy từ OpenAlex `filter=cites:<id>` ngày 2026-11-23.

| Hướng | Bài gốc | Bài trích dẫn (OpenAlex) | Bài tiếp nối trực tiếp | Data / code | Kết luận |
|---|---|---|---|---|---|
| W. ERC giám sát yếu: chỉ có nhãn cấp hội thoại | DERC, AAAI 2025 (doi 10.1609/aaai.v39i23.34629; full text đã đọc) | 8 | **0** (đã đọc tiêu đề 8 bài trích dẫn; không bài nào làm DERC) | DERC chỉ dùng text (BERT); data MELD, IEMOCAP, EmoryNLP; nhãn DERC dẫn xuất tự động "by directly adding the utterance-level emotions into the corresponding conversation-level emotion sets"; bài không có link code | **SỐNG** |
| D. MER trên dữ liệu đã xóa danh tính | DEEMO, ACM MM 2025 (arXiv 2504.19549; full text đã đọc) | 32 | 0 bài làm cùng task (đa số trích dẫn là micro-gesture, survey); phía ẩn danh giọng nói giữ cảm xúc đang sôi động (EASY Interspeech 2025, DECRA Interspeech 2026, 2 bài khác ICASSP 2025, SLT 2024) | De-identification = làm mờ Gaussian vùng mặt + ẩn danh giọng; nguồn là phỏng vấn vận động viên sau trận; **không tìm thấy link data** trong bài và trên HF | **SỐNG, yếu hơn W** (phải tự dựng bản xóa danh tính của MER2024/MELD; chưa có ý tưởng phương pháp rõ) |
| C. Xung đột modality trong MLLM | CA-MER, ACM MM 2025 | 30 | **≥ 8 bài 2026 cùng chủ đề**: CERA, CAIR (Findings ACL), VISTA, CALM-MER, Layer Roles Matter (ACM MM 2026), Beyond Multimodal Emotion Conflict, When Audio and Video Mislead, Rethinking Attention Reallocation | — | **ĐÃ LOẠI (bão hòa; cùng vùng với G8)** |

## W: phát biểu nhanh (tầng 3)
- **Gap:** DERC định nghĩa ERC giám sát yếu nhưng chỉ làm trên text. Trong cảnh đa modality, tín hiệu audio-visual có thể giúp xác định câu nào trong hội thoại mang cảm xúc nào của tập nhãn. Chưa có bài nào làm DERC đa modality.
- **Data (C2, C5d):** MELD (A+V+T; feature đang trích), M3ED (A+T), EmoryNLP (text). Nhãn cấp hội thoại dẫn xuất tự động theo đúng cách của DERC.
- **Kịch bản bị scoop cần kiểm:** (a) multiple-instance learning đa modality cho ERC; (b) ERC học từ nhãn cấp hội thoại hoặc cấp video; (c) học nhãn một phần (partial-label) cho cảm xúc đa modality; (d) weakly-supervised MSA từ nhãn video (MOSI/MOSEI); (e) MIL cho sentiment văn bản (nhãn tài liệu → câu), một tiền lệ cũ trong affective cần trích dẫn.
- **Phép thử premise (CPU):** trên MELD và M3ED, huấn luyện với tập nhãn cấp hội thoại so với nhãn đầy đủ (đo khoảng cách = headroom), và đo xem đa modality thu hẹp khoảng cách đó nhiều hơn text không. Tiêu chí đặt trước khi chạy.
- **Rủi ro C3:** DERC-PL không có code công khai → phải tự cài lại; baseline giám sát yếu ở miền khác (PLOT, ItS2CLR) có code.
