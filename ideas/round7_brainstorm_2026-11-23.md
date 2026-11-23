# Đợt 7 (2026-11-23): brainstorm trong lúc chờ P13

| # | Hướng | Bằng chứng đã có (đã đọc) | Trạng thái |
|---|---|---|---|
| A | **Ước lượng độ mơ hồ của nhãn từ độ đa dạng các mẫu sinh ra của MLLM** (không huấn luyện), rồi dùng để đưa ra phân bố hoặc tập nhãn khớp với mức bất đồng của người gán nhãn | d1: nhãn có annotator bất đồng chứa 66,8% lỗi (M3ED) và nhãn unimodal mâu thuẫn chứa 73% lỗi (CH-SIMS). EmoS (ACL 2026) phát hành tỉ lệ annotator theo 28 nhãn GoEmotions cho MELD (5.000 câu) và CH-SIMS v2 (4.403 clip) (đã mở file, đã đọc cột). P13 đang sinh 16 mẫu cho 400 clip EmoS-MELD → kiểm premise không tốn thêm GPU | **SỐNG**; tiền lệ cần đọc: "Disentangling Reasoning in Large Audio-Language Models for Ambiguous Emotion Prediction" (Interspeech 2026, có trong bộ quét); vùng nhãn mơ hồ (G3) đông |
| B | Mở rộng F1 sang các task sinh khác (mô tả cảm xúc EMER, MER-UniBench) nếu P13 đạt | Phụ thuộc P13 | CHỜ P13 |
| C | Cảm xúc ẩn khác biểu hiện (MIE, ACM MM 2026; MindTracker/EIA, IJCAI 2026) | Hai task mới, 0 trích dẫn (OpenAlex 2026-11-23); chỉ có DOI, chưa thấy bản arXiv hay link data | **BỊ CHẶN (C5)**: chưa tìm được data |
| D | MER-Prefer (chọn mô tả cảm xúc người ưa thích) | 2607.18508 (abstract đã đọc): hồi quy logistic chỉ dùng độ dài mô tả và danh tính model sinh đạt 65,8 WAF so với 66,8 của judge LoRA 7B trên EmoPrefer-V2; judge đã huấn luyện đi theo prior phong cách ở 63–80% cặp mâu thuẫn | **ĐÃ LOẠI**: benchmark có shortcut đã được kiểm toán; phương pháp đánh giá trên nó khó thuyết phục |
| E | MER trên dữ liệu đã xóa danh tính (DEEMO) | Xem `ideas/round6_newtasks_2026-11-23.md` | SỐNG, yếu (chưa có ý tưởng phương pháp) |
| F | Cảm xúc theo luồng (EmoS streaming) | Repo ghi phần streaming phát hành qua Google Drive; chưa kiểm link | Chưa kiểm C5; một dataset (C2 yếu) |
