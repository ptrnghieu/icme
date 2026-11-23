# Đợt 7 (2026-11-23): brainstorm trong lúc chờ P13

| # | Hướng | Bằng chứng đã có (đã đọc) | Trạng thái |
|---|---|---|---|
| A | **Ước lượng độ mơ hồ của nhãn từ độ đa dạng các mẫu sinh ra của MLLM** (không huấn luyện), rồi dùng để đưa ra phân bố hoặc tập nhãn khớp với mức bất đồng của người gán nhãn | d1: nhãn có annotator bất đồng chứa 66,8% lỗi (M3ED) và nhãn unimodal mâu thuẫn chứa 73% lỗi (CH-SIMS). EmoS (ACL 2026) phát hành tỉ lệ annotator theo 28 nhãn GoEmotions cho MELD (5.000 câu) và CH-SIMS v2 (4.403 clip) (đã mở file, đã đọc cột). P13 đang sinh 16 mẫu cho 400 clip EmoS-MELD → kiểm premise không tốn thêm GPU | **SỐNG**; tiền lệ cần đọc: "Disentangling Reasoning in Large Audio-Language Models for Ambiguous Emotion Prediction" (Interspeech 2026, có trong bộ quét); vùng nhãn mơ hồ (G3) đông |
| B | Mở rộng F1 sang các task sinh khác (mô tả cảm xúc EMER, MER-UniBench) nếu P13 đạt | Phụ thuộc P13 | CHỜ P13 |
| C | Cảm xúc ẩn khác biểu hiện (MIE, ACM MM 2026; MindTracker/EIA, IJCAI 2026) | Hai task mới, 0 trích dẫn (OpenAlex 2026-11-23); chỉ có DOI, chưa thấy bản arXiv hay link data | **BỊ CHẶN (C5)**: chưa tìm được data |
| D | MER-Prefer (chọn mô tả cảm xúc người ưa thích) | 2607.18508 (abstract đã đọc): hồi quy logistic chỉ dùng độ dài mô tả và danh tính model sinh đạt 65,8 WAF so với 66,8 của judge LoRA 7B trên EmoPrefer-V2; judge đã huấn luyện đi theo prior phong cách ở 63–80% cặp mâu thuẫn | **ĐÃ LOẠI**: benchmark có shortcut đã được kiểm toán; phương pháp đánh giá trên nó khó thuyết phục |
| E | MER trên dữ liệu đã xóa danh tính (DEEMO) | Xem `ideas/round6_newtasks_2026-11-23.md` | SỐNG, yếu (chưa có ý tưởng phương pháp) |
| F | Cảm xúc theo luồng (EmoS streaming) | Repo ghi phần streaming phát hành qua Google Drive; chưa kiểm link | Chưa kiểm C5; một dataset (C2 yếu) |

## Cập nhật 2026-11-23: novelty cho A → ĐÃ LOẠI (C4)
Đọc full text (pdftotext; phần abstract, method, kết quả, kết luận) 5 bài gần nhất:
- **2601.14620** "Scaling Ambiguity" (ICASSP 2026): dùng audio-language model sinh nhãn tổng hợp bằng cách "systematically varying both the prompt phrasing and the sampling temperature", coi mỗi mẫu như một annotator để dựng phân bố cảm xúc; IEMOCAP, MSP-Podcast. Kết quả: nhãn tổng hợp "exhibit higher agreement and lower entropy than human labels", lợi ích giảm ở mẫu mơ hồ cao. → **trùng phần cốt lõi** của A (độ đa dạng của mẫu sinh ra làm ước lượng phân bố nhãn của người), và là bằng chứng ngược với premise của A (mẫu sinh ra ít đa dạng hơn người).
- **2409.18339** AER-LLM (ICASSP 2025, 41 trích dẫn): prompting LLM zero/few-shot để đoán phân bố cảm xúc mơ hồ trên 3 dataset; có code. → trùng một phần.
- **2603.08230** (Interspeech 2026): coi nhận diện cảm xúc mơ hồ là suy luận phân bố trong LALM, huấn luyện bằng SFT/DPO/GRPO với mục tiêu khớp phân bố người; IEMOCAP, CREMA-D. → trùng một phần (có huấn luyện).
- **2601.18010** AmbER² (ICASSP 2026): mô hình hóa đồng thời mơ hồ do người chấm và do modality. → trùng một phần.
- **2506.19467** (EACL 2026, NLP chung): RLVR làm giảm khả năng mô hình hóa bất đồng của annotator. → miền khác, chỉ trích dẫn.
Kết luận: lõi của A đã có trong affective computing (2601.14620, AER-LLM) → A không đạt C4. Không chạy A như một hướng riêng. Mẫu sinh ra của P13 vẫn dùng được làm **phân tích phụ cho F1**: MBR có lợi hơn ở clip mà annotator EmoS bất đồng nhiều hay không (`explore/ovmer/p13_ambiguity.py`, mô tả, không có tiêu chí đạt/trượt).
