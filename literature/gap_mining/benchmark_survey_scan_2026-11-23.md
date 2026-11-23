# Quét benchmark SOTA còn thấp, survey và bài tổng kết challenge mới nhất (bước đầu, 2026-11-23)

Full text đã tải và đọc các đoạn liên quan (pdftotext): 2604.19417 (MER2026, ACM MM 2026), 2605.21239 (survey MER với LLM, IJCAI 2026 survey track), 2502.04424 (EmoBench-M), 2508.09210 (MME-Emotion, ICLR 2026), 2509.24322 (survey MLLM + MER), 2505.20511 (survey MERC), 2601.16449 (Emotion-LLaMAv2/MMEVerse), 2504.19423 (MER2025). Mới đọc kỹ 3 bài đầu; các bài còn lại mới tải.

## Benchmark có kết quả tốt nhất còn thấp (số liệu từ full text)
| Benchmark | Task | Số tốt nhất đã công bố | Nguồn | Ghi chú truy cập |
|---|---|---|---|---|
| MER2026 MER-Cross | cảm xúc người nghe (6 lớp) | test WAF 57,44 (Top1 fusion; Train&Val 80,91) | 2604.19417 Bảng 2 | user đã loại (gần bài COLING) |
| MER2026 MER-FG | cảm xúc từ vựng mở | AffectGPT post-train 59,54 / 60,27; zero-shot 30,83–47,38 (EW-based F) | 2604.19417 Bảng 4 | Human-OV (1.532 mẫu có nhãn người) trong MER2026, user có quyền |
| MER2026 MER-Prefer | đoán mô tả cảm xúc nào được người chọn | zero-shot tốt nhất Qwen2.5-Omni WAF 76,77 / ACC 78,74; Qwen2-Audio 36,08 | 2604.19417 Bảng 6 | train EmoPrefer-Data 574 + V2 2.096; test 379 (nhãn test chưa kiểm tra) |
| EmoBench-M | 13 kịch bản, video/audio/text | Gemini-3.0-Pro TB 70,5 (FER 69,3; CEU 62,3; SCEA 80,0); open-source FER cao nhất Qwen2-Audio-7B 59,9 | 2502.04424 mục Findings | chưa kiểm tra data |

## Hướng mở do survey hoặc bài tổng kết nêu (trích nguyên văn)
- 2605.21239, Future Directions: "it remains underexplored how to establish theoretically grounded open-vocabulary emotions, as well as how to evaluate open-ended interpretations"; "determining how subjectivity should be incorporated into models"; "which model parameters critically influence emotion perception, and the underlying reasons why affective reasoning facilitates recognition"; về văn hóa và bias: "limited exploration of concrete solutions".
- 2604.19417, MER-Cross: "although multimodal fusion yields noticeable performance gains for individual emotions, it does not improve performance on interlocutor emotions".
- 2604.19417, MER-FG: "there are no considerable differences between those trained on Human-OV versus MER-Caption+" (đánh đổi chất lượng và số lượng nhãn).
- 2502.04424, Conclusion: "Future work should prioritize enhancing contextual understanding and social awareness in MLLMs".

## Chưa làm
- Bảng đầy đủ: mỗi benchmark cần số SOTA, mức người (nếu có), truy cập data của user (C5), vừa compute T4, độ bão hòa.
- Đọc kỹ 2508.09210, 2509.24322, 2505.20511, 2601.16449, 2504.19423; thêm ABAW, MuSe, EmotiW 2025–2026; các bài limitations từ 09–11/2026 (đợt tầng 1 dừng ở 21/11/2026, chỉ lấy bài có comment venue).
