# Decode What You Evaluate: giải mã tập nhãn theo emotion wheel, không huấn luyện, cho nhận diện cảm xúc đa phương thức từ vựng mở

*Báo cáo nội bộ tính đến 2026-11-25. Mọi số liệu lấy từ các report trong `explore/ovmer/`; số của bài khác lấy từ full text đã đọc (ghi trong `literature/reading_log.md`). Đây chưa phải bản thảo nộp; phần thực nghiệm chính còn đang chạy.*

---

## Abstract

**Bài toán.** Open-vocabulary multimodal emotion recognition (OV-MER) yêu cầu MLLM đọc một clip (video, âm thanh, phụ đề) rồi trả về một *tập* nhãn cảm xúc tự do. Benchmark chấm tập này bằng EW-F, một F-score tính trên các nhóm cảm xúc của 5 emotion wheel.

**Lệch giữa cách sinh và cách chấm.** Các MLLM sinh câu trả lời theo từng token, mỗi lần một chuỗi. Metric lại chấm cả tập nhãn và gộp các từ đồng nghĩa vào cùng một nhóm. Các phương pháp mạnh nhất hiện nay thu hẹp khoảng cách này bằng huấn luyện RL tốn kém.

**Đề xuất: CONSTRUCT**, một bộ giải mã không huấn luyện:
- lấy N câu trả lời ngẫu nhiên từ pipeline chuẩn của benchmark;
- ước lượng điểm EW-F kỳ vọng của mỗi tập ứng viên bằng Monte Carlo;
- dựng tập nhãn cuối bằng tìm kiếm tham lam (minimum Bayes risk trên không gian các tập ghép được).

**Kết quả hiện có.** Trên OV-MERD+ với Qwen2.5-Omni-7B zero-shot và extractor đúng protocol chính thức, không dùng nhãn nào của tập test:

| So sánh | Kết quả |
|---|---|
| GREEDY → CONSTRUCT | 60,46 → **68,91** (+8,45, CI 95% [+6,34; +10,73]) |
| So với AffectGPT (SFT, 62,52) và AffectGPT-R1 (SFT + RL, 68,39) | Cao hơn cả hai |
| So với MER-R1 (70,68) | Thấp hơn |
| So với luật gộp của hai đội thắng challenge, khi dùng N = 16 mẫu | Hòa |
| Khi ít mẫu (N ≤ 8) | Dẫn rõ |

**Dư địa còn lớn.** Tập con tốt nhất dựng từ chính các mẫu (oracle, nhìn ground truth) đạt 91,48. Nút thắt vì vậy là tín hiệu dùng để chọn nhãn. Báo cáo kết thúc bằng các cải tiến có cơ sở cho nút thắt này.

## 1. Introduction

Nhận diện cảm xúc đa phương thức đang chuyển từ phân loại trên tập nhãn đóng sang **hiểu cảm xúc bằng mô hình sinh**. Bài tổng kết MER2026 (arXiv 2604.19417) mô tả bước chuyển này là "from discriminative emotion recognition to generative emotion understanding". Trong OV-MER, đầu ra là một tập nhãn tự do, ví dụ {worried, tense}, và được chấm bằng EW-F trên các emotion wheel (mục 4).

Hiện có hai cách làm phổ biến:
1. **Huấn luyện thêm MLLM:** SFT, rồi RL với reward theo metric. Đại diện là AffectGPT, AffectGPT-R1 / -RL và MER-R1.
2. **Gộp nhiều lượt sinh bằng luật thủ công:** đại diện là Agent-MER và MultiMER, hai hệ thống thắng MER2025 và MER2026.

Nhóm 1 tốn tài nguyên. Ví dụ MER-R1 dùng 16×H100, riêng RL khoảng 18 giờ. Nhóm 2 phụ thuộc ngưỡng chỉnh tay và luật viết tay; MultiMER có 211 luật chuẩn hóa từ.

**Quan sát cốt lõi.** Metric chấm tập nhãn theo **nhóm cảm xúc**: đồng nghĩa không được cộng điểm, nói thừa bị trừ precision, nói thiếu bị trừ recall. Vậy quy tắc ra quyết định tối ưu là chọn tập có **điểm EW-F kỳ vọng** cao nhất. Ta có thể ước lượng điểm này ngay lúc suy luận từ chính các mẫu của model, không cần huấn luyện.

**Đóng góp (dạng mạnh nhất đang nhắm tới):**
1. **CONSTRUCT:** bộ giải mã MBR không huấn luyện, dùng utility EW-F và **dựng tập mới** thay vì chọn một mẫu có sẵn.
2. **Đánh giá trên một protocol thống nhất**, khớp code chính thức, so trên cùng mẫu với mọi cách gộp đã công bố.
3. **Phân tích nút thắt:** trần oracle, nhiễu của mẫu, chi phí theo N. Từ đó đề xuất bước tự kiểm chứng (mục 8).

**Dạng yếu nhất còn lại** nếu các cải tiến không thành: một bộ giải mã có cơ sở lý thuyết, không cần luật viết tay, ngang các heuristic thắng challenge khi N = 16 và tốt hơn khi N nhỏ. [Đánh giá] Dạng này chưa đủ mạnh cho tiêu chí "phương pháp mới cải thiện nhất quán" của dự án.

## 2. Related Work

**OV-MER và các mô hình huấn luyện.**
- **OV-MER** (arXiv 2410.01495) đề xuất bài toán và benchmark. **AffectGPT** (ICML 2025) đạt 62,52 trên OV-MERD+.
- **AffectGPT-R1 / -RL** (arXiv 2508.01318, 2605.06126) thêm GRPO trên Qwen2.5-7B, đạt 68,39 (68,91 khi thêm data). Theo `AffectGPT-RL/statistic.py`, con số Stage 2 là **giá trị lớn nhất qua các checkpoint từ epoch 5 đến 60, mỗi checkpoint chấm trên chính OV-MERD+**.
- **MER-R1** (arXiv 2606.27652) tách reward recall và precision, hiệu chỉnh "slow/fast thinking", đạt **70,68**, chấm ở checkpoint cuối. Bài không ghi cỡ backbone Qwen2.5-Omni. MER-R1 đọc nhãn thẳng từ thẻ `<answer>`, không qua LLM extractor.
- **HyDRA** (arXiv 2603.16463) dùng quy trình Propose–Verify–Decide học bằng GRPO.

Tất cả đều cần huấn luyện. CONSTRUCT bổ sung cho các phương pháp này: nó áp được lên đầu ra của bất kỳ mô hình nào.

**Gộp nhiều lượt sinh trong OV-MER.**
- **Agent-MER** (ACM MM 2025, hạng 1 MER2025): chạy N = 10 lượt suy luận phân tầng, giữ nhãn xuất hiện ít nhất k lần (k = 4 chọn trên OV-MERD).
- **MultiMER** (ACM MM 2026, hạng 1 MER2026 Track 2): 26 voter; NORM_MAP gồm 211 luật viết tay; bỏ phiếu khớp chính xác có trọng số; ngưỡng τ·max với τ = 0,29; kẹp 3–6 nhãn.
- Cả hai đều bỏ phiếu **theo từ**. Không bài nào tối ưu trực tiếp điểm EW-F kỳ vọng hay dùng cấu trúc wheel để dựng tập.

**Test-time scaling cho cảm xúc.** Arxiv 2602.03873 đánh giá best-of-N, gộp có trọng số theo log-likelihood (Dirichlet mixture) và verifier GPT-4o cho nhận diện cảm xúc mơ hồ trong giọng nói (IEMOCAP, MSP-Podcast, CREMA-D). Đây là tiền lệ gần nhất về "sinh nhiều lượt rồi gộp" trong affective computing. Khác ta: tập nhãn đóng, chỉ giọng nói, verifier là model ngoài chấm cả câu trả lời, không có bước dựng tập theo metric.

**Kiểm chứng trong cảm xúc.** Các bài dùng một dạng "kiểm chứng" khác hẳn bước tự kiểm chứng ở mục 8:
- EmObserver (arXiv 2607.21061): phán đoán đúng/sai cho phát biểu cảm xúc trên ảnh, có huấn luyện;
- arXiv 2610.31685: agent có critic, kiểm chứng bằng chứng để tìm lỗi;
- Emotional Rationale Verifier (arXiv 2510.23506): verifier dùng làm reward khi huấn luyện.

**Minimum Bayes risk ở lĩnh vực khác.** MBR là quy tắc giải mã quen thuộc trong dịch máy và sinh văn bản. *(Cần bổ sung trích dẫn cụ thể sau khi đọc full text; báo cáo này chưa trích theo trí nhớ.)*

## 3. Task Formulation

Cho clip x gồm video, âm thanh và phụ đề. Mô hình trả về tập nhãn Ŷ ⊂ 𝒱, với 𝒱 là từ vựng mở. Ground truth Y* là tập nhãn do người gán; số nhãn khác nhau giữa các clip và giữa các dataset.

**Pipeline chuẩn của benchmark** (code chính thức `AffectGPT-RL/statistic.py`):
1. MLLM sinh một mô tả cảm xúc tự do d.
2. Extractor Qwen2.5-7B-Instruct rút danh sách nhãn L(d).
3. Chấm L(d) với Y* bằng EW-F.

**Bài toán giải mã.** Gọi q(·|x) là phân phối của danh sách nhãn khi chạy pipeline có lấy mẫu. Ta muốn tìm

  **Ŷ = argmax_Y E_{Y′ ~ q(·|x)} [U(Y, Y′)]**

với U là utility EW-F trên từng clip. Kỳ vọng được ước lượng bằng N mẫu (Monte Carlo):

  **R̂(Y) = (1/N′) Σᵢ U(Y, Sᵢ)**, trong đó N′ là số mẫu không rỗng.

## 4. Evaluation Metric: EW-F

Code chính thức: `wheel.py` và `ew_metric.py` trong repo AffectGPT. Bản cài lại của ta được đối chiếu từng dòng, ở `explore/ovmer/ew_metric.py`.

1. **Ánh xạ nhãn.** Mỗi nhãn đi qua bảng chuẩn hóa dạng từ, rồi bảng gom đồng nghĩa, rồi bảng tra nhóm cấp 1 trên từng wheel w (5 wheel, file `wheel_mapping.npz`). Nhãn ngoài từ vựng bị bỏ; ví dụ "neutral" không có trong từ vựng.
2. **Theo từng clip i và wheel w.** Với G = nhóm của Y*ᵢ và Q = nhóm của Ŷᵢ:
   - Pᵢ,w = |G ∩ Q| / |Q|, Rᵢ,w = |G ∩ Q| / |G|.
   - Clip có G rỗng thì bỏ qua; Q rỗng thì P = R = 0.
3. **Gộp trên cả tập.** P̄_w và R̄_w là trung bình trên các clip. F_w = 2 P̄_w R̄_w / (P̄_w + R̄_w). **EW-F = trung bình F_w trên 5 wheel.** Benchmark báo level 1 (nhóm thô); level 2 (nhóm mịn) dùng làm kiểm tra phụ.

**Hệ quả:** đồng nghĩa không cộng điểm; nói thừa nhóm bị trừ P; thiếu nhóm bị trừ R.

Vì P̄ và R̄ được lấy trung bình *trước* khi tính F, metric này **không tách được thành tổng điểm từng clip**. Mức đáng giá của một đơn vị recall so với một đơn vị precision là (P̄/R̄)²; con số này phụ thuộc toàn tập (xem mục 6.3).

**Thống kê.** Bootstrap ghép cặp trên clip, 2.000 lần, báo CI 95%.

## 5. Datasets

| Tập | Số clip | Nguồn | Ground truth | Nhãn/clip (TB) | Vai trò |
|---|---|---|---|---|---|
| **OV-MERD+** | 532 | Phim truyền hình Trung Quốc (MER2026, kho gán nhãn bởi người) | Nhãn mở, người gán (`ovmerdplus-emotion.csv`) | 3,84 | **Benchmark chính**, so với số đã công bố |
| **MER2025-FG** | 1.000 | Cùng kho với OV-MERD+, **0 clip trùng** | Nhãn mở, người gán (`mer2025ov1000-emotion.csv`; cũng là dữ liệu huấn luyện RL của AffectGPT-RL) | 6,23 | Tập dev đề xuất cho các cải tiến (mục 8) |
| **EmoS-MELD** | 2.732 | MELD-train (phim Mỹ), nhãn của EmoS (ACL 2026, "dual-layer human annotation") | Ta tự dẫn xuất: các nhãn GoEmotions có ≥ 50% annotator đồng ý, bỏ neutral và unsure | 1,34 (mẫu pilot 400 clip) | Kiểm tra khả năng tổng quát, khác miền |

**Phụ đề.** Theo code chính thức (`ovmerdplus_dataset.py` dòng 74), dùng **tiếng Anh**: có cho 510/532 clip OV-MERD+ và 997/1.000 clip MER2025-FG; các clip còn lại không có phụ đề ở cả hai ngôn ngữ.

**Điều khoản.** Dữ liệu MER chỉ dùng cho nghiên cứu học thuật, không phân phối lại. Repo chỉ chứa số tổng hợp.

## 6. CONSTRUCT Framework

### 6.1. Pipeline

```
Clip (video + audio + English subtitle)
  │
  ├─[MLLM] Qwen2.5-Omni-7B thinker, bf16, zero-shot, official OV-MER prompt
  │     1 greedy description + N = 16 sampled descriptions (T = 1.0, top_p = 0.95), max 512 tokens
  │     cut invented dialogue turns ("Human:" …), rule `turn`
  │
  ├─[Extractor] Qwen2.5-7B-Instruct via vLLM, exactly as AffectGPT-RL/statistic.py
  │     (prompt with "at most four labels", T 0.7, top_p 0.8, repetition_penalty 1.05, max_tokens 512, no translation)
  │     → 16 label lists S₁…S₁₆ (+ greedy list)
  │
  └─[Decoder] CONSTRUCT  →  final label set Ŷ  →  EW-F
```

Bước MLLM và bước extractor giống cách benchmark chạy MLLM zero-shot. Đóng góp nằm ở bước giải mã.

### 6.2. Thuật toán CONSTRUCT (đóng băng, không tham số học)

1. **Pool ứng viên:** các nhãn xuất hiện trong các mẫu (cộng nhãn greedy), chỉ giữ nhãn trong từ vựng wheel, xếp theo số mẫu chứa nhãn, lấy tối đa 12.
2. **Utility trên từng clip:** U(Y, S) = trung bình trên các wheel (bỏ wheel mà S không có nhóm) của 2|G_w(S) ∩ G_w(Y)| / (|G_w(S)| + |G_w(Y)|).
3. **Dựng tập tham lam:** bắt đầu từ Y = ∅. Mỗi bước thêm nhãn làm R̂(Y) tăng nhiều nhất. Dừng khi không nhãn nào làm R̂ tăng, hoặc đã có 6 nhãn.
4. **Dự phòng:** nếu Y rỗng, lấy ứng viên phổ biến nhất, hoặc danh sách greedy.

**So với MBR thường:** MBR thường chỉ chọn một trong các mẫu. CONSTRUCT tìm trong **các tổ hợp ghép từ pool**, nên giữ được phần các mẫu cùng đồng ý và bỏ phần mỗi mẫu nói lệch một kiểu. Vì utility tính theo nhóm cảm xúc, CONSTRUCT tự bỏ nhãn đồng nghĩa và tự quyết số nhãn.

### 6.3. Biến thể CONSTRUCT-D

- **Utility:** thay F bằng F_β, với β > 1 ưu tiên recall.
- **Số nhãn tối đa:** 8.
- **Chọn β:** từ lưới {0,5; 0,75; 1; 1,25; 1,5; 2; 2,5; 3; 4}, bằng cross-fitting 5 fold trên nhãn của tập.

Lý do: metric gộp trên cả tập, nên giá trị của recall so với precision là (P̄/R̄)². Con số này phụ thuộc mật độ nhãn của người gán (3,84 so với 6,23 so với 1,34 nhãn/clip), trong khi mẫu của model luôn có khoảng 2,5 nhãn.

**Vì dùng nhãn, CONSTRUCT-D chỉ dùng cho phân tích; dòng benchmark dùng CONSTRUCT không tham số.**

### 6.4. Các baseline gộp (cùng mẫu)

| Baseline | Cách gộp |
|---|---|
| GREEDY | Chỉ dùng danh sách greedy |
| TOPK | k nhãn phổ biến nhất, k = trung vị độ dài mẫu |
| MBR | Chọn một mẫu (hoặc greedy) có R̂ cao nhất |
| Agent-MER vote | Giữ nhãn xuất hiện ≥ k lần |
| MultiMER rule | NORM_MAP, bỏ phiếu giảm dần theo thứ hạng, giữ từ có điểm ≥ τ·max, kẹp 3–6 nhãn. Một model nên trọng số voter bằng nhau; tắt boost từ vựng train vì vocab đó trùng với tập chấm |

## 7. Experimental Results (tính đến 2026-11-25)

### 7.1. Kết quả chính: OV-MERD+, Qwen2.5-Omni-7B, protocol chính thức (P17)

**Dòng benchmark** (dòng của ta không dùng nhãn OV-MERD+):

| Hệ thống | Huấn luyện | Cách lấy nhãn | EW-F L1 |
|---|---|---|---|
| Omni-7B zero-shot, GREEDY (ta) | không | LLM extractor | 60,46 |
| **Omni-7B zero-shot + CONSTRUCT (ta)** | không | LLM extractor | **68,91** |
| AffectGPT | SFT 7B | LLM extractor | 62,52 |
| AffectGPT-R1 / -RL | SFT + RL 7B | `<answer>` rồi LLM extractor; *max qua epoch trên test* | 68,39 |
| MER-R1 | SFT + RL, Qwen2.5-Omni (không ghi cỡ) | đọc thẳng `<answer>` | **70,68** |

**Độ nhạy của bước trích nhãn:**

| Biến thể | CONSTRUCT |
|---|---|
| Seed 0 (chính) | 68,91 |
| Seed trích nhãn 1 | 69,01 |
| Prompt cũ (không giới hạn 4 nhãn) | 69,13 |

**Cùng 16 mẫu (CONSTRUCT − X, CI 95%):**

| X | Chênh lệch |
|---|---|
| GREEDY | +8,45 [+6,34; +10,73] |
| TOPK (64,92) | +3,99 [+2,72; +5,30] |
| MBR (68,05) | +0,87 [+0,16; +1,61] |
| MultiMER rule, τ = 0,29 như bài gốc (68,92) | −0,01 [−1,28; +1,19] |

**Với tham số chỉnh bằng cross-fitting (dùng nhãn):**

| So sánh | Chênh lệch |
|---|---|
| CONSTRUCT-D (69,07, β mode 1,25) − MultiMER rule (68,72) | +0,35 [−0,65; +1,38] |
| CONSTRUCT-D − Agent-MER vote (68,27, k = 2) | +0,80 [−0,30; +1,97] (có ý nghĩa ở seed 1 và prompt cũ) |

Ngưỡng gốc của Agent-MER (4/10 lần, tức khoảng 40%) không chuyển được sang mẫu của Omni-7B: k = 6/16 chỉ đạt 44,82.

**Đường cong theo số mẫu N** (EW-F, 1 lần lặp cross-fitting):

| N | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| CONSTRUCT-D | 55,56 | 61,80 | **66,44** | **67,90** | 69,07 |
| MultiMER rule | 48,52 | 60,50 | 64,60 | 66,97 | 68,72 |
| Agent-MER vote | 48,00 | 59,97 | 63,35 | 63,88 | 68,27 |
| MBR | 53,38 | 57,81 | 63,36 | 66,48 | 68,05 |
| TOPK | 48,00 | 56,37 | 60,52 | 64,74 | 64,92 |

Dòng N = 16 lấy từ bảng chính (5 lần lặp); các cột khác lấy từ đường cong (1 lần lặp).

### 7.2. Pilot: Qwen2.5-Omni-3B (P13, P14′, P16)

Protocol pilot lệch so với chính thức: phụ đề tiếng Trung, extractor 4-bit, dịch nhãn Trung→Anh, cắt mô tả ở dòng đầu. Các kết quả này chỉ để so giữa các bộ giải mã.

| Tập | GREEDY | TOPK | MBR | CONSTRUCT | C − TOPK | C − MBR |
|---|---|---|---|---|---|---|
| OV-MERD+ | 58,64 | 66,27 | 68,00 | 69,27 | +3,00 [+1,75; +4,22] | +1,27 [+0,54; +2,07] |
| MER2025-FG | 50,95 | 58,43 | 63,02 | 66,04 | +7,61 [+6,51; +8,80] | +3,02 [+2,35; +3,75] |
| EmoS-MELD (400) | 56,29 | 57,65 | 59,13 | 59,55 | +1,89 [+0,46; +3,42] | +0,42 [−0,62; +1,46] |

**So với các luật gộp trên cùng mẫu (pilot):**
- CONSTRUCT thắng Agent-MER và MultiMER trên OV-MERD+ (+1,1 đến +1,2) và EmoS-MELD (+1,6 đến +2,1).
- CONSTRUCT **thua MultiMER trên MER2025-FG** (−1,73 [−2,65; −0,85]). Nguyên nhân: ground truth dày, CONSTRUCT lệch về precision (P 73 / R 60).

**P16** (CONSTRUCT-D, calib 30% / test 70%, 20 lần chia), hơn luật gộp tốt hơn đã hiệu chỉnh:

| Tập | Chênh lệch |
|---|---|
| OV-MERD+ | +1,11 [−0,66; +2,87] |
| MER2025-FG | +1,02 [−0,00; +2,14]; điểm thua cũ được sửa |
| EmoS-MELD | +1,40 [−1,21; +4,19] |

Tiêu chí đăng ký trước (CI > 0 trên ≥ 2/3 tập) **không đạt**. Nguyên nhân: tập test chỉ 70%, và thiếu phân tích power trước khi đăng ký.

### 7.3. Phân tích nút thắt (OV-MERD+, Omni-7B)

**Trần oracle** (nhìn ground truth):

| | EW-F |
|---|---|
| CONSTRUCT | 68,91 (P 69,4 / R 68,5) |
| Chọn 1 mẫu tốt nhất mỗi clip | 84,73 |
| Chọn tập con tốt nhất từ các nhãn đã sinh | **91,48** |

Ở 93,8% cặp (clip, wheel), các mẫu chứa ít nhất một nhóm đúng. **Kết luận: độ phủ của mẫu đủ; nút thắt là tín hiệu chọn.**

**Nhiễu trong mẫu:**
- Trung bình chỉ còn **14,0 mẫu không rỗng / 16** (thấp nhất 7).
- 25,6% số nhãn khác nhau nằm ngoài từ vựng, bị bỏ.
- 61,8% số nhãn trong từ vựng chỉ xuất hiện 1 lần.

**Pool bị cắt:**
- 72% số clip có hơn 12 ứng viên (trung bình 15,6).
- Nhãn ở hạng 10–12 vẫn được chọn 41 lần.
- 10,5% nhãn được chọn chỉ xuất hiện trong 1 mẫu.

**Nguyên nhân:** pool xếp hạng theo **tần suất từ**, trong khi điểm tính theo **nhóm cảm xúc**. Một nhóm được nói bằng nhiều từ khác nhau sẽ có từng từ xếp hạng thấp và dễ bị cắt.

**Chi phí:** 17 chuỗi giải mã mỗi clip (3 lượt gọi `generate`), khoảng 46 giây/clip trên A100 40 GB, so với 1 lượt sinh của các mô hình huấn luyện.

### 7.4. Tính hợp lệ của so sánh

- Metric và ground truth khớp code chính thức (đối chiếu từng dòng).
- Extractor và tham số khớp `AffectGPT-RL/statistic.py`; prompt khớp nguyên văn (kiểm tra tự động).
- **Lệch còn lại:** cắt lượt hội thoại bịa (luật `turn`). Omni-7B tự sinh lượt "Human:" ở 44–73% mô tả. Luật được chốt từ thống kê cấu trúc trước khi chấm.
- Kiểm định gộp 3 tập của P17 (đã đăng ký trước) **chưa chạy**, vì phase 1 dừng sau OV-MERD+ theo quyết định của user.

## 8. Next Improvements

Cải tiến xếp theo nút thắt ở mục 7.3. **Mọi siêu tham số và lựa chọn thiết kế sẽ được chỉnh trên tập dev MER2025-FG, đăng ký trước, rồi chấm OV-MERD+ đúng một lần.** Không chỉnh trên tập test.

1. **Tự kiểm chứng nhãn ứng viên (P18, hướng chính).**
   - Với mỗi nhãn trong pool, hỏi lại chính Omni-7B kèm clip: "Người này có cảm thấy *X* không?". Đọc xác suất "Yes" từ logits.
   - Gộp tín hiệu này với mức đồng thuận của mẫu trong CONSTRUCT.
   - Đây là tín hiệu **phân biệt**, độc lập với nhiễu khi lấy mẫu; nhắm thẳng vào khoảng cách 68,91 so với 91,48.
   - Chi phí: chỉ đọc prompt (không sinh), khoảng 10 ứng viên/clip; cần đo.
   - Novelty vòng 1 (arXiv): chưa thấy scoop. Còn thiếu OpenAlex và snowballing.
2. **Làm sạch các mẫu đóng vai đáp án** khi tính R̂(Y).
   - Hiện mỗi mẫu có trọng số 1/N′, kể cả mẫu nhiễu.
   - Ba kiểu nhiễu: mẫu rác hoàn toàn (làm giảm N′ hiệu dụng); mẫu đúng một phần có nhãn rác (làm lệch số nhãn được chọn); nhiễu hệ thống do prior của model (thiên lệch, không giảm theo N).
   - Hướng: (a) trọng số theo đồng thuận giữa các mẫu; (b) bỏ nhãn có P(Yes) thấp trong từng mẫu; (c) trọng số mẫu theo P(Yes); (d) trừ prior nhãn của model, ước lượng không cần nhãn từ tần suất trên mọi clip.
3. **Xếp hạng pool theo nhóm cảm xúc** thay vì theo từ, hoặc bỏ giới hạn 12 (trung bình chỉ 15,6 ứng viên).
4. **Ước lượng cân bằng P/R mà không cần nhãn**, thay cho β chọn bằng nhãn của CONSTRUCT-D.
5. **Chỉnh N, số ứng viên, số nhãn tối đa trên tập dev.** Báo chất lượng kèm chi phí (đường cong theo N, giây/clip), vì nhiều lượt gọi model là điểm yếu so với các phương pháp chỉ sinh một lần.
6. **P15: checkpoint SFT công khai của AffectGPT cộng CONSTRUCT** trên OV-MERD+. Trả lời câu "lợi ích do pipeline hay do model" trên đúng model của baseline 62,52.
7. **Sau khi OV-MERD+ đạt kết quả tốt:** chạy EmoS-MELD (tổng quát hóa khác miền) và kiểm định gộp đã đăng ký; 3 seed lấy mẫu trên OV-MERD+ để báo mean ± sd.
8. **Novelty:** hoàn tất OpenAlex và snowballing; trích dẫn tài liệu MBR ở dịch máy sau khi đọc full text; kiểm tra lại arXiv khoảng 20/12/2026.

**Kế hoạch ngắn hạn:** sinh mô tả cho MER2025-FG (khoảng 12,5 giờ GPU, khoảng 9 USD), rồi thiết kế P18 và các mục 2–4 trên tập dev, đăng ký trước, rồi chấm OV-MERD+ một lần.

---

*Nguồn số liệu: `explore/ovmer/p14_report_2026-11-24.md`, `p14_prior_aggregators_report_2026-11-24.md`, `p16_report_2026-11-24.md`, `p17_ovmerd_report_2026-11-24.md`; novelty: `ideas/P18_verification_novelty_2026-11-24.md`; kế hoạch: `ideas/F1_plan_2026-11-24.md`.*
