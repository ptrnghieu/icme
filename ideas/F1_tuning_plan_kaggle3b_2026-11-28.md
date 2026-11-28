# F1 / CONSTRUCT: kế hoạch thực nghiệm tune cấu hình (Kaggle, Omni-3B), 2026-11-28

**Mục tiêu.** Với từng thành phần của pipeline, đo xem thay đổi nào làm EW-F tăng, thay đổi nào làm giảm, rồi chốt một cấu hình duy nhất. Cấu hình đó sau đó được kiểm tra chuyển giao sang 7B và chấm OV-MERD+ **đúng một lần**.

**Nguyên tắc** (theo CLAUDE.md §4.6 và §5):
- Tune chỉ trên dev **MER2025-FG** (1.000 clip, không trùng clip nào với OV-MERD+).
- Không chạm vào OV-MERD+ và EmoS-MELD-fresh cho tới bước cuối.
- Quy tắc quyết định được viết vào script chấm và commit **trước** khi có kết quả của tầng tương ứng.
- Mỗi biến thể ghi kèm cơ chế dự kiến và thống kê không dùng nhãn để kiểm tra cơ chế đó.

---

## 0. Trả lời hai câu hỏi về thiết kế hiện tại

**Vì sao 17 lần sinh.** 17 = 1 greedy + 16 mẫu. Mẫu greedy chỉ phục vụ **dòng baseline GREEDY**, tức mô hình zero-shot chạy theo đúng cách benchmark chạy.

[Sự thật, đo ngày 28/11 trên 532 clip OV-MERD+, không dùng nhãn] Mẫu greedy **không ảnh hưởng gì** tới CONSTRUCT:
- bỏ greedy khỏi đầu vào thì tập chọn đổi ở **0/532** clip;
- không clip nào có nhãn chỉ xuất hiện trong mẫu greedy;
- không clip nào phải dùng đến nhánh dự phòng.

→ Trong các lần chạy tune, chỉ sinh greedy **một lần** (ở cấu hình tham chiếu S0) để có dòng baseline. Các cấu hình khác bỏ greedy.

**Một hay nhiều lượt gọi model.** [Sự thật, `p17_describe.py` dòng 94–106] Mỗi clip có 3 lượt `generate`:
1. **Lượt 1 (greedy):** `do_sample = False`. Phải tách riêng vì khác chế độ giải mã.
2. **Lượt 2 và 3:** mỗi lượt sinh `num_return_sequences = 8` (`CHUNK = 8`). Khi hết bộ nhớ thì tự chia đôi.

Lý do gom 8 chuỗi vào một lượt:
- phần prefill đa phương thức (khoảng 1.634 token; bộ mã hóa hình và tiếng chạy một lần) được dùng chung cho cả 8 chuỗi;
- 8 chuỗi được giải mã song song trong một batch.

Tách thành 2 lượt × 8 vì bộ nhớ KV cache: 16 chuỗi × (1.634 + 512) token trên A100 40 GB.

So với các mô hình chỉ sinh 1 lần, chi phí thêm chủ yếu nằm ở phần giải mã, không ở prefill.

[Đánh giá] Có thể giảm còn 1 lần prefill bằng cách tính KV cache của prompt một lần rồi dùng lại cho mọi lượt. Đây là tối ưu kỹ thuật, không đổi kết quả.

---

## 1. Các nút cần tune (theo bản đồ bottleneck `ideas/F1_pipeline_bottlenecks_2026-11-28.md`)

| Tầng | Nút | Biến thể | Bottleneck nhắm tới | Cơ chế dự kiến (kiểm bằng thống kê không dùng nhãn) |
|---|---|---|---|---|
| **S** (sinh, đắt) | Cách lấy mẫu | **S0** T = 1,0, top_p = 0,95 (hiện tại) · **S1** T = 0,7, top_p = 0,95 · **S2** T = 1,0, min_p = 0,05 · **S3** T = 1,0, epsilon_cutoff = 3e-4 | B2 (21,1% mẫu thoái hóa ở 7B) | Giảm tỉ lệ chạy hết 512 token, giảm mẫu rỗng và mẫu chỉ có từ ngoài từ vựng; không làm sập độ đa dạng (độ khớp giữa các mẫu không được tăng lên gần 1) |
| S | Stop string ("Human:", "User:", …) | Bật cho mọi cấu hình | B3 | **Không phải nút tune:** phần văn bản trước lượt bịa có cùng phân phối, nên chỉ tiết kiệm token. Kiểm tra: tỉ lệ lượt bịa sau khi cắt bằng 0 |
| S | max_new_tokens | 256 / 384 / 512 | Chi phí | **Giả lập chính xác** bằng cách cắt hậu kỳ mô tả 512 token (cùng phân phối tiền tố), nên không cần chạy thêm |
| S | Số mẫu N | 4 / 8 / 16 / 32 | D3 | Chỉ S0 sinh 32 mẫu; các N nhỏ hơn lấy bằng tập con. Đường cong EW-F theo N |
| **X** (trích, vừa) | Cách giải mã của extractor | **X0** chính thức (T = 0,7, 1 lần) · **X1** greedy · **X2** trích K = 4 lần, mỗi lần làm một reference có trọng số 1/4 | C1 (20,6% nhiễu) | X1 và X2 giảm nhiễu riêng của extractor: độ khớp giữa hai lần trích tiến gần 1 (X1), hoặc giảm phương sai (X2) |
| X | Ánh xạ từ ngoài từ vựng | **O0** tắt · **O1** bật: bảng ánh xạ từ ngoài từ vựng sang từ trong từ vựng gần nghĩa nhất, dựng tự động (embedding e5 và ngưỡng cosine) chỉ từ danh sách từ, **không dùng GT**, đóng băng trước khi chấm | C3 | Giảm tỉ lệ mẫu chỉ có từ ngoài từ vựng (hiện 5,8%) |
| **W** (CPU) | Trọng số mẫu | **W0** đều · **W1** bỏ mẫu chạy hết max token · **W2** trọng số theo độ khớp trung bình với các mẫu khác | D2 | Giảm ảnh hưởng của mẫu rác |
| **U** (CPU) | Utility | **U0** F của từng clip (hiện tại) · **U1** tuyến tính hóa theo F của corpus (w_P ∝ R̄², w_R ∝ P̄², lặp điểm bất động dưới q, không dùng nhãn) · **U2** reference = hợp của m mẫu, m ∈ {2, 3} · *U3 F_β cross-fit (chỉ để phân tích, dùng nhãn)* | D4, D5 | Tập dự đoán dày hơn ở tập nhãn dày (MER2025-FG có 6,23 nhãn/clip) |
| **V** (prefill, vừa) | Tự kiểm chứng (P18) | **V0** tắt · **V1** P(Yes \| clip, "người này có cảm thấy X không?") cho mỗi nhãn trong kho, gộp vào R̂ theo trọng số λ (λ cross-fit trên dev) | B1/D1 | Dịch chuyển lựa chọn về phía nhãn được mô hình xác nhận |
| *Tùy chọn* | View sampling | 16 mẫu chia đều cho: đầy đủ, bỏ phụ đề, bỏ tiếng, bỏ hình | A1 | Tăng độ đa dạng giữa các góc nhìn so với trong cùng góc nhìn. Có cổng: không tăng thì dừng |

Giữ cố định (đã đo, không còn dư địa): kho 12 nhãn, tìm tham lam, trần 6 nhãn, luật cắt lượt bịa.

---

## 2. Thiết lập trên Kaggle

| Mục | Cấu hình |
|---|---|
| Model sinh | Qwen2.5-Omni-3B thinker, **fp16** (T4 không có bf16), 1 process mỗi GPU, 2 shard song song trên 2×T4 |
| Đầu vào | Giống P17 7B: phụ đề tiếng Anh (`subtitle_chieng.csv`), audio xen kẽ trong video, prompt chính thức |
| Extractor | Qwen2.5-7B-Instruct fp16, prompt chính thức. Thứ tự ưu tiên: (a) vLLM, tensor parallel 2 trên 2×T4; nếu cổng 0 thất bại thì (b) transformers fp16 chia trên 2 GPU, batch có left padding. **Không dùng bản 4-bit** (lệch protocol ở P14) |
| Dữ liệu | MER2025-FG 1.000 clip, cùng file `items.jsonl` mà `p17_prepare.py` tạo |
| Output | `desc_*.jsonl` và `labels_*.jsonl` đóng gói `.tar` (mô tả và nhãn, không có media). Claude chấm trên CPU trong container; dữ liệu chỉ nằm ở scratchpad, không commit |
| Chạy lại được | Ghi theo từng clip; resume theo key; ngân sách 10 giờ mỗi session, dừng cứng ở 10,5 giờ (như các notebook trước) |

**Không tái dùng output 3B cũ trên MER2025-FG (P14′).** Các run đó dùng phụ đề tiếng Trung và extractor 4-bit, nên không cùng protocol.

---

## 3. Các tầng, ngân sách và cổng

Thời gian ước lượng [Đánh giá]:
- P13 trên Kaggle 1×T4 với Omni-3B mất khoảng 17 giây/clip (1 greedy + 16 mẫu, max 256 token; CLAUDE.md, mục 10, ngày 23/11).
- Với max 512 token, ước lượng khoảng 25–35 giây/clip cho 16 mẫu.
- **Cổng 0 sẽ đo con số thật**, và ngân sách được tính lại theo đó.

| Tầng | Việc | GPU-giờ ước lượng | Cổng / đầu ra |
|---|---|---|---|
| **0** Smoke test (20 clip) | Omni-3B fp16 với transformers 5.24 trên T4; stop string; `min_p` và `epsilon_cutoff` chạy được; extractor 7B fp16 (vLLM TP2 hoặc fallback); đo giây/clip và prompt/giây | < 1 | Mọi đường chạy hết lỗi; có số đo thời gian. Nếu 1.000 clip × S0 (N = 32) vượt 2 session thì giảm dev xuống 600 clip và tính lại MDE |
| **1** Sinh | S0 (greedy + 32 mẫu), S1, S2, S3 (16 mẫu mỗi cấu hình) trên 1.000 clip | khoảng 15 (S0) + 3 × 8 ≈ 40 | Cổng cơ chế: báo tỉ lệ thoái hóa, mẫu rỗng, mẫu chỉ có từ ngoài từ vựng và độ khớp giữa các mẫu cho từng S, **trước** khi tính điểm |
| **2** Trích | X0 cho mọi S; X1 và X2 cho S0 và cấu hình S tốt nhất | khoảng 5–10 (đo ở cổng 0) | Báo phần nhiễu của extractor (như ngày 27/11) cho X0, X1, X2 |
| **3** CPU | Lưới đầy đủ {S} × {X} × {O} × {W} × {U}, cộng đường cong N trên S0 | 0 GPU | Bảng chính và bảng tương tác |
| **4** Tự kiểm chứng | V1 trên cấu hình tốt nhất của tầng 3 | khoảng 2–4 | Đăng ký trước riêng (hướng P18) |
| **5** (tùy chọn) View sampling | 500 clip | khoảng 10 | Cổng đa dạng |
| **6** Chuyển giao 7B (vast.ai) | Cấu hình chốt so với cấu hình hiện tại, trên MER2025-FG | khoảng 13 giờ A100, khoảng 10 USD | Hướng của Δ phải giữ nguyên (Δ > 0) |
| **7** Chấm cuối (vast.ai) | OV-MERD+ (1 lần) và EmoS-MELD-fresh với cấu hình chốt, 7B | khoảng 7 + 30 giờ A100 | Kết quả cho bài |

Tổng trên Kaggle khoảng 50–65 GPU-giờ, trong khả năng 200 giờ/tuần. Thời gian thực khoảng 2–3 ngày nếu chạy 2 notebook song song.

---

## 4. Quy tắc quyết định (đóng băng vào script trước tầng 3)

1. **Metric chính:** EW-F level 1 trên 1.000 clip MER2025-FG. Khoảng tin cậy 95% bằng bootstrap ghép cặp theo clip, 2.000 lần.
2. **Thứ tự quyết định:** S → X → O → W → U → N → V. Mỗi bước giữ cố định các lựa chọn đã chốt trước đó.
   - Một biến thể **tăng điểm** chỉ được nhận khi Δ so với lựa chọn hiện tại có CI > 0.
   - Biến thể **giảm chi phí** (X1, N nhỏ hơn, max token nhỏ hơn) được nhận khi không thua: cận dưới CI > −0,3.
   - Không đạt thì giữ lựa chọn đơn giản hơn.
3. **Kiểm tra tương tác:** báo luôn ô tốt nhất của cả lưới tầng 3. Nếu ô đó hơn cấu hình chọn theo thứ tự với CI > 0, chọn ô đó và ghi rõ.
4. **Kiểm tra cơ chế:** biến thể tăng điểm mà thống kê không dùng nhãn không dịch chuyển theo hướng dự kiến thì bị đánh dấu, và không được dùng làm claim cơ chế trong bài.
5. **Bảng bỏ từng thành phần** (cho bài): từ cấu hình chốt, bỏ từng thành phần một, báo Δ và CI trên dev.
6. **Power** [Đánh giá, ngoại suy]: từ sai số chuẩn ghép cặp trên OV-MERD+ (khoảng 0,37 ở 532 clip), MDE khoảng 0,8 trên 1.000 clip. Sẽ tính lại từ dữ liệu S0 trước khi đóng băng quy tắc.
7. **Đọc kết quả:** điểm dev của cấu hình chốt bị thổi phồng do chọn lọc (winner's curse), nên **không báo làm kết quả**. Kết quả cho bài chỉ lấy từ tầng 7.

---

## 5. Rủi ro

| Rủi ro | Xử lý |
|---|---|
| Cấu hình lấy mẫu tối ưu cho 3B không chuyển sang 7B (3B tự bịa lượt ở 81% mô tả, 7B ở 75,2% mẫu) | Tầng 6 kiểm tra hướng của Δ trên 7B trước khi chấm OV-MERD+. Các nút ở X, O, W, U ít phụ thuộc model hơn S |
| Extractor 7B fp16 không chạy được trên 2×T4 | Cổng 0; fallback transformers. Không dùng bản 4-bit |
| fp16 tràn số với Omni-3B | P13/P14′ đã chạy fp16 trên T4; cổng 0 kiểm tra NaN và tỉ lệ mô tả rỗng |
| Nhiều so sánh trên cùng dev | Thứ tự quyết định cố định, ngưỡng CI, kết quả cuối chỉ từ test |
| ε-sampling và min-p: tài liệu gốc chưa đọc | Đọc full text trước khi trích dẫn trong bài; không ảnh hưởng thực nghiệm |
