# Thí nghiệm can thiệp cho giả thuyết H (đăng ký trước, 2026-11-20)

Tài liệu này được viết **trước khi** có bất kỳ run can thiệp nào. Thiết kế, chỉ số và quy tắc quyết định dưới đây không được sửa sau khi đã thấy kết quả. Nếu buộc phải sửa, ghi rõ ngày, lý do và thời điểm so với dữ liệu, như đã làm ở `README.md`.

## Câu hỏi
**H:** mô hình dựa vào từng modality giống cách con người dựa vào trên từng clip thì tổng quát hóa tốt hơn sang diễn viên chưa thấy.

Trong pilot, mức khớp với người gần như chỉ phụ thuộc vào α của OGM-GE (ρ ≈ 0,96). Vì vậy pilot không tách được H khỏi câu "điều biến mạnh hơn thì tốt hơn" (`results/prelim_G3a_2026-11-20.md`). Thí nghiệm này **giữ nguyên cường độ cân bằng** và chỉ thay đổi việc mức nhấn từng modality **có khớp với người trên đúng clip đó hay không**.

## Cơ chế
- Mô hình giống hệt pilot: hai ResNet-18 train từ đầu, concat fusion, 1 frame, giữ nguyên siêu tham số OGM-GE.
- Thêm loss phụ unimodal có trọng số theo từng clip, đặt trên logit đã tách của mỗi nhánh (giống cách tách của OGM-GE):

  L = CE(fused, y) + λ · mean_i [ s_i · CE(out_v,i, y_i) + (1 − s_i) · CE(out_a,i, y_i) ],  với λ = 1,0

- `s_i ∈ (0, 1)` là mức nhấn nhánh visual cho clip i. Nó được tính **chỉ từ các clip trong tập train** (không dùng nhãn người của val hay test):
  - `d_i = r_face,i − r_voice,i` (độ chính xác của người khi chỉ xem mặt trừ khi chỉ nghe giọng).
  - `s_i = (rank(d_i) − 0,5) / N`, xếp hạng trong tập train. Clip có `d_i` trùng nhau được phá thế trùng bằng **một thứ tự ngẫu nhiên cố định** (seed 12345, không phụ thuộc seed của run). Hạng khi đó là một hoán vị của 1..N, nên `s_i` nhận đúng các giá trị (k − 0,5)/N, trung bình 0,5.
  - *Sửa trước khi có run nào (2026-11-20):* bản nháp đầu dùng hạng trung bình khi trùng. `test_intervention_weights.py` phát hiện rằng khi đó `anti = 1 − s` **không** có cùng tập giá trị với `human`, vì `d_i` trùng nhau rất nhiều (khoảng 10 phiếu mỗi modality). Điều đó vi phạm điều kiện cùng cường độ cân bằng nên đã sửa.

## Bốn điều kiện (cùng λ, cùng phân phối biên của trọng số)
| Điều kiện | `s_i` | Vai trò |
|---|---|---|
| `human` | xếp hạng theo `d_i` của chính clip i | Can thiệp: nhấn modality mà người thấy mang thông tin |
| `shuffled` | cùng tập giá trị nhưng hoán vị ngẫu nhiên giữa các clip train (seed 1000 + seed) | **Đối chứng chính:** cùng phân phối, mất tương ứng với clip |
| `anti` | 1 − `s_i` của `human` | Đối chứng ngược chiều |
| `uniform` | 0,5 cho mọi clip | Loss phụ unimodal thông thường, không thay đổi theo clip |

## Lưới chạy
- 4 điều kiện × 4 fold độc lập diễn viên (`actor0`–`actor3`) × 2 seed (0, 1) = **32 run**.
- Ghép cặp `(fold, seed)` → 8 cặp cho mỗi phép so sánh.
- Chạy trên Kaggle: 2 shard, mỗi shard 16 job ≈ 8 lượt × 0,9 giờ ≈ 7,2 giờ, chạy song song trên 2 notebook.

## Chỉ số
- **Accuracy chính:** trung bình test accuracy (diễn viên chưa thấy) của **10 epoch cuối** (90–99).
  - Không chọn epoch theo test.
  - Không dùng chọn checkpoint theo val, vì pilot cho thấy val chỉ có 7 diễn viên nên rất nhiễu. Ví dụ run OGM-GE α = 0,5 ở fold actor1: val-selected 46,24% so với 54,44%.
  - Accuracy val-selected được báo cáo như chỉ số phụ.
- **Mức khớp với người:** Spearman(`rel_lf`, `d_human`) trên test clip, tại checkpoint val-selected, giống pilot.
- **Tỷ trọng visual:** trung bình `kl_drop_v / (kl_drop_v + kl_drop_a)`, dùng để kiểm tra cường độ cân bằng có ngang nhau giữa các điều kiện không.

## Quy tắc quyết định (cố định)
Với mỗi cặp `(fold, seed)`, tính Δacc = acc(`human`) − acc(`shuffled`) và Δalign = align(`human`) − align(`shuffled`). Khoảng tin cậy 95% là khoảng t ghép cặp trên 8 cặp. Kèm theo là bootstrap phân tầng (cặp, rồi clip) để đối chiếu.

| Kết luận | Điều kiện |
|---|---|
| **H ĐƯỢC ỦNG HỘ (nhân quả)** | **Kiểm tra thao tác đạt:** trung bình Δalign > 0 và CI 95% không chứa 0. **Và** trung bình Δacc > 0 và CI 95% không chứa 0. |
| **H KHÔNG ĐƯỢC ỦNG HỘ** | Kiểm tra thao tác đạt, nhưng CI của Δacc chứa 0 hoặc trung bình Δacc ≤ 0. |
| **KHÔNG KẾT LUẬN ĐƯỢC** | Kiểm tra thao tác không đạt: can thiệp không làm thay đổi mức khớp với người, nên không kiểm định được H. |

**Báo cáo thêm (mô tả, không dùng để quyết định):**
- Thứ tự `human` > `uniform` > `anti` về accuracy và mức khớp.
- Δ tỷ trọng visual giữa `human` và `shuffled`. Nếu |Δ| > 0,1 thì ghi chú: cường độ cân bằng không còn ngang nhau.
- Số cặp có Δacc > 0.

## Rủi ro đã biết trước
- 8 cặp là ít. Nếu hiệu ứng thật nhỏ hơn khoảng 1 điểm, CI có thể chứa 0. Kết quả "không được ủng hộ" khi đó nghĩa là **không phát hiện được** ở độ lớn đo được, không phải bằng chứng H sai.
- `d_i` được ước lượng từ khoảng 10 phiếu chấm mỗi modality, nên có nhiễu. Nhiễu này làm hiệu ứng nhỏ đi chứ không tạo ra hiệu ứng giả.
