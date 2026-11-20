# Kết quả tạm thời của pilot: shard 0 và 1 (lần chạy 2, commit `2431780`)

- **Nguồn:** `pilot/results/logs/shard0_run2_2431780.log` và `shard1_run2_2431780.log`, do user chạy trên Kaggle ngày 2026-11-20.
- **Cách tổng hợp:** số liệu trích tự động từ bảng `pilot_report_partial.md` có trong mỗi log. Phần "Đánh giá" là nhận định của Claude.
- **Trạng thái:** 25/49 run. Mỗi cấu hình mới có 1–2 run, nên **chưa có kết luận go/no-go**. G3a, tiêu chí quyết định, cần ≥ 6 run AV dưới protocol A; mỗi shard chỉ có 5. Dòng `Outcome: NO-GO` in trong từng log là hệ quả của việc G3a chưa tính được, **không phải kết luận của pilot**.

## Vận hành
- 25/25 job chạy xong, không lỗi, `shard jobs still missing: 0` ở cả hai shard. Shard 0 mất 6,01 giờ, shard 1 mất 5,47 giờ.
- Thời gian mỗi job: trung bình 0,89 giờ (thấp nhất 0,56, cao nhất 0,98), nhanh hơn ước tính 1,05 giờ.

## Sự thật đã xác minh (từ log)
**1. Frame dùng lúc eval làm thay đổi lớn accuracy visual.** Cùng một checkpoint, test accuracy dùng frame giữa so với frame đầu (gộp có trọng số theo số run):

| Phương pháp | Số run | Frame giữa | Frame đầu | Chênh lệch |
|---|---|---|---|---|
| visual-only | 4 | 45,35 | 26,61 | **+18,74** |
| joint | 3 | 52,17 | 52,11 | +0,07 |
| OGM-GE α = 0,1 | 4 | 52,36 | 52,20 | +0,16 |
| OGM-GE α = 0,3 | 4 | 53,72 | 52,51 | +1,22 |
| OGM-GE α = 0,5 | 3 | 52,33 | 49,65 | +2,68 |
| OGM-GE α = 0,8 | 3 | 56,58 | 53,35 | +3,23 |

- Visual-only với frame giữa đạt 42,8–48,9% (val-selected), cùng tầm với 41,9% trong bài OGM-GE.
- Code gốc OGM-GE lấy frame qua `os.listdir(...)[i]`, mà theo tài liệu Python thứ tự của `os.listdir` là tùy ý. Vì vậy frame được chọn phụ thuộc vào hệ thống file.

**2. Chọn checkpoint theo test thổi phồng accuracy.** Trên 25 run, test-selected trừ val-selected trung bình **+1,61 điểm** (trung vị 1,21; nhỏ nhất 0,21; lớn nhất 8,20). Giá trị lớn nhất là run OGM-GE α = 0,5 ở fold actor1, seed 1: val-selected 46,24 so với test-selected 54,44.

**3. Alignment label-free với người (Q1) tăng theo α:**

| Phương pháp | Số run | Các giá trị alignment | Trung bình | Tỷ trọng visual |
|---|---|---|---|---|
| joint | 3 | −0,047; −0,032; −0,020 | −0,033 | 0,220 |
| OGM-GE α = 0,1 | 4 | 0,000; −0,092; −0,028; −0,054 | −0,043 | 0,279 |
| OGM-GE α = 0,3 | 4 | 0,029; 0,017; 0,031; −0,038 | 0,010 | 0,321 |
| OGM-GE α = 0,5 | 3 | 0,050; 0,042; 0,047 | 0,046 | 0,541 |
| OGM-GE α = 0,8 | 3 | 0,133; 0,244; 0,128 | **0,168** | 0,459 |

Targeting (OGM-GE α = 0,8 trừ joint, ghép theo split và seed): 0,149; 0,224; 0,155. Tỷ trọng visual tăng tương ứng +0,270; +0,280; +0,167.

**4. Accuracy (val-selected, %):**

| Phương pháp | Protocol A (độc lập diễn viên) | Split chuẩn |
|---|---|---|
| joint | 51,29 / 51,74 | 53,49 (1 run) |
| OGM-GE α = 0,8 | 56,06 / 56,70 | 56,99 (1 run) |
| audio-only | 50,14 / 50,77 | 48,79 / 53,23 |
| visual-only | 42,83 / 42,90 | 46,77 / 48,92 |

Lợi ích của OGM-GE so với joint, với α chọn trên val: protocol A +3,29 (shard 0) và +4,96 (shard 1); split chuẩn +3,49 (shard 0).

**5. G3b (chỉ mô tả, label-free):** +2,77 [−0,69; +6,13], +1,60 [−3,06; +5,74], +5,97 [+2,49; +9,66], +4,38 [+0,76; +8,28]. Đây là chênh lệch về tỷ lệ đoán đúng giữa clip mà mô hình dựa vào modality giống người và clip không giống.

## Đánh giá chuyên môn (chưa phải kết luận)
- **Q1 có tín hiệu dương rõ ràng.** OGM-GE với α lớn tăng mức dựa vào visual **có định hướng**: targeting dương, nghĩa là tăng nhiều hơn ở những clip mà người dựa vào mặt. Tuy vậy, độ lớn còn nhỏ (ρ ≤ 0,24). Joint thì gần như không khớp với người.
- **Frame lúc eval và việc chọn checkpoint theo test** là hai biến protocol bị giấu trong benchmark CREMA-D của mảng imbalance. Độ lớn đo được (+18,7 điểm cho visual, trung bình +1,6 điểm cho chọn theo test) đủ để làm đổi các kết luận kiểu "visual là modality yếu".
- **Luận điểm "rò rỉ diễn viên"** vẫn yếu: lợi ích của OGM-GE không giảm khi chuyển sang protocol A. Hai điều kiện GO (yếu) chưa đạt.
- **G3a rất khắt khe.** Val và test đều là diễn viên chưa thấy, nên val accuracy vốn đã dự báo tốt test accuracy. G3a đòi alignment phải dự báo **thêm** phần mà val accuracy chưa giải thích. Quy tắc này đã cố định trước khi có kết quả OGM-GE và giữ nguyên.

## Việc tiếp theo
1. Chạy shard 2 (notebook A) và shard 3 (notebook B).
2. User gửi `runs_shard0.tar` và `runs_shard1.tar`. Khi gộp lại sẽ có 10 run AV dưới protocol A, đủ để tính **G3a sơ bộ** ngay.
3. Sau shard 2 và 3: gộp đủ 49 run và chạy `analyze.py` cho kết luận go/no-go chính thức.
