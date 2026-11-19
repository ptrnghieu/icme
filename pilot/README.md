# Pilot go/no-go cho ứng viên A (hạn quyết định: 28/11/2026)

Mục tiêu: kiểm tra nhanh hai câu hỏi lõi của `ideas/A_human_modality_reliability.md` §0 trước khi đầu tư full thực nghiệm.
- **Q1.** OGM-GE có tăng mức dựa vào visual *đúng ở những clip mà người thấy khuôn mặt mang thông tin*, hay chỉ tăng đồng loạt?
- **Q2 (H).** Mô hình có mức dựa vào modality khớp với người thì có tổng quát hóa tốt hơn sang **diễn viên chưa thấy** không?

## Thiết kế
| Thành phần | Nội dung |
|---|---|
| Dữ liệu | CREMA-D: 7.442 clip. Phiếu chấm của người theo voice-only, face-only, AV (`finishedResponses.csv`). |
| Protocol S | Split OGM-GE 6.698/744, trùng 91/91 diễn viên. Tách 10% train làm validation để chọn checkpoint. |
| Protocol A | 5 fold độc lập diễn viên. Pilot chạy fold 0 và 1. Validation cũng là diễn viên chưa thấy. |
| Phương pháp | `joint` (concat), `ogm_ge` (α ∈ {0,1; 0,3; 0,5; 0,8}), `audio_only`, `visual_only` |
| Seed | S: 3 seed. A: 2 seed mỗi fold. Tổng **49 run**. |
| Siêu tham số | Giữ đúng OGM-GE cho CREMA-D: SGD lr 1e-3, momentum 0,9, wd 1e-4, batch 64, 100 epoch, StepLR(70; 0,1), 1 frame, modulation ở epoch 0–50 |
| Thước đo người | r_voice và r_face = tỷ lệ người chấm đúng cảm xúc diễn khi chỉ nghe hoặc chỉ nhìn; d_human = r_face − r_voice |
| Thước đo mô hình | Shapley 2 modality, giá trị = log p(lớp đúng), loại một modality bằng cách đặt feature về 0. Kèm score ratio kiểu OGM-GE. |

**Điểm khác với code gốc OGM-GE (cố ý, đã ghi lại):**
1. Chọn checkpoint theo validation, không theo test. Test accuracy vẫn được log mỗi epoch, nên con số "chọn theo test" kiểu OGM-GE vẫn có trong báo cáo để so sánh.
2. Frame lúc train được lấy ngẫu nhiên đúng nghĩa; frame lúc eval cách đều nhau. Code OGM-GE tính chỉ số ngẫu nhiên nhưng không dùng tới.
3. Dùng ResNet-18 của torchvision (train từ đầu, không pretrained) thay cho bản sao ResNet-18 trong repo OGM-GE.

## Chạy
Yêu cầu: Python ≥ 3.9, GPU NVIDIA (T4 dùng `--amp`, tức FP16), khoảng **5 GB đĩa** (raw 3,05 GB, spectrogram khoảng 0,75 GB, frame khoảng 0,8 GB, repo khoảng 0,25 GB).

1. Cài thư viện:
```bash
pip install -r requirements.txt
```

2. Tải và tiền xử lý CREMA-D (khoảng 3 GB qua media.githubusercontent.com):
```bash
python prepare_cremad.py --root /data/cremad --workers 16
```

3. Tính độ tin cậy theo modality của người:
```bash
python human_reliability.py --root /data/cremad
```

4. Tạo split S và A:
```bash
python make_splits.py --root /data/cremad
```
Số liệu mong đợi khi đủ 7.442 clip. Claude đã tính trước bằng đúng logic này trên danh sách clip thật.
- `[standard]`: train 6.028, val 670, test 744; trùng 91/91 diễn viên.
- `[actor0]`: diễn viên 65/7/19, clip 5.316/574/1.552.
- `[actor1]`: diễn viên 66/7/18, clip 5.404/574/1.464.

Nếu số lệch so với trên thì có clip tiền xử lý lỗi. Xem `manifest.csv`, cột `err`.

5. **Đo thời gian trước (bắt buộc).** Chạy 1 run ngắn 3 epoch để biết số giây mỗi epoch trên T4:
```bash
python train.py --root /data/cremad --split standard --method ogm_ge --alpha 0.5 --seed 0 --epochs 3 --amp --out runs_timing
```

6. Tạo danh sách 49 job:
```bash
python make_jobs.py --root /data/cremad --out runs --jobs jobs.txt
```

7. Chạy trên 2 GPU. Script tự bỏ qua run đã xong, nên chạy lại được sau khi bị ngắt:
```bash
python launch.py --jobs jobs.txt --gpus 0,1 --per_gpu 1
```

8. Phân tích:
```bash
python analyze.py --root /data/cremad --runs runs --out pilot_report.md
```

Với ResNet-18 và 1 frame, một T4 thường còn dư VRAM. Nếu bước 5 cho thấy GPU chưa bị tận dụng hết thì thử `--per_gpu 2`.

## Quy tắc go/no-go (cố định trước khi chạy)
| Kết luận | Điều kiện |
|---|---|
| **GO (mạnh)** | G3a: tương quan riêng phần (alignment trên val, test acc trên diễn viên mới, sau khi kiểm soát val acc) dương, CI 95% không chứa 0. **Hoặc** G3b dương, CI không chứa 0 ở mọi fold. |
| **GO (yếu, cần bàn lại)** | Chỉ G2 cho thấy tăng đồng loạt (targeting ≈ 0 trong khi tỷ trọng visual tăng), **và** G1 cho thấy lợi ích của OGM-GE so với joint giảm hơn một nửa dưới protocol A. |
| **NO-GO** | Không đạt điều kiện nào ở trên. Khi đó dừng ứng viên A. |

## Cần gửi lại cho Claude sau khi chạy
- `pilot_report.md`
- Toàn bộ thư mục `runs/*/metrics.json`, `runs/*/preds_*.csv`, `runs/*/log.jsonl` (nén lại; không cần checkpoint)
- Số giây mỗi epoch đo được ở bước 5

## Trạng thái kiểm thử của code (2026-11-19)
Đã chạy smoke test end-to-end trên CPU trong container của Claude: 10 diễn viên × 12 clip thật, tải từ media.githubusercontent.com; 21 job, mỗi job 2 epoch × 3 batch.
- Đã xác nhận chạy được: `prepare_cremad.py` (120/120 clip dùng được; spectrogram 257×188; 2–5 frame mỗi clip), `human_reliability.py` (7.442 clip; trung bình r_voice 0,391, r_face 0,568, r_av 0,625), `make_splits.py`, `train.py`, `launch.py` (gồm cả resume: chạy lại thì bỏ qua 21/21 run đã xong), `analyze.py` (G1, G2, G3a, G3b đều in ra kết quả; G3a được kiểm trên dữ liệu tổng hợp có quan hệ dương biết trước, cho tương quan riêng phần 0,70).
- Hệ số OGM-GE được log đúng: α = 0,8 cho coeff_v ≈ 0,53–0,55; α = 0,1 cho khoảng 0,93.
- `test_ogm_equivalence.py`: so `ogm_ge_modulate` với khối code chép nguyên văn từ `main.py` của OGM-GE, cùng seed nhiễu. Hệ số trùng khớp, chênh lệch gradient tối đa = 0,00 (α = 0,1 và 0,8).
- Chưa chạy trên GPU. Đường AMP/FP16 chỉ hoạt động khi có CUDA, nên chưa được thử. Số liệu smoke test không mang ý nghĩa khoa học.
