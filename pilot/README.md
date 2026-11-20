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

## Chạy trên Kaggle (cách user dùng)
Dùng hai notebook trong `pilot/kaggle/`: upload lên Kaggle bằng *File → Import Notebook*. Notebook tự clone code từ branch `main` của repo public `ptrnghieu/icme`, nên không cần upload code.

| Bước | Notebook và cài đặt | Làm gì | Output |
|---|---|---|---|
| 1 | `01_prepare_data.ipynb` · Accelerator *None* · Internet **On** | *Save Version → Save & Run All*. Notebook tải khoảng 3 GB và tiền xử lý 7.442 clip. Raw nằm ở `/tmp` và bị xóa sau đó. | `cremad_processed.tar` (khoảng 1,6 GB) |
| 2 | `02_train_pilot.ipynb` · **GPU T4 x2** · Internet **On** · *Add Input* = output của bước 1 | Session tương tác, `TIMING_ONLY = True`: chạy 2 job × 3 epoch song song. Notebook in ra số giây mỗi epoch và `N_SHARDS` khuyến nghị. | (chỉ in ra màn hình) |
| 3 | Như bước 2 | `TIMING_ONLY = False`, đặt `N_SHARDS` theo bước 2. Với từng `SHARD = 0, 1, …`, bấm *Save Version → Save & Run All*. | `runs_shard{SHARD}.tar` và `pilot_report_partial.md` của từng version |
| 4 | — | **Tải `runs_shard{SHARD}.tar` ngay khi mỗi version chạy xong**, trước khi chạy shard tiếp theo. Sau đó gửi tất cả cho Claude (upload vào chat). Claude gộp các shard và chạy `analyze.py`. | `pilot_report.md` |

- **Giới hạn session của Kaggle: 11 giờ** (user xác nhận trên giao diện Kaggle, 2026-11-20). `TIME_BUDGET_H = 10`, hard stop 10,5 giờ: còn 30 phút cho bước chuẩn bị (clone, giải nén data) và bước đóng gói cuối. Với `N_SHARDS = 4`, shard dài nhất (13 job, khoảng 7,35 giờ) chịu được tốc độ chậm hơn số đã đo tới 36% mà không phải hoãn job nào (với ngân sách 8 giờ cũ thì chỉ chịu được 9%).
- **Chạy song song trên nhiều notebook:** được, vì các shard độc lập (chọn job theo `index % N_SHARDS`, mỗi shard một file tar riêng).
  - Thông báo cũ của Kaggle ghi giới hạn là **1 session GPU tương tác và 2 session commit cùng lúc** ([kaggle.com/general/105509](https://www.kaggle.com/general/105509)). Một bài hướng dẫn năm 2026 cũng ghi cộng đồng đo được 2 batch session GPU cùng lúc ([fastero.com](https://fastero.com/blog/kaggle-notebooks)).
  - Cách chạy: import `02_train_pilot.ipynb` thành 2 notebook (A và B), cùng `N_SHARDS = 4` và cùng input. Notebook A chạy shard 0, rồi đến 2. Notebook B chạy shard 1, rồi đến 3.
  - Thời gian thực giảm từ khoảng 26 giờ xuống khoảng 14 giờ (7,35 + 6,30). Tổng giờ GPU tiêu tốn **không đổi**, vẫn khoảng 26 giờ.

- **Ba lớp bảo vệ trong mỗi session:**
  1. `TIME_BUDGET_H = 10`: không *bắt đầu* job mới nếu ước tính job đó kết thúc sau 10 giờ.
  2. Hard stop ở 10,5 giờ: dừng hẳn các job còn chạy để notebook kịp đóng gói và lưu output trước mốc 11 giờ.
  3. `runs_shard{SHARD}.tar` được ghi lại sau **mỗi** job xong.

  Job bị hoãn hoặc bị dừng thì không mất gì đã xong. Chỉ job đó phải chạy lại từ đầu, vì mỗi job chưa lưu checkpoint giữa chừng.
- **Shard chưa chạy hết** (dòng cuối log báo `shard jobs still missing: N` với N > 0):
  1. Upload file `runs_shard{SHARD}.tar` vừa tải về thành một **Kaggle Dataset** mới.
  2. Gắn Dataset đó làm input của notebook 02.
  3. Chạy lại *đúng shard đó*. Các run đã xong được khôi phục và bỏ qua (đã kiểm thử ở mục cuối).

  Lý do không gắn thẳng output của version cũ: theo một [thread trên diễn đàn Kaggle](https://www.kaggle.com/discussions/product-feedback/207936), chỉ version **mới nhất** của một notebook mới dùng được làm input. Notebook 02 tìm mọi file `runs_shard*.tar` ở bất kỳ đâu trong `/kaggle/input`, nên đặt file trong Dataset vẫn hoạt động.
- **Đã đóng gói thành file `.tar`:** output chỉ gồm vài file, nên không vướng giới hạn số lượng file nếu Kaggle có giới hạn này.

## Chạy trên máy riêng
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

## Thí nghiệm can thiệp cho H (thêm ngày 2026-11-20)
- Pilot không tách được H khỏi tác dụng của α (`results/prelim_G3a_2026-11-20.md`), nên có thêm một thí nghiệm can thiệp.
- Thiết kế và quy tắc quyết định được **đăng ký trước** trong [`INTERVENTION.md`](INTERVENTION.md).
- Gồm 32 run (4 điều kiện `human`/`shuffled`/`anti`/`uniform` × 4 fold × 2 seed), chạy bằng notebook `kaggle/03_train_intervention.ipynb` (2 shard song song), phân tích bằng `analyze_intervention.py`.
- Test: `test_intervention_weights.py <root>` (tính chất của trọng số) và `test_analyze_intervention.py <thư mục tạm>` (logic quyết định trên dữ liệu tổng hợp).

## Quy tắc go/no-go (bản sửa ngày 2026-11-20; `analyze.py` tự in kết luận)
| Kết luận | Điều kiện |
|---|---|
| **GO (mạnh)** | G3a, dùng alignment **label-free**: tương quan riêng phần (alignment trên val, test acc trên diễn viên mới, sau khi kiểm soát val acc) dương, CI 95% không chứa 0. |
| **GO (yếu, cần bàn lại)** | G2 cho thấy tăng visual đồng loạt (\|targeting\| < 0,05 trong khi tỷ trọng visual tăng), **và** G1 cho thấy lợi ích của OGM-GE so với joint dưới protocol A nhỏ hơn một nửa so với dưới split chuẩn. |
| **NO-GO** | Không đạt điều kiện nào ở trên. Khi đó dừng ứng viên A. |

Bản gốc ngày 2026-11-19 có thêm nhánh "**hoặc** G3b dương, CI không chứa 0 ở mọi fold", và dùng alignment dựa trên nhãn đúng. Lý do sửa nằm ở mục dưới.

## Nhật ký sửa đổi sau lần chạy shard 0 và 1 (2026-11-20)
Lần chạy đầu (commit `f8c856d`) cho ra: **14/14 run OGM-GE bị lỗi**; joint (3), audio-only (4) và visual-only (4) chạy xong.

1. **Lỗi OGM-GE khi train FP16.**
   - Nguyên nhân: `GradScaler` tăng loss scale sau mỗi 2.000 bước, nên một bước sau đó gradient bị tràn thành `inf`. Khi đó `grad.std()` = `nan`, và `normal_(0, nan)` ném `RuntimeError`.
   - Thời điểm khớp: 2.000 bước rơi vào epoch 21–24, khoảng 0,21–0,24 giờ; các run lỗi ở 0,15–0,26 giờ.
   - Cách sửa: bỏ qua modulation ở đúng bước bị tràn (`GradScaler` cũng tự bỏ qua bước đó). Số bước bị bỏ qua được log vào `ogm_steps_skipped_overflow`.
   - Kiểm thử: `test_ogm_overflow.py` tái hiện lỗi và xác nhận bản sửa không crash. `test_ogm_equivalence.py` vẫn khớp code gốc tuyệt đối khi gradient hữu hạn.
2. **Frame dùng lúc eval.**
   - Code cũ luôn eval trên frame đầu tiên (t = 0). Code gốc OGM-GE thì lấy file đầu tiên theo `os.listdir`, tức một frame tùy ý.
   - Visual-only của pilot đạt 24,5–27,7% (val-selected), so với 41,9% trong bài OGM-GE. Audio-only (50,0–52,6%) và joint (51,1–51,2%) thì khớp bài báo (52,5% và 51,7%).
   - Đã đổi sang **frame ở giữa clip**. Mỗi run giờ báo cả accuracy với frame đầu (`test_acc_other_eval_frame`), nên ảnh hưởng của lựa chọn này được đo trực tiếp.
3. **Đo mức dựa vào modality.**
   - Shapley tính trên log p(**nhãn đúng**) gắn chặt với việc đoán đúng hay sai. Một clip được đoán đúng nhờ audio thì `rel < 0`, mà người thường dựa vào mặt (d_human trung bình +0,18), nên những lần đoán đúng đó bị tính là "không khớp với người". Con số G3b = −19 điểm của lần chạy đầu phản ánh cách đo này, không phải bằng chứng về H.
   - Chỉ số chính chuyển sang **label-free**: `rel_lf = KL(p_AV‖p_chỉ-audio) − KL(p_AV‖p_chỉ-visual)`, đúng bằng hiệu Shapley với giá trị −KL tới dự đoán đầy đủ.
   - G3b chỉ còn là mô tả, vì vẫn bị nhiễu bởi chuyện mỗi nhánh mạnh hay yếu.
   - **Thời điểm sửa:** sau khi thấy kết quả joint, audio-only và visual-only của shard 0 và 1, **trước khi có bất kỳ kết quả OGM-GE nào**, và trước khi G3a được tính trên bất kỳ dữ liệu nào (cần ≥ 6 run AV dưới protocol A).
4. **Chạy lại.**
   - Launcher coi một run là xong khi có `metrics.json` **và** `dump_version ≥ 2`. Riêng audio-only thì luôn giữ, vì không dùng frame.
   - Khi chạy lại shard 0 và 1 với file tar cũ làm input, 4 run audio-only được giữ; joint, visual-only và OGM-GE được chạy lại: 11 job ở shard 0 và 10 job ở shard 1.

## Cần gửi lại cho Claude sau khi chạy
- `pilot_report.md`
- Toàn bộ thư mục `runs/*/metrics.json`, `runs/*/preds_*.csv`, `runs/*/log.jsonl` (nén lại; không cần checkpoint)
- Số giây mỗi epoch đo được ở bước 5

## Trạng thái kiểm thử của code (2026-11-19)
Đã chạy smoke test end-to-end trên CPU trong container của Claude: 10 diễn viên × 12 clip thật, tải từ media.githubusercontent.com; 21 job, mỗi job 2 epoch × 3 batch.
- Đã xác nhận chạy được: `prepare_cremad.py` (120/120 clip dùng được; spectrogram 257×188; 2–5 frame mỗi clip), `human_reliability.py` (7.442 clip; trung bình r_voice 0,391, r_face 0,568, r_av 0,625), `make_splits.py`, `train.py`, `launch.py` (gồm cả resume: chạy lại thì bỏ qua 21/21 run đã xong), `analyze.py` (G1, G2, G3a, G3b đều in ra kết quả; G3a được kiểm trên dữ liệu tổng hợp có quan hệ dương biết trước, cho tương quan riêng phần 0,70).
- Hệ số OGM-GE được log đúng: α = 0,8 cho coeff_v ≈ 0,53–0,55; α = 0,1 cho khoảng 0,93.
- `test_ogm_equivalence.py`: so `ogm_ge_modulate` với khối code chép nguyên văn từ `main.py` của OGM-GE, cùng seed nhiễu. Hệ số trùng khớp, chênh lệch gradient tối đa = 0,00 (α = 0,1 và 0,8).
- **Notebook Kaggle:** Claude chạy chính hai file `.ipynb` bằng `nbclient` trong một thư mục giả lập `/kaggle/input` và `/kaggle/working`. Bản test chỉ khác hai điểm: đổi 3 biến đường dẫn, và dùng tập 120 clip với lưới rút gọn.
  - `01`: clone đúng branch, 120/120 clip dùng được, tạo `cremad_processed.tar`.
  - `02`, chế độ đo thời gian: chạy 2 job song song và in ra `N_SHARDS` khuyến nghị.
  - `02`, shard 0/3: xong 5/14 job, tạo `runs_shard0.tar` và `pilot_report_partial.md`.
  - **Resume:** gắn `runs_shard0.tar` làm input rồi chạy lại thì báo "5 already done, 0 to run".
  - **Ngân sách thời gian:** với `--time_budget_h 0.001`, launcher không khởi động job nào và ghi lại số job hoãn.
- **Đo thời gian trên Kaggle (user chạy, 2026-11-20, commit `70415b1`):** 2× Tesla T4 15.360 MiB, 4 CPU, 2 dataloader worker mỗi job. 36,1 giây/epoch khi 2 job chạy song song, ước tính 1,05 giờ/run, 49 run ≈ 25,8 giờ session. Chạy với **`N_SHARDS = 4`**: shard 0 có 13 job ≈ 7,35 giờ; shard 1–3 mỗi shard 12 job ≈ 6,30 giờ; tất cả dưới ngân sách 8 giờ. Đây là lần đầu code chạy trên GPU với `--amp` (FP16), và không có lỗi.
- Trong log Kaggle, output của mỗi lệnh con (`launch.py`, `nvidia-smi`) bị in 2 lần, còn các dòng `print()` của Python chỉ in 1 lần. Mỗi cell chỉ gọi `launch.py` một lần, nên đây là cách Kaggle ghi log, không phải job chạy lặp.
- Trước lần đo trên Kaggle, code chỉ chạy trên CPU. Số liệu smoke test không mang ý nghĩa khoa học.
