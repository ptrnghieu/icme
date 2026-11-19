# Nhật ký kiểm tra truy cập dữ liệu

Thang mức truy cập theo CLAUDE.md §4.4: L0 = có link trong paper · L1 = trang phân phối mở được · **L2 = file tải được thật, đã đọc license** · L3 = đã tải, parse và đếm mẫu.
Mọi kiểm tra chạy từ container cloud ngày **2026-11-19**, không đăng nhập tài khoản nào.

| Dataset | Mức | Cách kiểm tra và bằng chứng | License | Ghi chú cho dự án |
|---|---|---|---|---|
| **AFFEC** (Zenodo 19408373, v2, 2026-04-03) | **L3** | API Zenodo: `access_right=open`, 13 file, tổng 21,8 GB. Đã tải `core.zip` (5,6 MB), `README.md` và `participants.tsv`. Nhãn nằm ở `sub-*/beh/*_beh.tsv`, gồm cột `f_emotion_v`, `f_emotion_a`, `p_emotion_v`, `p_emotion_a`. Đã parse 273 file. | Zenodo ghi CC-BY-4.0, README ghi CC0. Cả hai đều cho phép nghiên cứu. | 72 người, 84 stimulus (clip CREMA-D), 5.524 trial chính có đủ 4 nhãn; 60–72 lượt chấm cho mỗi stimulus. corr(felt V, perceived V) = 0,635; corr(felt A, perceived A) = 0,575. Tỷ lệ \|felt−perceived\| > 1 điểm: 30,5% (V), 42,7% (A). Felt gần trung tính hơn: \|V−5\| trung bình 1,33 (felt) so với 1,81 (perceived). Các modality: EEG, gaze, pupil, GSR, AU khuôn mặt (OpenFace), cursor. **Không có video mặt người xem dạng raw** (chỉ có đặc trưng `videostream`). |
| AFFEC devkit (github.com/itubrainlab/AFFEC) | L2 | `git clone` thành công, commit f175910 (2026-04-03). Loader đọc nhãn từ `beh/*_beh.tsv`. | CC0 | Repo `brAInlab-ITU/AFFEC` được nhắc trong README thì không truy cập công khai được (git yêu cầu đăng nhập). |
| **CREMA-D** (github.com/CheyneyComputerScience/CREMA-D) | **L2** (nhãn: **L3**) | Clone thành công (bỏ qua LFS). Tải thật qua `media.githubusercontent.com`: `1001_DFA_ANG_XX.wav` (HTTP 200, WAV PCM 16 kHz) và `1001_DFA_ANG_XX.flv` (HTTP 200, Flash Video). Có 7.442 wav và 7.442 flv. Đã parse `finishedResponses.csv`. | ODbL 1.0 + DbCL 1.0 | Có **phiếu chấm của từng người theo modality**: 219.688 phiếu, 2.443 người chấm. `queryType`: 1 = chỉ giọng, 2 = chỉ mặt, 3 = AV; khoảng 9,8 phiếu cho mỗi clip mỗi modality. Accuracy của người so với nhãn diễn: giọng 39,2%, mặt 56,9%, AV 62,5%. Nhãn đa số giữa giọng và mặt khác nhau ở 50,2% số clip. 34,9% clip có acc(mặt) > acc(giọng) + 0,3; 8,7% clip theo chiều ngược lại. |
| RAVDESS (Zenodo 1188976) | L1+ | API Zenodo: `open`, 49 file, 25,6 GB. Có URL file cụ thể; chưa tải file. | CC-BY-NC-SA-4.0 | Chưa xác định được dữ liệu validation theo từng người chấm có được phát hành công khai không. |
| **VEATIC** (Google Drive `VEATIC.zip`) | **L2** | HTTP 206 với Range request, kích thước 3.218.338.827 byte. Đọc central directory từ xa: 124 video mp4 và `rating_averaged/*_{valence,arousal}.csv`. | Trang dự án ghi bản quyền video thuộc chủ sở hữu gốc; data dùng cho nghiên cứu | Bản zip chính **chỉ có rating trung bình**. |
| **VEATIC_Psych** (Google Drive `ratings_subj.zip`) | **L3** | Đã tải đủ 191 MB và giải nén 166 file (83 video × V/A). | Như trên | Rating liên tục theo frame của **từng người chấm** cho video 0–82: 61–74 người mỗi video (trung vị 73); 264–3.930 frame mỗi video (trung vị 1.800). Cột là PID, nhận diện được cùng một người qua nhiều video. |
| ReCANVo (Zenodo 5786860) | L1+ | API Zenodo: `open`, 1 file `ReCANVo.zip` 1.055 MB, có URL tải. | CC-BY-3.0-US | Không cần kiểm tra sâu hơn vì mục 1 bị loại vì lý do khác (xem `ideas/triage_user_survey.md`). |
| K-EmoCon (Zenodo 3931963) | **Gated** | API Zenodo: `access_right=restricted`, 0 file công khai | — | User chưa có quyền truy cập, nên loại khỏi kế hoạch. |
| MEmoR | **Gated** | README: phải gửi License Agreement qua email để nhận mật khẩu OneDrive | — | Loại khỏi kế hoạch. |
| Mult²EMo | **Gated** | Trang phát hành ghi "The dataset is available upon request" | CC BY-NC-SA 4.0 | Loại khỏi kế hoạch. |
| MECESD (phần video, audio, feature) | **Gated** | README: phải ký license agreement và gửi email | Chỉ cho nghiên cứu học thuật | Phần text có trên repo, nhưng phần đa phương thức bị gated. |
| OSMO | **Không tìm thấy đường tải** | Trang osmo-emos.github.io không có link tải dataset; link duy nhất dẫn tới trang tác giả | — | Loại khỏi kế hoạch. |

## Ghi chú kỹ thuật về môi trường kiểm tra
- Proxy của container không lấy được Git LFS qua `git lfs pull` với repo chưa attach. Đường `media.githubusercontent.com/media/<owner>/<repo>/<branch>/<path>` thì tải được. Đây là giới hạn của container này, không phải giới hạn của dataset.
- Có thể liệt kê nội dung zip lớn từ xa bằng HTTP Range (Zenodo và Google Drive đều hỗ trợ), không cần tải cả file: `tools/remote_zip_ls.py` (đã đưa vào repo).
