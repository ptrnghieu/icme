# Bản đồ tài sản (tầng 0, quy trình v2; cập nhật 2026-11-21)

## Dataset dùng được (chi tiết và bằng chứng: `datasets/access_log.md`)
| Dataset | Mức | Modality | Ghi chú |
|---|---|---|---|
| CMU-MOSI | L2 (raw), nhãn L3 | video, audio, text gold | Tiếng Anh; nhãn sentiment từ −3 đến +3 |
| CH-SIMS | L2 (raw), nhãn L3 | video, audio, text | Tiếng Trung; **có nhãn unimodal T/A/V** |
| CH-SIMS v2 | L1+ (listing) | video, audio, text | Có nhãn unimodal; chưa tải |
| CMU-MOSEI | L1+ (mirror bên thứ ba) | video, audio, text | Máy chủ CMU trả 503 |
| MELD | L3 (dev/test), train L2 | video, audio, text | ERC 7 lớp |
| CREMA-D | L3 | video, audio | Diễn viên đóng; có phiếu cảm nhận theo từng modality |
| RAVDESS | L2 | video, audio | Diễn viên đóng |
| MERGE | L2 | audio nhạc, lời | Music emotion |
| MER2023/2024/2025 | Gated auto, **user có quyền**; chờ L3 (`datasets/verify/04_verify_gated.ipynb`) | video, audio, text ASR | Tiếng Trung in-the-wild; MER2024 có `video-labeled.zip` 4,05 GB |
| EmotionTalk | Gated auto, user chưa xác nhận đã lấy | video, audio, text | Tiếng Trung |

## Baseline 2025–2026 có code (repo kiểm tra bằng `git ls-remote` ngày 2026-11-21)
Cột "Dataset trong bài" là các tên dataset xuất hiện trong full text của bài chứa link. Nguồn: 170 bài đã được nhận ở venue chính (`literature/gap_mining/accepted_2025_2026.tsv`), tìm link GitHub và tên dataset trong full text (`code_inventory_raw.txt`).

| Bài (arXiv) | Task | Dataset trong bài | Repo | ls-remote |
|---|---|---|---|---|
| SemMSA 2609.30238 | MSA thiếu modality | MOSI, MOSEI, CH-SIMS | peacelwh/SemMSA | OK |
| MIDAS 2608.09986 | MSA thiếu modality | MOSEI, CH-SIMS | ultramarineX/MIDAS | OK |
| TLRA 2605.16889 | MSA thiếu modality, decision drift | MOSI, MOSEI | LiuXY3366/TLRA | OK |
| EBMC 2604.12518 | MSA bền vững | MOSI, MOSEI | kangverse/EBMC | OK |
| MCIB 2508.10644 | Sarcasm, MSA (shortcut) | MOSI, MOSEI | sljgkjhwe/MCIB | OK |
| LNLN (trích trong 2609.10950) | MSA nhiễu | chưa kiểm tra | Haoyu-ha/LNLN | OK |
| P-RMF (trích trong 2609.10950) | MSA | chưa kiểm tra | aoqzhu/P-RMF | **lỗi** |
| MMSA toolkit (trích trong 2608.09986) | Bộ baseline MSA | chưa kiểm tra | thuiar/MMSA | OK |
| MissMAC-Bench 2602.00811 | Benchmark thiếu modality | MOSI, MOSEI, MELD | RH-Lin/MissMAC-Bench | OK |
| SURE 2604.01916 | ERC | MELD | swaggy66/SURE | OK |
| DRKF 2508.01644 | MER hai modality | MELD | PANPANKK/DRKF | OK |
| VEGA 2508.06564 | MERC (CLIP) | MELD | dkollias/VEGA | OK |
| ProEmoTrans 2508.19533 | ERC lớp chưa thấy | MELD | KunPunCN/ProEmoTrans | OK |
| PRC-Emo 2511.07061 | ERC với LLM | MELD | LiXinran6/PRC-Emo | OK |
| Light-MER 2607.12787 | MER sinh, dưới 1B | MOSI, MOSEI, CH-SIMS, SIMS v2, MELD, MER2023/2024 | GAIR-Lab/Light-MER | OK |
| AffectAgent 2604.12735 | MER bằng agent | MELD, MER2023–2025, MOSEI, SIMS v2 | Wz1h1NG1/AffectAgent | OK |
| EmotionThinker 2601.15668 | SER reasoning (RL) | MELD | dingdongwang/EmotionThinker | OK |
| HPSU 2511.23178 | Benchmark speech | MELD, MER2025 | Ichen12/HPSU-Benchmark | OK |
| ERM-MinMaxGAP 2603.21050 | Fairness SER | MELD | zihaurpang/ERM-MinMaxGAP | OK |
| MSER 2503.21806 | Zero-shot ER chéo ngôn ngữ | MELD, MOSEI | Vincent-ZHQ/MSER | OK |

**Kiểm tra ngày 2026-11-21 (clone nông, chỉ đọc file, không chạy code):** TLRA, EBMC, SemMSA, MIDAS, LNLN và MMSA đều đọc feature dạng MMSA (`unaligned_50.pkl` / `aligned_50.pkl` / `unaligned_39.pkl`, hoặc dataloader MMSA) và có config cho MOSI, MOSEI, SIMS. MMSA còn có config cho SIMS v2. Có file LICENSE: EBMC, SemMSA, MMSA, LNLN. Không có file license: TLRA, MIDAS. **Chưa kiểm tra:** code có chạy được trên Kaggle hay không (việc của tầng 4).

## Feature dạng MMSA tải được (HTTP 206 với Range request, 2026-11-21)
| File | Kích thước | Nguồn |
|---|---|---|
| MOSI `Processed/unaligned_50.pkl` (cùng `aligned_50.pkl` 367 MB) | 554 MB | HF `tamb2203579/CMU-MOSI` (mirror bên thứ ba, README tự khai apache-2.0) |
| CH-SIMS `Processed/unaligned_39.pkl` | 1,23 GB | HF `tamb2203579/CH-SIMS` (như trên) |
| CH-SIMS v2(s) `Processed/unaligned.pkl` | 3,62 GB | HF `tamb2203579/CH-SIMSv2` (như trên) |
| MOSEI `unaligned_50.pkl` | 13,65 GB | HF `AZYoung/MOSEI_processed` (mirror bên thứ ba, không có README license) |

**Lưu ý:** CH-SIMS ⊂ CH-SIMS v2(s): 2.281/2.281 clip trùng, cùng text, cùng nhãn, cùng split (`ideas/G8_modality_conflict.md`). Không được tính là hai dataset độc lập.

## Compute
Kaggle 2×T4, 11 giờ mỗi session, khoảng 200 giờ mỗi tuần (CLAUDE.md §9).
