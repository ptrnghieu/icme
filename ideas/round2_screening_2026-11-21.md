# Đợt tìm ý tưởng thứ hai: nhật ký sàng lọc (2026-11-21)

## Thanh đo
User chốt ngày 2026-11-21 (CLAUDE.md §3, C3): **đóng góp chính phải là một phương pháp mới**, cải thiện nhất quán trên ≥2 dataset mở so với baseline 2024–2026 có code.

Ràng buộc:
- Chỉ dùng data mở. Compute là Kaggle 2×T4, mỗi session tối đa 11 giờ.
- Loại mọi hướng có lõi là forecasting trong hội thoại, vì trùng bài COLING (Hi-EF, arXiv 2407.16406).

## Cách làm
- **arXiv API:** khoảng 50 truy vấn. Kết quả thô lưu ở `literature/search_logs/round2_2026-11-21/`:
  - `round1.txt`, `round2.txt`: quét theo nhánh;
  - `nov_*.txt`: sàng lọc novelty theo từng hướng;
  - `abs_*.txt`: abstract các paper gần nhất.
- **Mức đọc:**
  - Sàng lọc chủ yếu dựa trên **abstract**.
  - Đọc toàn văn: Kaffeza et al., *The Illusion of Balanced MSA* (arXiv 2609.11247, Interspeech 2026).
  - Đọc phần method: TIER-MoE (arXiv 2607.27289).
- **Ý nghĩa của "loại" trong file này:** loại ở bước sàng lọc. Hướng nào cần kết luận scoop chính thức theo §4.2 thì đã ghi rõ mức đọc.

## Bức tranh chung (sự thật từ truy vấn)
Mỗi truy vấn dưới đây sắp theo ngày và trả về 120 kết quả (giới hạn của truy vấn). Kết quả cũ nhất của từng truy vấn cho biết 120 bài đó trải dài bao lâu:

| Truy vấn (`abs:`) | 120 kết quả mới nhất trải từ khoảng |
|---|---|
| "multimodal sentiment analysis" | tháng 5/2026 |
| "speech emotion recognition" | tháng 5/2026 |
| "multimodal emotion recognition" | tháng 5/2026 |
| "emotion recognition in conversation" | tháng 10/2025 |

Các chủ đề lặp lại nhiều nhất từ tháng 5/2026:
- missing modality / incomplete;
- fusion theo độ tin cậy, MoE và routing;
- xung đột giữa modality;
- adapter LLM;
- LALM bị text lấn át.

Riêng `abs:"music emotion recognition"`: chỉ có **49 kết quả cho mọi thời điểm**.

## Các hướng đã sàng lọc
| # | Hướng | Kết luận | Lý do và bằng chứng |
|---|---|---|---|
| R2-1 | LALM bị text lấn át trong SER. Ý tưởng: tạo "góc nhìn chỉ có prosody" bằng content-masked speech (low-pass, random splicing) rồi kết hợp lúc inference, không train. | **ĐÃ LOẠI** (C3/C4, không gian bão hòa) | Đến 11/2026 đã có ≥7 benchmark: DEAF 2603.18048, ALME 2602.11488, ASPIRE 2610.20114, TWIN-SER 2609.04236, CASE 2601.04564, CREMA-ASIS 2608.28966, LISTEN-style 2610.25599. Đã có ≥10 phương pháp: FAS, DAS, ACR-Net (train); back-patching 2606.18924; steering bằng audio-specialist heads 2603.06854; can thiệp hidden-state 2608.19211; acoustic cue tokens 2606.07309; contrastive decoding 2506.07233, 2603.09232, 2604.15383, 2610.28146. Đặc biệt **2607.00247** (SLT 2026) đã khảo sát cả thư viện biến đổi audio (temporal, spectral, frequency, amplitude) làm nhánh âm. Cơ chế content-masking không bị scoop trực tiếp, nhưng reviewer sẽ coi là thêm một mẹo inference giữa hàng chục mẹo. |
| R2-2 | Học trọng số fusion theo mẫu từ hiệu năng held-out hoặc out-of-fold. Đây là hướng mở mà bài Illusion nêu ở §4.3.3, kèm headroom oracle theo mẫu +8,70 (MOSI) và +7,79 (MOSEI) điểm Acc-2. | **ĐÃ LOẠI** (C4) | **TIER-MoE** (2607.27289, đọc phần method): "cross-fitted conditional modality risk", estimator học từ Brier loss out-of-fold theo subject rồi dùng để route giữa các modality. Đây đúng là ý cốt lõi, chỉ khác miền (y sinh). **CONFER** (2608.07867, abstract) cũng dùng độ tin cậy ước lượng từ hiệu năng out-of-fold. |
| R2-3 | Hiệu chỉnh theo từng người nói cho LALM zero-shot SER, không cần nhãn. | **ĐÃ LOẠI** (C3, tính hợp lệ) | Trong corpus diễn xuất (CREMA-D, RAVDESS), mỗi diễn viên có đủ mọi lớp với số lượng đều. Hiệu chỉnh prior theo người nói vì thế khai thác thiết kế dataset, và reviewer sẽ bắt lỗi. Gần nhất: EMO-TTA 2509.25495, TTA cho SER 2601.16240 (ICASSP 2026). |
| R2-4 | Chọn frame giàu cảm xúc dựa vào audio, cho AV emotion có giới hạn số frame. | **ĐÃ LOẠI** (C3, hiệu ứng nhỏ) | Clip CREMA-D, RAVDESS và MELD dài 2–4 giây. Trong pilot, phần lớn mức +18,1 điểm của visual-only đến từ việc đổi frame đầu sang frame giữa, và lấy mẫu đều nhiều frame đã bao phủ clip. Bằng chứng gần nhất: 2604.15280 (ECCV 2026), về VLM với micro-expression trong video dài hơn. |
| R2-5 | Cá nhân hóa lúc test cho AV emotion. Giả thuyết làm sắc hướng này: kênh biểu cảm mạnh khác nhau theo từng người. | **ĐÃ LOẠI** (C1: giả thuyết bị bác bỏ bằng dữ liệu; C4: phần còn lại chỉ là gia tăng) | Phân tích của Claude (`ideas/evidence/person_channel_cremad_result.txt`): 91/91 diễn viên có r_face > r_voice; ICC(1) của d_human theo diễn viên là **0,016**; split-half của d theo diễn viên r = 0,10 (Spearman–Brown 0,19); oracle chọn modality theo diễn viên chỉ hơn **+0,5 đến +1,8 điểm**. Bỏ giả thuyết này thì hướng còn lại là mở rộng TTA-CaP 2603.21309, EB-CaP 2608.06467, CLIP-AUTT 2603.27999 (đều chỉ visual) và CASP 2412.07121 (TTA cho MSA) sang AV. |
| R2-6 | Mô hình hóa từng người chấm (perspectivist) cho cảm xúc đa phương thức, dùng CREMA-D, VEATIC, AFFEC. | **ĐÃ LOẠI** (C4/C2) | Đã có mô hình hóa từng annotator cho SER: 2408.11956 (Interspeech 2024), 2510.05934, 2611.01539. Nhãn theo người chấm *và* theo modality chỉ có ở CREMA-D. |
| R2-7 | MLLM bị mất cân bằng hoặc xung đột modality trong MER (video bị bỏ qua, phân vai theo layer). | **ĐÃ LOẠI** (bão hòa) | EmoMM/CHASE 2605.01024, LAMR 2611.11907, DCR 2605.04877, MiD-ERC 2611.08860, 2611.11987. |

## Bằng chứng giữ lại cho các vòng sau
- **Illusion** (2609.11247, đọc toàn văn):
  - Các phương pháp cân bằng dựa trên gradient hay loss không thắng late concat trên MOSI/MOSEI.
  - Chọn trọng số global theo val nằm trong khoảng 0,3–0,4 điểm so với oracle.
  - Headroom oracle theo mẫu: +8,70 (MOSI) và +7,79 (MOSEI).
  - §4.3.3 liệt kê bốn hướng mở; hướng "Robustness Profiling" chưa được sàng lọc.
- **TTA cho SER** (2601.16240, ICASSP 2026, abstract): entropy minimization và pseudo-labeling thất bại vì cảm xúc vốn mơ hồ; phương pháp không cần backprop là hứa hẹn nhất.
- **Phát hiện mới trên CREMA-D** (Claude tự chạy): mô hình audio-only thắng visual-only ở 78% diễn viên chưa thấy, trong khi với người, mặt luôn dễ nhận ra hơn giọng ở 100% diễn viên.

## Hướng còn mở, cần user quyết phạm vi
- **Music emotion recognition** (audio+lyrics, audio+symbolic, đa dataset). Nhánh này ít bài hơn hẳn: 49 kết quả arXiv cho mọi thời điểm.
- Các vấn đề mở đã được nêu:
  - khả năng tổng quát hóa giữa các dataset và nhãn không đồng nhất giữa dataset (survey 2406.08809; nghiên cứu khoảng cách phân phối 2510.04688);
  - ước lượng độ bất định của phản hồi chủ quan, "significantly more challenging" (2501.11570, ECIR 2025).
- Đã có phương pháp cho: thống nhất nhãn categorical và dimensional (2502.03979), căn nhãn bằng LLM embedding (2410.11522), train chung DEAM+PMEmo (2611.08966).
- ICME 2025 đã nhận một bài về music emotion (2409.15545).
- Ứng viên dataset mở (**chưa kiểm tra L2**): DEAM, PMEmo, EMOPIA, MERGE, MTG-Jamendo mood/theme, Memo2496.

## Đợt 2b–2c (2026-11-21, sau khi user yêu cầu ưu tiên multimodal + emotion)
Truy vấn thô: `r2b_*.txt`, `r2c_*.txt`, `abs_mmtta.txt` trong `literature/search_logs/round2_2026-11-21/`.

Nguồn ngoài arXiv bị chặn ngày 2026-11-21:
- Semantic Scholar và OpenAlex trả HTTP 429 (OpenAlex báo hết ngân sách miễn phí của IP dùng chung).
- DBLP trả trang chống bot.
- Vì vậy đã dùng WebSearch cho phần ngoài arXiv.

| # | Hướng | Kết luận | Lý do và bằng chứng |
|---|---|---|---|
| D | MER/MSA chéo ngôn ngữ với shift bất đối xứng giữa modality | **ĐANG KIỂM TRA** (thí nghiệm premise P0) | Xem `ideas/D_crosslingual_modality_shift.md` sau khi có P0. |
| E1 | MER với transcript ASR (WER, confidence) | **ĐÃ LOẠI** (C4) | 2406.08353 (SLT 2024): sửa lỗi ASR + fusion có gating cho SER trên IEMOCAP/MOSI/MSP-Podcast. SWRM 2203.00257: MSA với lỗi ASR, tinh chỉnh bằng audio/visual. Santoso et al. 2021 (trích trong 2406.08353): chỉnh trọng số attention bằng confidence theo từ. 2305.16065: phân tích lỗi và confidence ASR theo cảm xúc. Bằng chứng gap vẫn giá trị cho bài khác: 2609.20081 báo WER 32,38% trên MELD và kết luận có thể đổi dấu khi dùng transcript ASR. |
| E2 | Công bằng theo nhân khẩu học trong MER đa phương thức | **ĐÃ LOẠI** (C2/C3) | Thưa (2205.08383 năm 2022: text ít bias nhất), nhưng chỉ CREMA-D có đủ nhân khẩu học, mà 91 diễn viên thì yếu về thống kê. RAVDESS chỉ có giới tính. |
| E3 | Bán giám sát cho MSA (CH-SIMS v2(u)) | **ĐÃ LOẠI** (C4) | Semi-IIN 2412.09784, CorMulT 2407.07046, AV-MC (bài CH-SIMS v2). |
| E4 | Dùng audio để trừ thành phần khẩu hình khỏi feature khuôn mặt khi nhận diện cảm xúc | **ĐÃ LOẠI** (C4) | Mariooryad & Busso, *Facial Expression Recognition in the Presence of Speech Using Blind Lexical Compensation*, IEEE TAFFC 7(4):346–359, 2016 (WebSearch kèm trang của lab MSP): bù biến thiên do khẩu hình mà không cần nhãn âm vị, đánh giá trên IEMOCAP và SEMAINE; có bài FG 2013 đi trước. Trên arXiv, các bài "khử lời nói khỏi khuôn mặt" đều thuộc mảng sinh talking-head. |
| E5 | Phân rã lớp theo circumplex, giao arousal cho audio và valence cho mặt/text | **ĐÃ LOẠI** (C3/C4 yếu) | Phụ trợ VA cho MER categorical đã có; 2609.05755 ghi rằng phụ trợ VA đơn giản cho mức cải thiện tương đương. Quy luật audio–arousal, mặt–valence đã quá quen thuộc. |
| E6 | Nhãn nhiễu trong MER, phát hiện bằng bất đồng giữa các modality; dùng phiếu cảm nhận CREMA-D để kiểm chứng | **ĐÃ LOẠI** (C4/C3 yếu) | MUG 2408.16029 (meta-learn sửa nhãn unimodal), DGDA-Net 2603.26840, MELD-FAIR 2211.15377 (video MELD lệch so với utterance), MER-SEMI (2508.05991: tinh chỉnh nhãn bằng bỏ phiếu). Cơ chế "được ít nhất một modality ủng hộ" về bản chất là co-teaching đa góc nhìn. Pilot can thiệp trên CREMA-D cho thấy trọng số theo clip tạo hiệu ứng nhỏ. |
| E7 | Phân tích nguyên nhân cảm xúc đa phương thức trong hội thoại (ECF, SemEval-2024 Task 3) | **ĐÃ LOẠI** (bão hòa) | Hơn 10 bài của đội thi SemEval-2024, cùng M3HG 2508.18740, UniMEEC 2404.00403, 2606.18893. |
| D-rủi ro | Cơ chế chung "ước lượng modality nào bị shift rồi điều chỉnh fusion" | Ghi nhận cho C4 của D | MG-MTTA 2604.24602 (ACM MM 2026): ước lượng độ trôi tương đối của từng nhánh bằng running-anchor consistency khi shift visual/text bất đối xứng. BriMPR 2511.22862 (AAAI 2026), ABPEM 2503.02221, AdaPGC 2611.03232, CARAT 2610.07499. Audio-Adaptive (CVPR 2022): hướng dạy cố định. **D chỉ đứng được nếu cơ chế khai thác cấu trúc riêng của cảm xúc/ngôn ngữ.** |

## Đợt 2d (2026-11-21)
| # | Hướng | Kết luận | Lý do và bằng chứng |
|---|---|---|---|
| F4 | **"Expressivity shift"**: khi đổi văn hóa hoặc chuyển từ posed sang tự nhiên, tín hiệu phi ngôn ngữ chủ yếu đổi về độ lớn (gain) và ngưỡng dọc các hướng phân biệt cảm xúc, chứ không đổi hướng. Phương pháp: hiệu chỉnh gain theo từng modality, không cần nhãn, trước khi fusion. | **ĐANG KIỂM TRA** (P0: MOSI↔CH-SIMS; P0b: CREMA-D→MELD) | **Sàng lọc arXiv:** không thấy bài hiệu chỉnh cường độ/expressivity cho chuyển giao chéo domain trong FER, SER hay MER (`r2d_expr.txt`, `r2d_fer.txt`). **Gần nhất:** GCC-FER/CA-FER 2606.07063 (chỉ visual, cần nhãn văn hóa, hiệu chỉnh latent theo prior văn hóa) → trùng một phần; Norface 2407.15617 chuẩn hóa theo danh tính, không phải cường độ. **Bằng chứng tâm lý (WebSearch, chưa đọc toàn văn):** Cong, Yurdum, Fischer & Sauter (Emotion, 2025) cho thấy posed được nhận ra tốt hơn spontaneous trên cùng một người, ở nhiều nền văn hóa; Behavior Research Methods 2020 (s13428-020-01443-y): posed được đánh giá cường độ cao hơn, cấu hình AU điển hình hơn. **Lưu ý thiết kế (từ phép thử tổng hợp):** nếu mọi lớp bị co đều, accuracy unimodal không đổi; hậu quả nằm ở fusion (modality bị co bị lấn át) và ở việc dự đoán trượt về neutral. Vì vậy phải đánh giá trên dự đoán đã fusion. |
| F1 | MLLM "bịa" tín hiệu cảm xúc | **ĐÃ LOẠI** (đông) | EmotionHallucer 2505.11405, 2609.11154, ESC 2607.02089, AVERE 2602.07054, OPPO 2606.25325. |
| F2 | Học liên tục cho MER | **ĐÃ LOẠI** (C3) | Truy vấn ra 0 kết quả, nhưng không tìm được cấu trúc riêng của cảm xúc; chỉ là phương pháp CL tổng quát áp vào cảm xúc. |
| F3 | Federated MER, mỗi client khác tập modality | **ĐÃ LOẠI** (C4) | 2511.00344; federated thiếu modality là chủ đề tổng quát đã có nhiều bài. |
| G2 | Tín hiệu thời gian lượt lời (độ trễ, chen ngang) cho MERC | **ĐÃ LOẠI** (C3) | Timestamp MELD lấy từ phụ đề, phim sitcom đã qua dựng, hiệu ứng dự kiến nhỏ; gần bối cảnh hội thoại của bài COLING. |
