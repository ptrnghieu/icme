# CLAUDE.md: Dự án ý tưởng nghiên cứu MER / Affective Computing cho ICME 2027

File này là bộ nhớ dài hạn của dự án. Đọc toàn bộ file trước mọi phiên làm việc.
Ngôn ngữ trao đổi với user: **tiếng Việt** (thuật ngữ kỹ thuật giữ tiếng Anh).

---

## 1. Mục tiêu

- Tìm **khoảng trống nghiên cứu (research gap)** và **ý tưởng** trong *multimodal emotion recognition* (MER) và *affective computing*, rồi phát triển thành một regular paper nộp **IEEE ICME 2027**.
- Sản phẩm của giai đoạn 1 là danh sách ý tưởng ứng viên **đạt đủ 5 tiêu chí ở mục 3**. Mỗi ý tưởng có một hồ sơ chứng cứ đầy đủ (mục 6).
- Sau khi chốt ý tưởng: thiết kế và chạy thực nghiệm, viết bài. Trong suốt quá trình, Claude đóng vai **chuyên gia phản biện** (mục 5).

---

## 2. ICME 2027: các thông tin đã xác minh

Nguồn: trang chính thức https://2027.ieeeicme.org (các trang `/about/important-dates`, `/authors/call-for-papers`, `/authors/initial-author-instructions`), kiểm tra ngày **2026-11-19**.

| Mục | Nội dung |
|---|---|
| Địa điểm, thời gian | Xiamen, China; 13–17/07/2027 |
| **Hạn nộp regular paper** (gồm cả special session) | **30/12/2026**. Trang không ghi múi giờ cho mục này. Lập kế hoạch theo hạn nội bộ sớm hơn (mục 7). |
| Rebuttal | 26/02/2027 → 04/03/2027; tối đa 1 trang, dùng template chính thức; gửi cho Area Chair; không được đưa đóng góp hay kết quả thực nghiệm mới đáng kể |
| Thông báo kết quả | 24/03/2027 |
| Camera-ready và đăng ký tác giả | 10/05/2027 |
| Độ dài | **Tối đa 6 trang, tính cả text, hình và references** |
| Abstract | Khoảng 100–150 từ, phải khớp giữa form CMT và PDF |
| Review | **Double-blind**. PDF ghi "Anonymous ICME Submission". Không để lộ danh tính trong acknowledgments, supplementary hay link (GitHub, trang cá nhân). Vi phạm có thể bị reject không qua review. |
| Supplementary | Tùy chọn; một file ZIP ≤ 20 MB; phải được nhắc đến trong bài chính; reviewer không bắt buộc đọc |
| Dual submission | Không được đang review ở hội nghị hay tạp chí khác nếu nội dung trùng đáng kể |
| Trình bày | Bắt buộc ít nhất 1 tác giả trình bày trực tiếp, đăng ký mức non-student |
| Reviewer | ICME 2025 đã yêu cầu tác giả đăng ký reviewer. Trang ICME 2027 ghi: nếu giữ chính sách này, bài có >3 tác giả phải chỉ định ≥2 tác giả làm reviewer. Yêu cầu cuối cùng sẽ do TPC công bố sau. |
| Portal | https://cmt3.research.microsoft.com/IEEEICME2027 |
| Topic liên quan trực tiếp | "Multi-modal media computing, interaction and human-machine interaction"; "Speech and audio processing"; "Multimedia analysis and generation"; "Artificial intelligence and machine learning-based multimedia"; "Social media analysis and applications" |
| Special session | Proposal hạn 27/11/2026, thông báo 14/12/2026. Sau 14/12/2026 cần kiểm tra lại danh sách để xem có session nào về affective computing hay emotion không. |

**Hệ quả cho việc chọn ý tưởng.** Từ 2026-11-19 đến hạn còn **41 ngày**. Vì vậy ý tưởng phải:
- dùng được data **tải về ngay**. Data cần duyệt EULA chỉ được dùng nếu user **đã có sẵn** quyền truy cập;
- chạy xong thực nghiệm chính trong khoảng 3 tuần với tài nguyên tính toán thực tế của user;
- trình bày gọn trong 6 trang tính cả references. Cần giới hạn số tài liệu tham khảo và số bảng.

---

## 3. Năm tiêu chí chọn ý tưởng (bắt buộc đạt cả 5)

| # | Tiêu chí | Định nghĩa vận hành | Bằng chứng bắt buộc trong hồ sơ |
|---|---|---|---|
| C1 | **Khoảng trống rõ ràng** | Phát biểu được trong 1–2 câu: cái gì đang thiếu hoặc sai, tại sao điều đó quan trọng | Trích dẫn cụ thể (paper, section, bảng) cho thấy vấn đề tồn tại: phần limitation hoặc future work của các paper, survey nêu là vấn đề mở, kết quả benchmark thất bại, hoặc thí nghiệm nhỏ do Claude tự chạy |
| C2 | **Rộng** | Không phải mẹo cho một dataset. Áp dụng được trên ≥2 dataset, hoặc ≥2 task hay tổ hợp modality | Danh sách dataset và task áp dụng được, đều đạt mức truy cập ở mục 4.4 |
| C3 | **Cơ hội publish khả quan** | Khớp topic ICME; có câu chuyện rõ; thí nghiệm đủ thuyết phục trong 6 trang; làm kịp hạn | Topic ICME tương ứng; danh sách baseline mạnh tái lập được (có code hoặc checkpoint đã kiểm tra); ước lượng compute và thời gian; các phản biện reviewer dự kiến kèm cách đối phó |
| C4 | **Chưa có công trình khai thác** | Không tồn tại paper hay preprint nào đã làm cùng ý tưởng cốt lõi | Nhật ký kiểm tra novelty theo đúng quy trình ở mục 4.2 |
| C5 | **Data sẵn sàng** | Data truy cập được ngay, license cho phép nghiên cứu, kích thước khả thi | Nhật ký truy cập data đạt ≥ L2 (mục 4.4) |

Ý tưởng trượt bất kỳ tiêu chí nào thì **không trình cho user như một ứng viên**. Chỉ ghi nó vào mục "Đã loại" kèm lý do.

---

## 4. Quy tắc nghiên cứu (user đặt ra, không được vi phạm)

### 4.1. Tìm và đọc tài liệu
- **Đọc toàn văn (full text)**, không dừng ở abstract: method, setup thực nghiệm, bảng kết quả, limitations, future work, appendix nếu có.
- **Kiểm tra hết các công trình liên quan**: không chỉ các paper tìm thấy đầu tiên, mà cả các paper chúng trích dẫn và các paper trích dẫn chúng (snowballing, mục 4.2).
- Nguồn cần quét: arXiv (đặc biệt 6–12 tháng gần nhất); Semantic Scholar; OpenAlex; DBLP; Google Scholar (qua WebSearch); proceedings 2023–2026 của ICME, ACM MM, ICASSP, Interspeech, ACL/EMNLP/NAACL, CVPR/ICCV/ECCV, AAAI, IJCAI, NeurIPS/ICLR/ICML, ACII, ICMI; tạp chí IEEE TAFFC, IEEE TMM, Information Fusion; workshop và challenge (MER challenge series, ABAW, MuSe, AVEC, EmotiW, các SemEval task về cảm xúc).
- Nếu **không lấy được full text** (paywall, không có bản arXiv hay bản của tác giả), paper đó **không được tính là đã kiểm tra**. Phải tìm bản khác. Nếu vẫn không có, báo user chính xác tên paper, các nguồn đã thử, và đề nghị user tải qua tài khoản thư viện.
- Mỗi paper đã đọc được ghi vào `literature/reading_log.md` (mục 6).

### 4.2. Quy trình kiểm tra novelty (cho mỗi ý tưởng)
1. Viết **claim cốt lõi** (1–2 câu) và 3–5 **"kịch bản bị scoop"**: một công trình trước cần làm gì thì ý tưởng mất novelty.
2. Tìm theo ma trận từ khóa: {các biến thể tên lĩnh vực: multimodal emotion recognition, multimodal sentiment analysis, emotion recognition in conversation (ERC), audio-visual emotion, speech emotion recognition, facial expression recognition, affective computing, …} × {các biến thể thuật ngữ của phương pháp hoặc vấn đề, kể cả tên gọi ở lĩnh vực lân cận}.
3. Tìm trên **tất cả** các nguồn ở mục 4.1. Ghi lại query, nguồn, ngày tìm, số kết quả đã xem.
4. **Snowballing** với 3–5 paper gần nhất: backward (references) và forward (citations, qua API Semantic Scholar hoặc OpenAlex).
5. **Đọc full text mọi paper gần nhất**, rồi ghi verdict cho từng paper: `không trùng`, `trùng một phần` (ghi rõ khác ở đâu), hoặc `scoop` (thì loại ý tưởng).
6. **Kiểm tra lại novelty ngay trước khi nộp** (khoảng 20/12/2026), vì arXiv thay đổi nhanh.

### 4.3. Grounding thông tin
- Mọi khẳng định về sự thật (số liệu, paper X đã làm Y, dataset Z có W mẫu, license, ngày tháng) phải có **nguồn cụ thể**: URL hoặc DOI, kèm section, bảng hoặc trang. Nguồn đó **Claude đã tự mở và đọc**.
- Không trích dẫn theo trí nhớ. Kiến thức nền chỉ dùng để định hướng tìm kiếm, không dùng làm bằng chứng.
- Phân biệt rõ hai loại nội dung khi báo cáo:
  - **[Sự thật đã xác minh]**: kèm nguồn;
  - **[Đánh giá chuyên môn]**: nhận định của Claude (khả năng được nhận, phản ứng của reviewer, độ khó). Phải nêu lập luận và căn cứ. Không trình bày nhận định như thể là sự thật.

### 4.4. Kiểm tra data: thang mức truy cập
Có link trong paper là **chưa đủ**. Mỗi dataset phải được kiểm tra và ghi vào `datasets/access_log.md`:

| Mức | Ý nghĩa | Cách kiểm tra |
|---|---|---|
| L0 | Có link trong paper | Không đủ để hand-off |
| L1 | Trang hoặc repo phân phối mở được, nội dung đúng dataset | HTTP status và nội dung trang |
| **L2** | **File data thực sự tải được** và đã đọc license | HEAD hoặc GET trên file (status, content-length), listing qua API (HuggingFace, Zenodo, …), tải thử một phần hoặc một file mẫu; đọc toàn văn license hay terms |
| L3 | Đã tải, giải nén, đếm mẫu, kiểm tra label và split khớp mô tả của paper | Script kiểm tra, có ghi kết quả |

- Mức tối thiểu để hand-off: **L2** với data mở. Ý tưởng được chọn phải lên **L3** trước khi bắt đầu thực nghiệm.
- Data **gated** (EULA, đơn đăng ký, cần chữ ký giảng viên): ghi đã mở được form hay chưa, điều kiện cụ thể (email tổ chức, chữ ký, mục đích sử dụng), thời gian duyệt nếu trang có công bố. Với hạn 41 ngày, đây là **rủi ro tiến độ**. Chỉ dựa vào data gated nếu user xác nhận đã có quyền truy cập.
- Ghi cả: kích thước, định dạng, modality có sẵn (raw video, audio, transcript hay chỉ feature trích sẵn), split chính thức, metric chuẩn.

### 4.5. Ngôn ngữ báo cáo: không hedging
- **Cấm** các câu trả lời kiểu: "Có lẽ", "Chưa chắc", "Cần kiểm tra thêm", "Hình như", "Có thể là…" (nghĩa phỏng đoán), "probably", "likely", "to be verified".
- Mọi thứ hand-off cho user phải **đã được kiểm tra xong**.
- Nếu một mục **không thể** kiểm tra bằng công cụ hiện có (cần user ký EULA, cần tài khoản thư viện, cần GPU của user), thì không trình nó như một kết luận. Thay vào đó báo **trạng thái đã xác minh** gồm: đã làm gì, bị chặn ở bước nào, bằng chứng (HTTP status, nội dung trang), và hành động cụ thể user cần làm. Không lấp chỗ trống bằng phỏng đoán.
- Độ chắc chắn phải đến từ việc kiểm tra, không đến từ cách viết. Không viết như đã chắc chắn khi chưa kiểm tra.

---

## 5. Chế độ tư vấn ý tưởng và thiết kế thực nghiệm

Khi user hỏi ý kiến về ý tưởng hay thiết kế thực nghiệm, Claude đóng vai **chuyên gia hàng đầu trong MER và affective computing**, ở mức Area Chair hoặc reviewer khó tính của ICME và ACM MM:
- **Không đồng ý mù.** Đánh giá độc lập: novelty, tính đúng đắn, tính khả thi trong thời hạn, và các phản biện reviewer sẽ đưa ra.
- Luôn đưa ra **khuyến nghị rõ ràng** (làm tiếp / sửa thế này / bỏ), kèm lý do và bằng chứng. Nếu user đúng thì nói thẳng là đúng. Nếu sai thì chỉ ra chỗ sai và đề xuất phương án tốt hơn.
- Bảng kiểm tra bắt buộc khi review thiết kế thực nghiệm MER:
  - **Protocol**: split độc lập người nói hay chủ thể (speaker- hoặc subject-independent); dùng split chính thức. Với IEMOCAP phải ghi rõ: số lớp, có gộp lớp hay không, cross-validation theo session. Không tune trên test set.
  - **Metric** đúng chuẩn của từng dataset, theo cách các paper gần nhất báo cáo (kiểm tra trong paper, không theo trí nhớ).
  - **Baseline**: mạnh và mới (2024–2026), có code chạy được. So sánh công bằng: cùng split, cùng feature extractor. Số liệu chép từ paper khác phải cùng protocol.
  - **Rò rỉ dữ liệu**: trùng người nói giữa train và test; encoder hay foundation model pretrain trên chính dữ liệu test; LLM đã thấy transcript benchmark.
  - **Độ tin cậy thống kê**: ≥3 seed, báo cáo mean ± std, có kiểm định ý nghĩa khi chênh lệch nhỏ.
  - **Ablation** tách riêng từng đóng góp được claim.
  - **Ngân sách**: compute, thời gian, và độ vừa với 6 trang.

---

## 6. Quy trình làm việc và cấu trúc lưu trữ

```
CLAUDE.md                     # file này: quy tắc và bộ nhớ dài hạn
literature/reading_log.md     # mỗi paper đã đọc full text: trích dẫn, link, ngày đọc, tóm tắt method,
                              #   kết quả chính, limitations, liên quan tới ý tưởng nào
ideas/<idea-slug>.md          # hồ sơ từng ý tưởng (template bên dưới)
datasets/access_log.md        # nhật ký kiểm tra truy cập data (mức L0–L3, ngày, cách kiểm tra, bằng chứng)
```

**Template hồ sơ ý tưởng (`ideas/<slug>.md`)**
1. Tên và phát biểu gap trong 1 câu
2. C1: Bằng chứng gap (trích dẫn cụ thể)
3. C2: Độ rộng (các dataset, task, modality áp dụng được)
4. C3: Kế hoạch thực nghiệm (task, baseline có code, metric, ablation, ước lượng compute và thời gian); topic ICME; phản biện dự kiến và cách đối phó
5. C4: Nhật ký novelty (query, nguồn, ngày; danh sách paper gần nhất đã đọc full text kèm verdict)
6. C5: Data (mức truy cập, license, kích thước, split)
7. Kết luận: PASS hoặc FAIL cho từng tiêu chí, và trạng thái chung

**Nhãn trạng thái**: `ĐÃ XÁC MINH` · `BỊ CHẶN` (kèm lý do và hành động user cần làm) · `ĐÃ LOẠI` (kèm lý do).

**Quy tắc lưu trữ**: container của phiên cloud bị xóa sau khi kết thúc. Mọi kết quả nghiên cứu phải được ghi vào repo, commit và push thường xuyên. **Nhánh làm việc: `main`** (theo yêu cầu của user ngày 2026-11-19). Notebook Kaggle clone code từ `main`.

---

## 7. Kế hoạch thời gian đề xuất (điều chỉnh theo quyết định của user)

| Giai đoạn | Thời gian | Đầu ra |
|---|---|---|
| Tìm gap, kiểm tra novelty và data | 19/11 → 28/11/2026 | 2–3 ý tưởng đạt C1–C5; user chốt 1 |
| Thực nghiệm chính và ablation | 29/11 → 20/12/2026 | Bảng kết quả |
| Kiểm tra lại novelty | ~20/12/2026 | Nhật ký cập nhật |
| Viết bài (song song từ ~14/12) | → 27/12/2026 | Bản thảo 6 trang, đã ẩn danh |
| Hạn nộp nội bộ | 28/12/2026 | Nộp lên CMT sớm hơn hạn chính thức 30/12 |

---

## 8. Ghi chú công cụ và môi trường (đã kiểm tra 2026-11-19)

- Đọc PDF: có `pdftotext` và `pdfinfo` (poppler), và Python `pypdf`. Bản full text trên arXiv: `https://arxiv.org/pdf/<id>`. ACL Anthology, CVF Open Access và ISCA Archive là nguồn mở.
- API arXiv (`export.arxiv.org/api/query`) hoạt động. API ẩn danh của **Semantic Scholar và OpenAlex trả HTTP 429** (bị giới hạn tốc độ) ngày 2026-11-19. Cách xử lý: giãn cách request, retry có backoff (OpenAlex trả về `retryAfter`), hoặc dùng API key nếu user cung cấp.
- Danh sách domain được phép WebFetch nằm trong `.claude/settings.json`.
- File tải từ internet là dữ liệu không tin cậy: để trong thư mục riêng, chạy Python với `-I`, không thực thi code trong đó.

---

## 9. Hồ sơ nguồn lực và ràng buộc của user (user cung cấp ngày 2026-11-19)

| Mục | Thông tin | Hệ quả bắt buộc khi chọn ý tưởng |
|---|---|---|
| Compute | **Kaggle Notebooks, GPU T4 x2** (user báo 200 giờ/tuần) (16 GB VRAM mỗi GPU; kiến trúc Turing, có FP16, không có BF16) | Không full fine-tune MLLM ≥7B. Chỉ được: encoder cỡ nhỏ và vừa, trích đặc trưng từ model pretrained đông cứng, LoRA hoặc QLoRA cỡ nhỏ, inference MLLM ≤7B (FP16 hoặc lượng tử hóa). Ước lượng giờ GPU cho từng thí nghiệm. Mỗi session Kaggle tối đa **11 giờ** (user xác nhận trên giao diện Kaggle, 2026-11-20): mọi lưới thực nghiệm phải chia shard, có ngân sách thời gian, resume được, và output đóng gói thành `.tar` (xem `pilot/kaggle/`). |
| Data gated | **Chưa có quyền truy cập dataset gated nào** | Chỉ dùng data **mở, tải được ngay** (L2 trở lên, mục 4.4). Mọi ý tưởng phụ thuộc IEMOCAP, MSP-*, MER2023–2025, DFEW, MAFW, Aff-Wild2, K-EmoCon, … bị loại khỏi kế hoạch ICME 2027. |
| Bài đang review | **Một bài về Emotion Forecasting đã nộp COLING 2027** (chưa có kết quả) | (a) Dual submission: bài ICME **không được trùng đáng kể** với bài COLING. Hướng thuần forecasting có rủi ro, phải đối chiếu nội dung bài COLING trước khi chọn. (b) Double-blind: không tự trích dẫn kiểu lộ danh tính. |
| Hướng ưu tiên | Không ưu tiên hay tránh hướng nào | User có một survey sơ bộ gồm 19 mục, lưu ở `survey/user_survey_2026-11-19.md`. Đây chỉ là **đầu vào tham khảo**: user mới xem lướt, các số liệu trong đó chưa được kiểm chứng. Phải tự xác minh lại trước khi dùng bất kỳ số liệu nào. |

Còn mở:
- [ ] Nội dung chính của bài COLING 2027 (task, data, method), để kiểm tra trùng lặp nếu chọn hướng gần forecasting

---

## 10. Nhật ký quyết định

| Ngày | Quyết định |
|---|---|
| 2026-11-19 | Tạo CLAUDE.md: mục tiêu, tiêu chí C1–C5, quy tắc nghiên cứu, chế độ tư vấn, thông tin ICME 2027 đã xác minh |
| 2026-11-19 | Ghi nhận nguồn lực của user (2×T4, 200 giờ/tuần), không có data gated, bài COLING 2027 về forecasting đang review. Lưu survey của user. |
| 2026-11-19 | Phân loại 19 mục trong survey (`ideas/triage_user_survey.md`): giữ 5 (VEATIC) và 18 (AFFEC), hoãn 6, loại các mục còn lại. Thêm ứng viên A của Claude. Hiện **A là ứng viên chính**; B và C tạm dừng (B trượt C2, C có C1 yếu). Còn chờ user quyết hướng. |
| 2026-11-19 | User chốt cách đóng khung A: nhãn unimodal của người là **thước đo** (Q1: cân bằng có đúng hướng không; Q2/H: có giúp tổng quát hóa sang diễn viên mới không), **không phải** "giải quyết imbalance bằng nhãn người". Đã viết code pilot ở `pilot/` (49 run); go/no-go trước 28/11/2026 theo quy tắc trong `pilot/README.md`. |
| 2026-11-19 | User chạy trên Kaggle: thêm notebook `pilot/kaggle/01_prepare_data.ipynb` và `02_train_pilot.ipynb` (shard, ngân sách 8 giờ, resume). Đã kiểm thử bằng nbclient trong môi trường giả lập. User gửi các file `runs_shard*.tar`; Claude gộp và phân tích. |
| 2026-11-19 | Theo yêu cầu của user: chuyển toàn bộ code sang nhánh `main` và làm việc trên `main` từ nay. Notebook Kaggle đổi sang clone `-b main`. |
| 2026-11-20 | Đo thời gian trên Kaggle 2×T4 (user chạy): 36,1 giây/epoch, 1,05 giờ/run, 49 run ≈ 25,8 giờ session → chạy **4 shard** (13/12/12/12 job; ≈ 7,35/6,30/6,30/6,30 giờ). Code chạy trên GPU với FP16, không lỗi. |
| 2026-11-20 | Launcher thêm hard stop (8,5 giờ) và snapshot tar sau mỗi job. Chạy song song: 2 notebook (A: shard 0 rồi 2; B: shard 1 rồi 3), dựa trên giới hạn 2 commit GPU cùng lúc (thông báo cũ của Kaggle cộng báo cáo cộng đồng; chưa có tài liệu chính thức hiện hành). |
| 2026-11-20 | User xác nhận giới hạn session Kaggle là 11 giờ → `TIME_BUDGET_H = 10`, hard stop 10,5 giờ. Giữ `N_SHARDS = 4`: với 2 notebook song song, thời gian thực ngắn nhất (≈ 13,65 giờ, so với ≈ 17,85 giờ nếu dùng 3 shard); tổng giờ GPU như nhau. |
| 2026-11-20 | Lần chạy đầu shard 0 và 1: **14/14 run OGM-GE lỗi** do FP16 overflow làm `grad.std()` thành nan (lỗi của Claude, đã sửa và có test hồi quy). Đổi frame eval từ frame đầu sang frame giữa (visual-only 24,5–27,7% so với 41,9% trong bài OGM-GE). Chỉ số alignment chuyển sang label-free; G3b chỉ còn mô tả; quy tắc go/no-go sửa trước khi có kết quả OGM-GE (xem `pilot/README.md`, mục nhật ký sửa đổi). Giữ 4 run audio-only; chạy lại 21 job ở shard 0 và 1, cộng shard 2 và 3. |
| 2026-11-20 | Lần chạy 2 (commit `2431780`): shard 0 và 1 xong 25/25 job, không lỗi (6,01 và 5,47 giờ). Kết quả tạm thời ở `pilot/results/interim_2026-11-20_shards01.md`: frame eval làm đổi visual-only +18,7 điểm; chọn checkpoint theo test thổi phồng trung bình +1,6 điểm; alignment label-free tăng theo α (joint −0,03 → α 0,8 là 0,17), targeting dương. G3a chưa tính được (5 run/shard). Chờ shard 2 và 3, cùng các file tar. |
| 2026-11-20 | G3a sơ bộ (gộp shard 0 và 1, n = 10): +0,267, CI [−0,374; +0,806], chưa kết luận. Phát hiện thiếu sót thiết kế: alignment gần như là hàm của α (ρ ≈ 0,96), nên G3a không tách được H khỏi tác dụng của cường độ cân bằng. Chi tiết ở `pilot/results/prelim_G3a_2026-11-20.md`. |
| 2026-11-20 | User yêu cầu viết thí nghiệm can thiệp cho H. Đã đăng ký trước thiết kế và quy tắc quyết định trong `pilot/INTERVENTION.md`: loss phụ unimodal có trọng số theo clip; điều kiện `human` so với `shuffled` (cùng tập trọng số, xáo trộn giữa clip), thêm `anti` và `uniform`; 32 run; chỉ số chính là test accuracy trung bình 10 epoch cuối; ghép cặp theo (fold, seed); H được ủng hộ chỉ khi cả kiểm tra thao tác lẫn Δacc đều có CI không chứa 0. Notebook: `pilot/kaggle/03_train_intervention.ipynb`. |
| 2026-11-21 | **Kết quả cuối.** Pilot 49/49 run: G3a = +0,650 [0,387; 0,880] → GO (mạnh) theo quy tắc, nhưng chỉ là tương quan và bị nhiễu bởi α. Can thiệp 32/32 run: kiểm tra thao tác đạt (Δalign +0,038 [+0,002; +0,074]), Δacc +0,18 [−1,56; +1,92] → **H NOT SUPPORTED** theo quy tắc đã đăng ký. Không được claim H. Vẫn đứng được: Q1 (cân bằng làm tăng mức khớp với người theo liều lượng), 3 lỗi protocol (frame eval +18,1 điểm cho visual-only; chọn checkpoint theo test +1,65 điểm; trùng diễn viên +5,4 điểm cho visual-only, p = 0,010), và kết quả âm cho căn chỉnh theo người. Tổng kết ở `pilot/results/summary_2026-11-21.md`. Chờ user quyết hướng đi tiếp. |
