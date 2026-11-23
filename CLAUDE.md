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
| C3 | **Cơ hội publish khả quan** | Khớp topic ICME; có câu chuyện rõ; thí nghiệm đủ thuyết phục trong 6 trang; làm kịp hạn. **Từ 2026-11-21 (user chốt): đóng góp chính phải là một phương pháp mới**, cải thiện nhất quán trên ≥2 dataset mở so với baseline 2024–2026 có code. Bài chỉ phân tích hoặc kiểm toán không đạt. Hồ sơ phải nêu đóng góp ở dạng mạnh nhất và ở dạng còn lại nếu giả thuyết chính thất bại. | Topic ICME tương ứng; danh sách baseline mạnh tái lập được (có code hoặc checkpoint đã kiểm tra); ước lượng compute và thời gian; các phản biện reviewer dự kiến kèm cách đối phó |
| C4 | **Chưa có công trình khai thác** | **Nới từ 2026-11-21 (user chốt):** chưa có paper hay preprint nào làm cùng ý tưởng cốt lõi **trong MER và affective computing** (emotion, sentiment, affect, kể cả SER, FER, ERC, MSA, physiological). Nếu cơ chế đã có ở **miền khác** (y sinh, vision, NLP tổng quát, …) thì vẫn đạt, nhưng bắt buộc trích dẫn, nêu rõ khác ở đâu, và so sánh thực nghiệm với phương pháp đó khi có code. | Nhật ký kiểm tra novelty theo đúng quy trình ở mục 4.2; ghi riêng hai nhóm: tiền lệ trong affective computing (quyết định đạt/trượt) và tiền lệ ở miền khác (chỉ trích dẫn và so sánh) |
| C5 | **Data sẵn sàng** | **Nới từ 2026-11-21 (user chốt), chấp nhận:** (a) data mở đạt ≥ L2; (b) data gated mà user đã có (MER2023–2026); (c) data gated duyệt tự động hoặc tự phục vụ mà user lấy được ngay (ví dụ HF `gated: auto` như EmotionTalk), sau khi user xác nhận đã lấy; (d) nhãn hay split **dẫn xuất tự động** từ data hợp lệ (pseudo-label của mô hình pretrained, tổ hợp split mới), hoặc một tập đánh giá nhỏ do nhóm user tự gán (tối đa vài trăm mẫu, ≤ 2 ngày công) nếu thật cần. **Vẫn loại:** data cần chữ ký của trường hoặc có điều khoản xem bản thảo (MSP-*), data user không có (IEMOCAP, DFEW, MAFW, Aff-Wild2, K-EmoCon), data không có link tải. | Nhật ký truy cập data đạt ≥ L2 (mục 4.4); với (c) và (d) ghi rõ ai làm và thời gian |

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

### 4.6. Quy trình tìm gap v2 (từ 2026-11-21, thay cách "cơ chế trước")
**Vì sao đổi** (tự đánh giá, 2026-11-21): đợt 2–3 sinh ý tưởng từ danh sách cơ chế ML; 7/8 pilot xoay quanh một bài toán (chéo corpus) sau khi P0 đã bác giả định chung; harness LR trên feature đông cứng không chứng minh được C3; quên tính power; user chỉ thấy kết quả sau nhiều pilot.

**Nguyên tắc**
1. **Gap lấy từ bằng chứng, không từ danh sách cơ chế.** Nguồn xếp theo độ tin cậy:
   - (a) chỗ baseline mạnh 2024–2026 hỏng khi tái lập trên data của mình;
   - (b) limitation, future work, kết quả âm nêu rõ trong bài 2025–2026 ở venue chính, bài tổng kết challenge, survey;
   - (c) phê bình của reviewer trên OpenReview;
   - (d) bảng benchmark có lát cắt thất bại.
   Cơ chế từ miền khác chỉ dùng để **sinh lời giải sau khi** gap đã có bằng chứng.
2. **Rẻ trước, đắt sau.** Mỗi tầng có ngân sách thời gian. Ở tầng 1–3, đọc từng phần là đủ: nội dung ghi là "thu hoạch", chưa phải "đã kiểm tra". Quy tắc full text (§4.1–4.2) và L2/L3 (§4.4) áp dụng đầy đủ từ tầng 4, và cho mọi thứ trình user như ứng viên.
3. **Song song:** luôn giữ 2–3 hướng sống; Kaggle chạy nền trong lúc làm việc bàn giấy.
4. **Claude đóng vai PI** (user yêu cầu): chấm theo rubric công khai, ghi lý do loại; user có quyền phủ quyết.

| Tầng | Việc | Ngân sách | Đầu ra |
|---|---|---|---|
| 0 | Bản đồ tài sản: dataset ở L2/L3, baseline có code chạy được trên Kaggle, compute | 0,5 ngày, cập nhật dần | `ideas/assets.md` |
| 1 | Thu hoạch gap: trích tự động đoạn limitation / future work / conclusion của bài 2025–2026 (MER, MSA, ERC, SER, FER, AVER) đã được nhận ở venue chính, bài tổng kết challenge, survey; ghi nguyên văn kèm nguồn; gom thành cụm | 1 ngày | `literature/gap_mining/gap_bank.md` |
| 2 | Sàng lọc theo rubric (dưới) | 0,5 ngày | top 5 cụm |
| 3 | Novelty nhanh (claim, 3–5 kịch bản scoop, khoảng 1 giờ mỗi cụm) và 2–3 phác thảo phương pháp; viết đóng góp ở dạng mạnh nhất và dạng yếu nhất còn lại | 0,5 ngày | 2–3 hướng |
| 4 | Tín hiệu trên baseline thật: tái lập (≥3 seed) → phương sai và MDE → đo độ lớn thất bại trong lỗi của baseline → bản tối thiểu của phương pháp | 2–3 ngày, song song | go/no-go |
| 5 | Xác minh đầy đủ cho finalist: full text, snowballing, L3, hồ sơ | 1 ngày | `ideas/<slug>.md` |

**Rubric tầng 2** (mỗi mục 0–2; giữ khi tổng ≥ 9/12 và không mục nào bằng 0):
1. Bằng chứng: số nguồn độc lập, độ mới.
2. Đánh giá được bằng tài sản hiện có (data và baseline có code).
3. Chưa bão hòa: ít bài 2025–2026 đã giải.
4. Dẫn tự nhiên tới một phương pháp, không chỉ phân tích.
5. Hiệu ứng dự kiến lớn hơn MDE của benchmark.
6. Vừa 6 trang, kịp hạn, không trùng bài COLING.

**Quy tắc dừng**
- Một thí nghiệm bác giả định chung của một nhánh → loại cả nhánh, không thử biến thể dựa trên cùng giả định. Tối đa 2 pilot mỗi nhánh.
- Trước mỗi pilot phải ước lượng phương sai (seed, bootstrap) và MDE. Nếu MDE lớn hơn hiệu ứng hợp lý thì không chạy, hoặc phải tăng n.
- Pilot chạy trên chính baseline sẽ so sánh (hoặc pipeline của nó). Chỉ dùng harness LR trên feature đông cứng khi giả định là về biểu diễn.

**Nhịp báo cáo:** báo user sau mỗi tầng bằng bảng ngắn; không đợi chạy xong nhiều pilot.

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
| Data gated | **Cập nhật 2026-11-21 (user xác nhận, sửa lại cùng ngày): chỉ có quyền truy cập MER2023, MER2024, MER2025, MER2026 (HF `MERChallenge/*`). KHÔNG có IEMOCAP.** Vẫn **chưa** có MSP-* (MSP-Podcast cần chữ ký của trường; DUA §6 cho bên cấp data 30 ngày xem bản thảo), DFEW, MAFW, Aff-Wild2, K-EmoCon. | Được dùng MER2023–2026 cùng với data mở; **không dùng IEMOCAP**. Phải đưa lên **L3** (đếm mẫu, nhãn, split, đối chiếu với paper) trước khi dùng; L3 chạy trên Kaggle của user vì Claude không giữ token HF. Tuân thủ điều khoản: chỉ nghiên cứu học thuật, không phân phối lại data hay annotation, không đưa mẫu lên mạng (MER). Các ý tưởng phụ thuộc IEMOCAP, MSP-*, DFEW, MAFW, Aff-Wild2, K-EmoCon vẫn bị loại. |
| Bài đang review | **Một bài về Emotion Forecasting đã nộp COLING 2027** (chưa có kết quả) | (a) Dual submission: bài ICME **không được trùng đáng kể** với bài COLING. Hướng thuần forecasting có rủi ro, phải đối chiếu nội dung bài COLING trước khi chọn. (b) Double-blind: không tự trích dẫn kiểu lộ danh tính. |
| Hướng ưu tiên | Không ưu tiên hay tránh hướng nào | User có một survey sơ bộ gồm 19 mục, lưu ở `survey/user_survey_2026-11-19.md`. Đây chỉ là **đầu vào tham khảo**: user mới xem lướt, các số liệu trong đó chưa được kiểm chứng. Phải tự xác minh lại trước khi dùng bất kỳ số liệu nào. |

Còn mở:
- [x] Nội dung bài COLING 2027 (user cung cấp 2026-11-21): **emotion forecasting trong hội thoại multi-turn, dùng dataset Hi-EF**. Hi-EF là arXiv 2407.16406 ("Hi-EF: Benchmarking Emotion Forecasting in Human-interaction"; dự báo cảm xúc trong tương tác hai người); user cho biết bài này được nhận ở AAAI 2026. **Hệ quả:** loại mọi ý tưởng có lõi là dự báo cảm xúc trong hội thoại, và không dùng Hi-EF làm dataset chính. Mục 6 trong survey (dự báo lúc người chấm đổi đánh giá) giữ trạng thái loại vì gần forecasting.

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
| 2026-11-21 | **Sửa diễn giải** (user hỏi quy tắc có quá strict không). Phân tích power post-hoc (`pilot/analyze_intervention_power.py`): Δalign +0,038 chỉ dự đoán Δacc +0,56 theo độ dốc của pilot, trong khi MDE với 8 cặp là 2,40 → thí nghiệm **thiếu power**; NOT SUPPORTED **không** có nghĩa là H sai. Quy tắc giữ nguyên, không nới lỏng. Rút lại câu "tương quan pilot giải thích tốt nhất bằng α" vì không có căn cứ. Thăm dò: `anti` − `shuffled` −2,63 [−3,85; −1,41], gần với dự đoán −2,12; độ dốc trong cặp 15,4 ≈ 14,6 của pilot. Bài học: mọi thí nghiệm đăng ký trước phải có power analysis trước khi chạy. |
| 2026-11-21 | **User loại A và A′**: A′ (kiểm toán protocol CREMA-D + thước đo dựa trên người) không phải đóng góp đủ mạnh. Claude đồng ý: lỗi protocol là vệ sinh thực nghiệm, Q1 chỉ mang tính mô tả, loss phụ unimodal không phải phương pháp mới, và chỉ có 1 dataset. A → **ĐÃ LOẠI**. Giữ lại làm hạ tầng: data CREMA-D ở L3, fold độc lập diễn viên, pipeline Kaggle, protocol đã sửa. Bài học: trước khi chạy pilot, phải đánh giá đóng góp ở dạng mạnh nhất *và* ở dạng yếu nhất còn lại nếu giả thuyết chính thất bại. |
| 2026-11-21 | User chốt loại đóng góp: **phương pháp mới** (ghi vào định nghĩa C3). User cung cấp nội dung bài COLING (forecasting hội thoại multi-turn, Hi-EF) → loại các hướng forecasting hội thoại. Bắt đầu đợt tìm ý tưởng thứ hai. |
| 2026-11-21 | **Đợt tìm ý tưởng thứ 2 kết thúc: không ứng viên nào đạt C1–C5.** Đã sàng lọc khoảng 40 hướng (`ideas/round2_screening_2026-11-21.md`). Ba ứng viên được kiểm tra bằng thí nghiệm CPU với tiêu chí đặt trước (`explore/xling/`) đều trượt C1. **D** (chéo ngôn ngữ, text bị shift nặng nhất): ngược chiều, text chuyển giao tốt nhất. **F4** (expressivity shift): audio/face mất hướng phân biệt; CREMA-D→MELD có g âm, EM-prior làm tệ hơn. **H** (khái niệm cảm xúc làm điểm neo): 0/2 hướng ủng hộ. Data đã xác minh trong đợt: MOSI/CH-SIMS raw (L2), MELD dev/test (L3), CREMA-D media (L3), RAVDESS (L2), MERGE (L2), EmotionTalk (gated auto). Tiền lệ trực tiếp cho thích nghi domain trong MER: Amanda (Findings ACL 2024). Chờ user quyết hướng đi tiếp. |
| 2026-11-21 | User chọn phương án 1 (nới ràng buộc data). Kết quả kiểm tra data gated (`datasets/access_log.md`): **IEMOCAP BỊ CHẶN** (sail.usc.edu trả 503; lỗi chuỗi chứng chỉ TLS; web.archive.org bị chặn) → user cần tự mở trang release. **MSP-Podcast**: Academic DUA miễn phí, phải do người có thẩm quyền của trường ký (thường là sponsored research office); không công bố thời gian duyệt; §6 cho bên cấp data 30 ngày xem bản thảo trước khi nộp (gia hạn được thêm 30 ngày) → rủi ro cao với hạn 30/12; chỉ có audio. **MER2023/2024/2025** trên HF: `gated: auto`, nghĩa là truy cập ngay sau khi điền form (theo tài liệu HF); CC-BY-NC-4.0; MER2024 có video có nhãn 4,05 GB; MER2025 nặng 442 GB. Chờ user xin quyền truy cập trên HF và xác nhận. |
| 2026-11-21 | **User xác nhận đã truy cập được MER2023–2026 và IEMOCAP.** Cập nhật §9. Bước tiếp: (1) L3 cho IEMOCAP và MER2024/2025 bằng notebook Kaggle `datasets/verify/` (user chạy, gửi lại báo cáo); (2) đợt tìm ý tưởng thứ 3 trong không gian mở rộng (IEMOCAP + MELD + MER + các bộ mở đã xác minh), giữ nguyên C1–C5 và yêu cầu phương pháp mới. |
| 2026-11-21 | **User sửa lại: chỉ có MER2023–2026, không có IEMOCAP.** Cập nhật §9. Notebook L3 (`datasets/verify/04_verify_gated.ipynb`) đặt mặc định `RUN_IEMOCAP = False`. Đợt 3 tìm trong phạm vi MER2023–2026 cộng data mở (MELD, CH-SIMS/v2, MOSI, MOSEI mirror, CREMA-D, RAVDESS, MERGE). |
| 2026-11-21 | **User nới C4 và C5.** C4: chỉ tính là scoop nếu ý tưởng cốt lõi đã có trong MER hoặc affective computing; cơ chế đã có ở miền khác vẫn đạt nhưng phải trích dẫn và so sánh. C5: chấp nhận thêm data gated duyệt tự động mà user lấy được ngay, nhãn hay split dẫn xuất tự động, và tập đánh giá nhỏ tự gán (≤ 2 ngày công). Bước tiếp: rà lại các hướng đã loại chỉ vì C4 (miền khác) hoặc C5, rồi tìm tiếp. |
| 2026-11-21 | **User loại MER-Cross; quét cơ chế từ miền khác** (`ideas/round3_mechanism_scan_2026-11-21.md`). Bốn họ cơ chế không dùng nhãn đích đều trượt tiêu chí đặt trước: P2 xóa thông tin corpus bằng LEACE/CORAL (0/3), P3 chọn mô hình bằng Agreement-on-the-Line (0/3; headroom có ở 2/3 hướng), P4 Dawid–Skene trên các head modality (0/3; lỗi tương quan giữa các modality), P5 text dạy head phi ngôn ngữ trên corpus đích (trượt; cận trên khi có nhãn thật +6,1). Model merging đã có trong SER (AdaLTM). Các cơ chế còn lại (group DRO, WiSE-FT, MixStyle, IRM, learning to defer) xếp ưu tiên thấp. **Kết luận: dừng nhánh "sửa shift corpus mà không có nhãn đích".** Chờ user quyết hướng tiếp. |
| 2026-11-21 | User thấy cách tìm gap kém hiệu quả và yêu cầu cải tiến; user không đóng vai PI. Thêm **§4.6 Quy trình tìm gap v2**: gap lấy từ bằng chứng (lỗi khi tái lập baseline, limitation và future work, phê bình reviewer, lát cắt thất bại); 6 tầng từ rẻ đến đắt, có ngân sách thời gian; rubric chấm 12 điểm; quy tắc dừng theo nhánh và theo power; pilot chạy trên baseline thật; báo user sau mỗi tầng. Claude đóng vai PI. |
| 2026-11-21 | **Chạy tầng 0–2 của quy trình v2.**
- Tầng 1: tải full text 170 bài 2025–2026 đã được nhận ở venue chính và trích tự động limitation/future work (`literature/gap_mining/`); đọc conclusion của 15 bài chẩn đoán; đếm độ bão hòa của 18 cụm trên arXiv.
- Tầng 0: `ideas/assets.md`, gồm 19 repo baseline đã kiểm tra bằng `git ls-remote`.
- Tầng 2: chỉ **G8** (xung đột giữa các modality trên dữ liệu thật) đạt rubric (10/12). Lý do: CH-SIMS có nhãn unimodal, và 27,5% clip có cực tính text khác nhãn đa modality. Nhưng split test chỉ có 100 clip xung đột nên thiếu power; cần thêm CH-SIMS v2.
- Review trên OpenReview **BỊ CHẶN** (`/notes?forum=` trả HTTP 403).
- Bước tiếp: tầng 3 cho G8. |
| 2026-11-21 | **Tầng 3 cho G8 → ĐÃ LOẠI (C4)** (`ideas/G8_modality_conflict.md`). Các phần lõi đều đã có trong affective: tập test xung đột từ nhãn unimodal của CH-SIMS v2 (DiffEmo 2406.03004); fusion theo độ tin cậy, đánh giá trên tập con xung đột thật (UMC 2510.20256); học khi nào gộp, khi nào bỏ modality (DCR 2605.04877, MiD-ERC 2611.08860). Phát hiện kèm theo: CH-SIMS ⊂ CH-SIMS v2 (2.281/2.281 clip trùng). Kênh (b) không còn cụm nào. Bước tiếp: kênh (a), tái lập baseline. Đã có sẵn: 6 repo đọc feature dạng MMSA (TLRA, EBMC, SemMSA, MIDAS, LNLN, MMSA) và 4 file feature tải được (MOSI, SIMS, SIMS v2, MOSEI). |
| 2026-11-21 | **L3 cho MER (user chạy notebook):** số mẫu của MER2024 (5.030 nhãn, 5.030 video, test SEMI 1.169, test NOISE 1.170) và MER2025 (7.369 / test 2.026 / 124.802 không nhãn) khớp với bài tổng kết → L3 cho nhãn. MER2026 khớp, trừ test MER-Cross (572 so với 574). **Sự cố:** báo cáo gốc lộ mật khẩu giải nén MER2023 do bộ lọc của Claude sai → không đưa vào repo; đã sửa script, chỉ in heading an toàn, có test hồi quy. Ghi sạch ở `datasets/access_log.md`. |
| 2026-11-21 | **P6 (ý tưởng E, tín hiệu hậu kỳ) → DOES NOT HOLD theo tiêu chí đặt trước → E ĐÃ LOẠI.** CH-SIMS đạt (38,7% clip có nhạc; tag hậu kỳ dự đoán cực tính +9,1 [+4,8; +14,5]). MELD trượt (tỉ lệ xuất hiện 9,6% < 15%). Post-hoc: split của CH-SIMS dùng chung video (59/59); tách theo video thì tín hiệu còn 54,9 bacc, tức khoảng một nửa đến từ rò rỉ theo video. Không nới quy tắc. |
| 2026-11-22 | Lần chạy 2 của notebook L3 (báo cáo sạch). Repo MER2026 **không có data EEG–fNIRS** → ý tưởng F **BỊ CHẶN (C5)**. MER2025 có `audio.zip` 11,29 GB cho toàn kho. Ý tưởng E: Claude đề xuất 1 pilot cuối (E′) trên MER2024, vì P6 không bác giả định chung (CH-SIMS ủng hộ) và quy trình v2 cho phép tối đa 2 pilot mỗi nhánh. Chờ user quyết. |
| 2026-11-22 | **E′ chưa chạy vì không kiểm soát được rò rỉ.** Hiệu chỉnh gom nhóm nguồn trên CH-SIMS (`explore/editorial/group_calibration_chsims_v2.md`): không cách tự động nào (giọng người nói, CLIP, cả hai) đưa rò rỉ xuống dưới 0,95 khi vẫn đạt ràng buộc; chia ngẫu nhiên là 1,00. MER2024 không có ID nguồn → scheme G của E′ tương đương chia ngẫu nhiên. Claude đề xuất đóng nhánh E, hoặc thiết kế lại thành phép thử chéo dataset (MER2024 ↔ CH-SIMS theo cực tính). Chờ user quyết. |
| 2026-11-22 | **User quyết đóng nhánh E.** E → ĐÃ LOẠI, không chạy E′. Code trích feature MER2024 trong `explore/editorial/` (Demucs, AST, WavLM, EasyOCR, CLIP, e5; đã test trên mock) được giữ làm hạ tầng. |
| 2026-11-22 | Đợt 4 (user chọn chỉ nghĩ ý tưởng, chưa dùng GPU). Peak-end: bỏ trước khi chạy pilot, vì clip ngắn. R (dùng lý do cảm xúc để định tuyến fusion): C4, trùng một phần 2508.03722 và RGL. **P8 (neutral là trạng thái vắng cảm xúc): 0/3 → ĐÃ LOẠI**; "hố hút neutral" có thật ở CH-SIMS và MELD, nhưng cách tách hai đầu không sửa được. **Chẩn đoán:** phép sàng lọc trên CPU dùng tập test nhỏ (n = 457–2.329), MDE 2,2–4,4 điểm, nên chỉ bắt được hiệu ứng lớn. |
| 2026-11-22 | **User đồng ý chạy 1 session GPU để trích sẵn feature MER2024.** Notebook 06 (`explore/mer24/06_mer24_features_baseline.ipynb`) dùng các encoder baseline của bài MER2024 (chinese-hubert-large, CLIP-L/14, chinese-roberta-wwm-ext-large), cộng WavLM, e5, tag AST, x-vector người nói và cờ phụ đề. Cache được giữ trên Kaggle. Notebook tái lập baseline (5-fold CV, WAF) so với Bảng 4–5 của bài, đo MDE và các lát cắt lỗi. Sau đó, mọi phép sàng lọc chạy bằng notebook 07 trên CPU của Kaggle. Đã test cả hai bước trên mock. |
| 2026-11-23 | **User chốt: dùng split chính thức mà cộng đồng đang dùng** (CH-SIMS và v2: split chính thức; MER2024: 5-fold Train&Val; M3ED: split chính thức tách theo phim truyền hình). Không tự đặt thêm ràng buộc tách theo video. Tìm thêm dataset phim truyền hình cho E2: **M3ED lên L3** (24.449 câu, 56 phim, split tách theo phim, có audio, không có video); SpEmoC gated (EULA thủ công). |
| 2026-11-23 | **P9 (E2, hoán đổi nhạc nền) → DOES NOT HOLD → nhánh E đóng hẳn.** CH-SIMS: Δflip = +1,9 [−0,7; +4,9] (MDE 3,9), dưới ngưỡng +5. User yêu cầu dừng phần M3ED vì kết luận không đổi. Peak-end: P10 (đăng ký trước, CH-SIMS + MOSI) cộng P10b (cổng phương pháp: peak-end phải thắng attention pooling học được ≥ +0,02 r trên cả hai dataset; thêm sau phản biện của user, trước khi có kết quả). Kiểm tra data: MOSEI L2 (mirror); OMG-Emotion nhãn L2, video chỉ có link YouTube (11/20 link còn sống); MovieGraphs và EmoVid duyệt thủ công → loại; LIRIS-ACCEDE không truy cập được. |
| 2026-11-23 | **User đóng nhánh peak-end** trước khi P10 có kết quả. Lý do: attention pooling học được bao trùm peak-end về biểu diễn, nên quy tắc cố định khó đạt C3; clip ngắn (CH-SIMS median 3,21 s). Đã dừng P10 và P10b; không có kết quả. Còn mở: chờ report e06 (MER2024 baseline, MDE, lát cắt lỗi) để tìm gap theo kênh (a). |
| 2026-11-23 | **e06 (MER2024 baseline, notebook 06, commit 09caf41) xong**: 5.029/5.030 clip có feature. Tái lập trong ngưỡng sanity 5 điểm: A_hub 72,68 (bài 73,02), V_clip 64,62 (66,66), T_rob 50,02 (52,66), AVT 78,58 (79,31–80,34) → pipeline dùng được. MDE so sánh cặp ≈ 1,32 WAF. Lát cắt: AVT − AV = −0,04 [−0,97; +0,90] (text không đóng góp khi fusion bằng LR); surprise recall 50,5, worried 59,7; oracle 'ít nhất một modality đúng' 90,3% so với AVT 78,7%. Lỗi: cột valence có NaN làm hỏng lát cắt theo |valence| (đã sửa screen). Kaggle API key của user lưu làm network secret cho www.kaggle.com. Report: `explore/mer24/results/e06_report_2026-11-23.md`. |
| 2026-11-23 | **User đồng ý tải cache feature MER2024 về container** (21 file `part_*.npz`, `text.npz`, 2 CSV nhãn; 177 MB) qua Kaggle API. Chỉ lưu ở scratchpad của phiên, **không bao giờ commit hay chia sẻ**; output trong repo vẫn chỉ là thống kê tổng hợp. Kiểm tra: 5.030 clip, 5.029 có feature; valence NaN ở 834 clip (khớp n = 834 của Q1 bị lỗi trong e06). Container bị xóa khi phiên kết thúc → phiên sau phải tải lại. |
