# Ứng viên C: VEATIC, hai cách hiểu bền vững hay chỉ là lệch nhịp giữa người chấm

Mục 5 trong survey của user.
**Trạng thái (2026-11-19):** **CHƯA ĐẠT, TẠM DỪNG.** C5 đã xác minh (L3). **C1 yếu**: hiện tượng cốt lõi hiếm trong dữ liệu (xem bên dưới). C2 và C4 chưa kiểm tra xong.

## Dữ kiện đã xác minh
- [SỰ THẬT] VEATIC_Psych: 83 video, 61–74 người chấm mỗi video, rating liên tục theo frame; nhận diện được cùng một người qua nhiều video. Video gốc tải được từ `VEATIC.zip` (3,2 GB).
- [SỰ THẬT] **Pilot độ phổ biến của phân phối hai đỉnh:**
  - Dữ liệu: valence, lấy mẫu 1/10 frame, tổng 15.200 frame.
  - Tiêu chí: Sarle bimodality coefficient > 0,555.
  - Kết quả: chỉ **7,2% số frame** đạt tiêu chí; chỉ **6/83 video** có hơn 20% số frame đạt.
- [SỰ THẬT] VEATIC có 25 bài trích dẫn (Semantic Scholar). Các bài gần nhất:
  - Nature Communications 2025 về tích hợp tín hiệu ngữ cảnh và nhân vật theo từng cá nhân, bằng mô hình Bayes. Không dùng VEATIC, không mô hình hóa phân phối hai đỉnh hay độ trễ. Đọc qua PMC.
  - Chehre (arXiv 2606.21657): nhận diện biểu cảm dạng phân phối, nhãn tĩnh theo video, khoảng 30 người chấm mỗi video.

## Đánh giá chuyên môn
- [ĐÁNH GIÁ] Hiện tượng "hai cách hiểu đồng thời" chỉ chiếm thiểu số dữ liệu. Câu chuyện chính có nguy cơ hẹp. Sarle BC là heuristic thô, nhạy với độ lệch (skew); cần thêm phép thử mixture có hiệu chỉnh trễ trước khi kết luận chắc.
- [ĐÁNH GIÁ] Mảng dự đoán phân phối hoặc độ bất định cho cảm xúc liên tục đã có các công trình trên RECOLA. Chưa đọc các bài này, nên **C4 chưa được kiểm tra**. Đây là bước bắt buộc nếu mở lại ứng viên.
- Hướng phụ có thể đáng giá hơn: mô hình hóa người chấm cá nhân, tức dự đoán vệt rating của một PID từ lịch sử của chính họ trên các video khác. Cần kiểm tra trùng lặp với bài COLING forecasting của user trước.
