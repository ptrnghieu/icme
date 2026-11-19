# Phân loại 19 mục trong survey của user (`survey/user_survey_2026-11-19.md`)

Ngày: 2026-11-19. Ràng buộc áp dụng (CLAUDE.md §9): chỉ data mở; 2×T4; tránh trùng với bài forecasting đang nộp COLING 2027; hạn ICME là 30/12/2026.
Ký hiệu cột "Căn cứ": **[SỰ THẬT]** = đã kiểm chứng (xem `datasets/access_log.md`); **[ĐÁNH GIÁ]** = nhận định chuyên môn kèm lý do.

| # | Mục | Kết luận | Căn cứ |
|---|---|---|---|
| 1 | AAC / ReCANVo: hỏi người thân lúc nào | ĐÃ LOẠI | [ĐÁNH GIÁ] Số sự kiện "khám phá nghĩa mới" rất ít theo chính số liệu của survey (10–11 cặp người–nhãn), nên không đủ lực thống kê. Chỉ có một dataset, nên trượt C2. Bài toán hỏi người chăm sóc thuộc HCI/AAC, khó khớp với topic chính của ICME. [SỰ THẬT] Data mở (Zenodo, CC-BY-3.0-US). |
| 2 | Phán đoán V và AV theo Blackwell | ĐÃ LOẠI (giữ lại tài nguyên) | [ĐÁNH GIÁ] Đây là phân tích tri giác của người, không phải phương pháp đa phương tiện; hiệu ứng chính rất nhỏ (−0,9 đến −1,0 điểm %). [SỰ THẬT] Phiếu chấm theo modality của CREMA-D tải được và parse được. Tài nguyên này được dùng lại cho ứng viên A (`ideas/A_human_modality_reliability.md`). |
| 3 | Định lý số hành động n+1 | ĐÃ LOẠI | [ĐÁNH GIÁ] Thuần lý thuyết quyết định, không có thực nghiệm đa phương tiện, nên trượt C3 (không khớp ICME). |
| 4 | ASR: hướng lỗi thay từ theo ngữ điệu | ĐÃ LOẠI | [ĐÁNH GIÁ] TESS chỉ có 2 người nói, nên trượt C2. Survey đã tự ghi nhận nhiều đối thủ gần và đã hạ ưu tiên. |
| 5 | VEATIC: đa nghĩa đồng thời hay lệch nhịp | **GIỮ để kiểm tra sâu** | [SỰ THẬT] VEATIC_Psych tải được (L3): 83 video, 61–74 người chấm mỗi video, rating theo frame. Video tải được (L2). Xem `ideas/C_veatic_rater_structure.md`. |
| 6 | Dự báo lúc người chấm đổi đánh giá | TẠM HOÃN | [ĐÁNH GIÁ] Bản chất là forecasting, nên rủi ro trùng với bài COLING 2027 của user. Cần đọc nội dung bài COLING trước khi xét tiếp. |
| 7 | CHIARO: cùng valence, khác cảm xúc | ĐÃ LOẠI | [ĐÁNH GIÁ] Chỉ có text; phải tự xây cặp dữ liệu cùng dấu khác cảm xúc, nên trượt C5 trong 41 ngày và khó khớp ICME. |
| 8 | Lỗi hoán vai (MEmoR) | ĐÃ LOẠI | [SỰ THẬT] MEmoR phải gửi License Agreement qua email để nhận mật khẩu, tức là data gated. User chưa có quyền, nên trượt C5. |
| 9 | Dự báo đối chiếu đa nhân vật | ĐÃ LOẠI | [SỰ THẬT] Phụ thuộc MEmoR (gated). [ĐÁNH GIÁ] Là forecasting, nên rủi ro trùng COLING; survey cũng ghi chưa có data phù hợp. |
| 10 | Mult²EMo: độc giả và tác giả | ĐÃ LOẠI | [SỰ THẬT] Trang phát hành ghi "available upon request", tức là gated. |
| 11 | MECES: hiểu sai lý do | ĐÃ LOẠI | [SỰ THẬT] Video, audio và feature phải ký agreement và gửi email. [ĐÁNH GIÁ] Phần text còn lại không phải bài toán đa phương thức. |
| 12 | DAR | ĐÃ LOẠI | [ĐÁNH GIÁ] Là tác vụ đã công bố; survey không nêu được khoảng trống riêng, nên trượt C1. |
| 13 | OSMO | ĐÃ LOẠI | [SỰ THẬT] Trang dự án không có link tải dataset. [ĐÁNH GIÁ] Không có RQ riêng, nên trượt C1. |
| 14 | ParaS2S | ĐÃ LOẠI | [ĐÁNH GIÁ] Cần nhãn người mới, tách nội dung khỏi cách nói, nên trượt C5. Mô hình speech-to-speech nặng so với T4. |
| 15 | MER-Bench | ĐÃ LOẠI | [ĐÁNH GIÁ] Cần nhãn người mới về độ trung thành với sự kiện, nên trượt C5. |
| 16 | EEmo-Logic | ĐÃ LOẠI | [ĐÁNH GIÁ] Survey tự kết luận chuyển top-k từ ảnh sang video chưa đủ khác biệt, nên trượt C1/C4. |
| 17 | RESCUE-Bench | ĐÃ LOẠI | [ĐÁNH GIÁ] Benchmark đã bao trùm các ý mở rộng. Survey ghi raw AV không được tái phân phối. |
| 18 | AFFEC: felt và perceived | **GIỮ để kiểm tra sâu** | [SỰ THẬT] Data tải và parse được (L3). Xem `ideas/B_affec_felt_vs_perceived.md`. |
| 19 | ETC | ĐÃ LOẠI | [ĐÁNH GIÁ] Chỉ có text tiếng Nhật; phải gán thêm nhãn cảm xúc được kể, nên trượt C5 và khó khớp ICME. |

Ngoài survey, mình bổ sung **ứng viên A** (`ideas/A_human_modality_reliability.md`), dùng phiếu chấm theo modality của CREMA-D cho bài toán modality imbalance.
