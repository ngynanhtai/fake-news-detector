# 🛡️ Kiểm Tra Tin Tức — Hybrid ML + LLM

## Kiến trúc hệ thống

```
Tin nhắn người dùng
        │
        ▼
  Ẩn thông tin cá nhân (SĐT, TKNH) [llm_engine.py]
        │
        ▼
  ┌───────────────────────────────────────────────┐
  │   ML Model Layer (TF-IDF + Logistic Reg)      │ ← Nhanh, chạy local, chi phí thấp [ml_model.py]
  └──────────────────────┬────────────────────────┘
                         │
              Confidence ≥ 85%? [ml_model.py]
             /                        \
          Yes                          No
          /                              \
  Hiển thị phân tích sơ bộ        Chuyển giao dữ liệu sang [app.py]
  bởi "Hybrid ML + LLM" [app.py]  LLM (Ollama gemma:7b) để phân tích sâu [llm_engine.py]
          \                              /
           ▼                            ▼
         Hiển thị kết quả & Thu nhận feedback Cô/Chú (👍 / 👎) [app.py]
                                │
                    Ghi nhận nhãn điều chỉnh [app.py]
                                │
                 Lưu vào feedback_samples.csv [ml_model.py]
                                │
  Gộp Dataset gốc + Dataset Feedback ──► Retrain hệ thống máy học liên tục [ml_model.py]
```

## Cài đặt

```bash
pip install streamlit pandas scikit-learn joblib requests feedparser
```

Cần Ollama cho tính năng LLM (phân tích sâu):
```bash
# Cài Ollama: https://ollama.ai
ollama pull gemma:7b
ollama serve   # để nền
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

## Thu thập dữ liệu báo chí bổ sung và tự động retrain (Optional):
```bash
python data_collector.py --max 30 --retrain
```

## Cấu trúc file

```
fake_news_detector/
├── app.py                  ← Giao diện Streamlit, quản lý trạng thái tương tác và nhận feedback Cô/Chú
├── ml_model.py             ← Lớp xử lý Học máy (TF-IDF + LR), quản lý nạp dữ liệu gộp, đánh giá & huấn luyện lại
├── llm_engine.py           ← Lớp xử lý LLM (Ollama), lọc thông tin nhạy cảm và định dạng kết quả Markdown
├── data_collector.py       ← Module kết nối RSS cào dữ liệu bài viết chính thống từ các cơ quan báo chí lớn
├── audit_logger.py         ← Ghi log chi tiết lịch sử tương tác và luồng dữ liệu phục vụ giám sát
├── du_lieu_tin_tuc.csv     ← Tập dữ liệu tin tức cơ sở (Tự động khởi tạo dữ liệu mẫu nếu chưa tồn tại)
├── feedback_samples.csv    ← Tập dữ liệu thu thập trực tiếp từ các đóng góp/sửa lỗi của người dùng
├── model_metrics.json      ← Tệp lưu trữ các chỉ số phân tích hiệu năng mô hình (Accuracy, Precision, Recall, F1)
├── mo_hinh_ai_tin_gia.pkl  ← Pipeline mô hình học máy đã được đóng gói sau huấn luyện
└── audit_log.csv           ← Nhật ký ghi vết lịch sử kiểm tra của hệ thống
```

## Điều chỉnh ngưỡng confidence

Trong `ml_model.py`, thay đổi `CONFIDENCE_THRESHOLD`:
- `0.85` (mặc định): ML chịu trách nhiệm xử lý các tin tức mang tính rõ ràng, LLM xử lý phân tích các tin tức mơ hồ
- `0.95`: Thắt chặt độ an toàn, hệ thống sẽ ưu tiên đẩy nhiều tác vụ phân tích chi tiết sang cho LLM đảm nhiệm.
- `0.70`: Tối ưu hóa tốc độ xử lý phần cứng tại chỗ, ưu tiên sử dụng năng lực phân loại của mô hình học máy local.

## Vòng lặp học liên tục

Mỗi lần người dùng bấm 👍/👎:
1. Hệ thống tính nhãn đúng (đảo ngược nếu 👎)
2. Lưu mẫu mới vào `du_lieu_tin_tuc.csv`
3. Tiến hành kích hoạt tiến trình Huấn luyện lại (Retrain) toàn bộ mô hình để cải thiện các phân tách kế tiếp mà không làm gián đoạn hệ thống

## Các Tính Năng Nổi Bật
- **Data Masking (Privacy):** Tự động ẩn số điện thoại, số tài khoản ngân hàng bằng Regex trước khi đưa vào AI phân tích.
- **Feedback & Logging (AI Audit):** Các tương tác và phản hồi Đúng/Sai của người dùng được lưu trữ an toàn tại `audit_log.csv` để kỹ sư phát triển cải thiện hệ thống.
