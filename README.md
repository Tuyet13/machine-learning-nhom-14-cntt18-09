# Machine Learning - Nhóm 14 CNTT18-09

Dự án phân loại tin tức thành **tin giả (Fake)** hoặc **tin thật (True)** bằng các mô hình học máy trên dữ liệu văn bản.

## Tệp trong dự án

- `fake_new.py`: tiền xử lý dữ liệu, huấn luyện mô hình và đánh giá kết quả.
- `Fake.csv`: dữ liệu tin giả.
- `True.csv`: dữ liệu tin thật.

Hai tệp CSV cần có các cột `title` và `text`.

## Mô hình và quy trình

Chương trình kết hợp tiêu đề và nội dung bài viết, chuẩn hóa văn bản, rồi chia dữ liệu thành tập huấn luyện và kiểm tra theo tỷ lệ 80/20. Văn bản được chuyển thành đặc trưng TF-IDF (tối đa 5.000 đặc trưng, unigram và bigram). Chương trình huấn luyện và so sánh:

- Multinomial Naive Bayes
- Logistic Regression

Kết quả gồm báo cáo phân loại trong terminal và biểu đồ ma trận nhầm lẫn.

## Cài đặt và chạy

Cần cài Python cùng các thư viện sau:

```bash
python -m pip install pandas numpy matplotlib scikit-learn
```

Đặt `fake_new.py`, `Fake.csv` và `True.csv` trong cùng thư mục, sau đó chạy từ thư mục đó:

```bash
python fake_new.py
```
