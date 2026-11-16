import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

print("1. Đang tải và gán nhãn dữ liệu...")
df_true = pd.read_csv('True.csv')
df_fake = pd.read_csv('Fake.csv')

# Gán nhãn: 1 cho tin thật, 0 cho tin giả
df_true['label'] = 1
df_fake['label'] = 0

# Gộp và xáo trộn dữ liệu
df = pd.concat([df_true, df_fake], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Kết hợp tiêu đề và nội dung chính
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

print(f"Tổng số lượng mẫu: {len(df)}")

print("2. Tiền xử lý văn bản...")


def clean_text(text):
  text = text.lower()  # Chuyển chữ thường
  text = re.sub(
      r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE
  )  # Xóa URL
  text = re.sub(r'[^\w\s]', '', text)  # Xóa dấu câu
  text = re.sub(r'\s+', ' ', text).strip()  # Xóa khoảng trắng thừa
  return text


df['clean_content'] = df['content'].apply(clean_text)

print("3. Chia tập train/test (tránh Data Leakage)...")
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_content'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

print("4. Vector hóa văn bản bằng TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("5. Huấn luyện và đánh giá Baseline Model (Multinomial Naive Bayes)...")
nb_model = MultinomialNB() if 'MultinomialNB' in globals() else None
# (Đảm bảo đã import MultinomialNB ở trên nếu bị thiếu, dưới đây mình viết gọn lại)
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)

print('--- Kết quả Naive Bayes ---')
print(classification_report(y_test, y_pred_nb, target_names=['Fake (0)', 'True (1)']))

print("6. Huấn luyện và đánh giá Mô hình so sánh (Logistic Regression)...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)

print('--- Kết quả Logistic Regression ---')
print(classification_report(y_test, y_pred_lr, target_names=['Fake (0)', 'True (1)']))

print('7. Đang vẽ biểu đồ Confusion Matrix...')
# Tạo khung chứa 2 biểu đồ cạnh nhau
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Biểu đồ cho Naive Bayes
disp_nb = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred_nb),
    display_labels=['Fake (0)', 'True (1)'],
)
disp_nb.plot(ax=axes[0], cmap=plt.cm.Blues, values_format='d')
axes[0].set_title('Confusion Matrix - Naive Bayes')

# Biểu đồ cho Logistic Regression
disp_lr = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred_lr),
    display_labels=['Fake (0)', 'True (1)'],
)
disp_lr.plot(ax=axes[1], cmap=plt.cm.Greens, values_format='d')
axes[1].set_title('Confusion Matrix - Logistic Regression')

plt.tight_layout()
plt.show()
print('Hoàn tất! Biểu đồ ma trận nhầm lẫn đã được hiển thị.')