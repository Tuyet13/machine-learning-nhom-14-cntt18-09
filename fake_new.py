import re
import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

sys.stdout.reconfigure(encoding='utf-8')


def remove_source_leakage(text: str) -> str:
    """Xóa phần tiêu đề nguồn tin (dateline) như 'WASHINGTON (Reuters) - ' ở đầu bài."""
    if not isinstance(text, str):
        return ""
    # 1. Xóa cụm địa danh + (Reuters) - ở đầu bài (giới hạn 60 ký tự đầu để tránh xóa nhầm đoạn văn)
    text = re.sub(r'^.{0,60}?\(Reuters\)\s*[-–—]\s*', '', text, flags=re.IGNORECASE)
    # 2. Xóa tên hãng tin Reuters còn sót lại trong nội dung
    text = re.sub(r'\breuters\b', '', text, flags=re.IGNORECASE)
    return text


def clean_text(text: str) -> str:
    """Chuẩn hóa và làm sạch văn bản."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Xóa URL
    text = re.sub(r'[^\w\s]', ' ', text)                 # Đổi dấu câu thành dấu cách tránh dính từ
    text = re.sub(r'\s+', ' ', text).strip()             # Gộp khoảng trắng thừa
    return text


# 1. Tải, gán nhãn và xử lý rò rỉ dữ liệu
print("1. Đang tải và chuẩn bị dữ liệu...")
df_true = pd.read_csv('True.csv').assign(label=1)
df_fake = pd.read_csv('Fake.csv').assign(label=0)
df = pd.concat([df_true, df_fake], ignore_index=True)

# Xóa nguồn báo trên cột 'text' TRƯỚC khi ghép với 'title'
df['text'] = df['text'].fillna('').apply(remove_source_leakage)
df['content'] = df['title'].fillna('') + ' ' + df['text']

# Lọc các bài bị trùng lặp nội dung để tránh vừa xuất hiện ở Train vừa ở Test
before_dedup = len(df)
df = df.drop_duplicates(subset=['content']).reset_index(drop=True)
print(f"Tổng số mẫu sau khi lọc trùng: {len(df)} (đã loại {before_dedup - len(df)} mẫu)")

# 2. Tiền xử lý văn bản
print("2. Đang tiền xử lý văn bản...")
df['clean_content'] = df['content'].apply(clean_text)

# 3. Chia tập Train/Test (train_test_split mặc định đã tự xáo trộn dữ liệu)
print("3. Chia tập Train/Test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_content'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label'],
)

# 4. Vector hóa văn bản bằng TF-IDF
print("4. Vector hóa văn bản bằng TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Huấn luyện, đánh giá và vẽ Confusion Matrix bằng vòng lặp
models = {
    'Naive Bayes': (MultinomialNB(), plt.cm.Blues),
    'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000), plt.cm.Greens),
}

fig, axes = plt.subplots(1, len(models), figsize=(14, 6))

for ax, (name, (model, cmap)) in zip(axes, models.items()):
    print(f"\n--- Kết quả {name} ---")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    print(classification_report(y_test, y_pred, target_names=['Fake (0)', 'True (1)']))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake (0)', 'True (1)'])
    disp.plot(ax=ax, cmap=cmap, values_format='d')
    ax.set_title(f'Confusion Matrix - {name}')

plt.tight_layout()
plt.show()
print("Hoàn tất!")