# Amazon Reviews Sentiment Analysis

### โครงงานโดย
- **นิธิโรจน์ รัตนเรืองมาศ (6610505446)**
- **สิโรฒม์ วีระกุล (6610502234)**

---

## ความน่าสนใจของหัวข้อ

หัวข้อนี้เกี่ยวข้องกับโลกธุรกิจจริงและสามารถนำผลลัพธ์ไปต่อยอดในการตัดสินใจทางธุรกิจได้ เช่น  
- แสดงสินค้าที่ได้รับคำตอบรับที่ดีให้กับผู้ใช้ในแอปหรือเว็บไซต์  
- วิเคราะห์ความคิดเห็นลูกค้าเพื่อปรับปรุงสินค้าและบริการ  

นอกจากนี้ยังเกี่ยวข้องอย่างมากกับ **Natural Language Processing (NLP)**  
ซึ่งเป็นเทคโนโลยีที่ทำให้คอมพิวเตอร์เข้าใจและประมวลผลภาษามนุษย์ได้  
โดยเฉพาะในยุคปัจจุบันที่มีการนำ **Large Language Models (LLMs)** มาใช้อย่างแพร่หลาย

---

## ทำไมต้องใช้ Deep Learning

เพราะ Sentiment Analysis เป็นการวิเคราะห์ "บริบท" ของประโยครีวิวสินค้า ซึ่งมีความซับซ้อนเกินกว่าการวิเคราะห์เชิงสถิติทั่วไป  
รีวิวจากลูกค้ามักมีลักษณะภาษาไม่ตรงไปตรงมา เช่น การใช้คำประชดหรือคำที่มีความหมายเปลี่ยนตามบริบท  

**Deep Learning** สามารถเรียนรู้ “representation” ของคำได้โดยอัตโนมัติผ่าน **Word Embedding**  
ต่างจากการสร้าง feature แบบเดิมที่ต้องนับคำบวก/ลบเอง

---

## Model Architecture

### Stacked Bidirectional LSTM (2 Layers)
ช่วยให้โมเดลเข้าใจบริบทของประโยคได้ทั้งจากอดีตและอนาคต

| Layer | Description | Parameters |
|--------|--------------|-------------|
| **Embedding** | แปลง token → vector | `VOCAB_SIZE * EMBEDDING_DIM = 20,000 * 256 = 5,120,000` |
| **Bidirectional LSTM (Layer 1)** | Forward + Backward Context | Hidden Dim = 512 |
| **Bidirectional LSTM (Layer 2)** | รับ input จาก Layer 1 | Hidden Dim = 512 |
| **Dense (Fully Connected)** | Output 1 node (sigmoid) | Weights = 1024, Bias = 1 |

**Hyperparameters**
```python
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
DROPOUT = 0.4
BATCH_SIZE = 64
EPOCHS = 10
MAX_SEQUENCE_LENGTH = 250
MAX_TOKENS = 20000
```

---

## Data Preparation & Preprocessing

### 1️Upsampling Minority Class
เนื่องจาก dataset มีการกระจาย label ไม่สมดุล (imbalanced dataset)

```python
pos_df = df[df["label"] == 1]
neg_df = df[df["label"] == 0]

neg_df_upsampled = resample(
    neg_df,
    replace=True,
    n_samples=len(pos_df),
    random_state=42
)

df_balanced = pd.concat([pos_df, neg_df_upsampled]).sample(frac=1, random_state=42)
```

### Tokenization & Encoding
ข้อมูลรีวิวเป็นข้อความ ต้องแปลงเป็นตัวเลขก่อนนำเข้าโมเดล

```python
vectorize_layer = layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH
)
vectorize_layer.adapt(train_texts)
VOCAB_SIZE = vectorize_layer.vocabulary_size()
```

---

## Model Implementation

```python
embedding_layer = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)

model = keras.Sequential([
    layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int64"),
    embedding_layer,
    layers.Dropout(DROPOUT),

    layers.Bidirectional(layers.LSTM(HIDDEN_DIM, return_sequences=True)),
    layers.Dropout(DROPOUT),

    layers.Bidirectional(layers.LSTM(HIDDEN_DIM, return_sequences=False)),
    layers.Dropout(DROPOUT),

    layers.Dense(1, activation='sigmoid')
])
model.summary()
```

โมเดลใช้ **Dropout Layers** เพื่อป้องกัน Overfitting  
และ **Dense Layer** สุดท้ายในการทำนายผลแบบ Binary Classification (Positive / Negative)

---

## Training Process

```python
model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=5.0
    ),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds
)
```

- **Loss Function:** `binary_crossentropy` — เหมาะกับ binary classification  
- **Optimizer:** `Adam` (learning_rate=0.001, clipnorm=5.0 เพื่อป้องกัน exploding gradient)  
- **Epochs:** 10 รอบ  
- **Validation:** ใช้ชุด test_ds เพื่อตรวจสอบ overfitting  

---

## Evaluation

### ผลลัพธ์การประเมินโมเดล
```
Accuracy: 0.9924
Loss: 0.0391
Test Accuracy (Balanced via Upsampling): 0.9933
```

### Confusion Matrix

|               | Predict Positive | Predict Negative |
|----------------|------------------|------------------|
| **Actual Positive** | 878 | 0 |
| **Actual Negative** | 12 | 890 |

### Classification Report

| Class | Precision | Recall | F1-Score |
|--------|------------|---------|-----------|
| Negative | 0.9865 | 1.0000 | 0.9932 |
| Positive | 1.0000 | 0.9867 | 0.9933 |

---

## Example Predictions

```python
predict_sentiment("I really love this product!")  # Positive
predict_sentiment("Terrible quality, very disappointed.")  # Negative
```

ตัวอย่างผลลัพธ์การทำนาย (Unseen Data):

**Positive Sentiment Predictions**
- "Excellent product, would buy again." → Positive  
- "Very happy with this, five stars!" → Positive  

**Negative Sentiment Predictions**
- "Do not buy this, completely useless." → Negative  
- "Cheap material, doesn't work at all." → Negative  

---

## Dataset

ข้อมูลรีวิวสินค้าจาก **Amazon.com**  
(สามารถหาได้จาก [Kaggle Datasets](https://www.kaggle.com))

### โครงสร้างข้อมูลก่อนประมวลผล
| Column | Description |
|---------|--------------|
| `reviewText` | ข้อความรีวิว เช่น `"This product is great!"` |
| `overall` | คะแนนรีวิว 1–5 ดาว |

### หลังประมวลผล
| Column | Description |
|---------|--------------|
| `reviewText` | ข้อความรีวิว |
| `overall` | คะแนนรีวิว 1–5 |
| `label` | 0 = Negative (1–2 ดาว), 1 = Positive (4–5 ดาว) |

---

## อ้างอิง

- Cui, Ke & Wang (2018). *Deep Bidirectional and Unidirectional LSTM Recurrent Neural Network for Network-wide Traffic Speed Prediction.* [arXiv:1801.02143](https://arxiv.org/abs/1801.02143)  
- [Bidirectional LSTM in NLP - GeeksforGeeks](https://www.geeksforgeeks.org/bidirectional-lstm/)
- [Word Embeddings in NLP - GeeksforGeeks](https://www.geeksforgeeks.org/word-embedding-in-nlp/)

---

## การแบ่งงาน

| ผู้จัดทำ | หน้าที่ |
|-----------|----------|
| **นิธิโรจน์ รัตนเรืองมาศ (6610505446)** | Data Preparation, Preprocessing, Upsampling, TextVectorization, Data Pipeline, New Sentence Testing |
| **สิโรฒม์ วีระกุล** | Model Architecture Design (Embedding, Bi-LSTM, Dropout, Dense), Model Training, Evaluation (classification_report, confusion_matrix), Model Explanation |

---

## สรุป

โมเดล **Stacked Bidirectional LSTM** ที่พัฒนาขึ้นสามารถวิเคราะห์รีวิวจาก Amazon ได้อย่างมีประสิทธิภาพ  
มีความแม่นยำสูงถึง **99%** หลังการปรับสมดุลข้อมูล (Upsampling)  
และสามารถเข้าใจบริบทของภาษาทั้งจากอดีตและอนาคตได้ดีเยี่ยม เหมาะสำหรับงานด้าน **Sentiment Analysis** บนข้อความจริงจากผู้ใช้งาน
