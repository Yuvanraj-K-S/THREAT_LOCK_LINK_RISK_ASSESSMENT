import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

# === Load dataset ===
df = pd.read_csv(r"E:\Threat_lock_link_risk_assessment\data\dataset1.csv")
urls = df['url'].astype(str).values
labels = df['type'].values  # 0 = good, 1 = malicious

# === Check class balance ===
unique, counts = np.unique(labels, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# === Tokenization ===
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(urls)
sequences = tokenizer.texts_to_sequences(urls)

# === Pad sequences ===
max_len = 200
X = pad_sequences(sequences, maxlen=max_len)
y = labels

# === Save tokenizer ===
with open("outputs/url_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# === Train/test split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# === Handle imbalance ===
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# === Define CNN model ===
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_len),
    Conv1D(128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# === Train model ===
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[early_stop]
)

# === Evaluate with tuned threshold ===
val_probs = model.predict(X_val)
threshold = 0.15  # Tuned threshold for better recall
y_pred = (val_probs > threshold).astype("int32")

print("\n✅ Evaluation Results (Threshold = 0.15):")
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred, digits=4))

# === Plot Precision-Recall vs Threshold ===
precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall vs Threshold')
plt.legend()
plt.grid(True)
plt.show()

# === Evaluate on custom malicious test cases ===
custom_urls = ["maliciouswebsitetest.com", "testsafebrowsing.appspot.com", "youtube.com", "accounts.youtube.com"]
custom_seq = tokenizer.texts_to_sequences(custom_urls)
custom_pad = pad_sequences(custom_seq, maxlen=max_len)
custom_probs = model.predict(custom_pad)

print("\n✅ Custom URL Predictions:")
for url, prob in zip(custom_urls, custom_probs):
    is_malicious = prob[0] > threshold
    print(f"{url:40}  |  Probability: {prob[0]:.4f}  |  Malicious: {is_malicious}")

# === Save model ===
model.save("outputs/deep_url_classifier.h5")
print("\n✅ Model and tokenizer saved!")