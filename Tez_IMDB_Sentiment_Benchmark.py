# ==================================================================================================
# TEZ BAŞLIĞI: IMDB FİLM YORUMLARI DUYGU ANALİZİNDE 4 KUŞAK KIYASLAMASI
# KATEGORİLER: ML -> DL -> STANDARD TRANSFORMER -> ADVANCED TRANSFORMER
# GÜNCELLENMİŞ VERSİYON: 18 MODEL
# ORTAM: Google Colab Pro+ (GPU)
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# 1. KURULUM VE KÜTÜPHANELER
# --------------------------------------------------------------------------------------------------
!pip install sentencepiece --quiet

import os
import gc
import time
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shutil

# Makine Öğrenmesi
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Derin Öğrenme (Keras/TensorFlow)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, Bidirectional, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

# Transformerlar (Hugging Face & PyTorch)
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# AdamW optimizasyonu
from torch.optim import AdamW

# Google Colab Dosya İndirme
from google.colab import files

# Uyarıları kapat
import warnings
warnings.filterwarnings('ignore')

# Grafik Ayarları
sns.set(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.figsize'] = [10, 6]

print("✅ Tüm kütüphaneler başarıyla yüklendi.")
print(f"GPU Durumu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU BULUNAMADI!'}")

# ==================================================================================================
# 2. VERİ HAZIRLIĞI VE YARDIMCI SINIF
# ==================================================================================================

class TezYonetimi:
    def __init__(self):
        self.results_df = pd.DataFrame(columns=["Kategori", "Model", "Doğruluk", "Kesinlik", "Duyarlılık", "F1-Skoru", "Eğitim Süresi (sn)"])
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.raw_text_train = None
        self.raw_text_test = None

    def veri_yukle_ve_hazirla(self):
        print("\n📥 VERİ SETİ İNDİRİLİYOR VE HAZIRLANIYOR (IMDB)...")
        try:
            from datasets import load_dataset
            dataset = load_dataset("imdb")

            # Pandas'a çevir
            train_df = pd.DataFrame(dataset['train'])
            test_df = pd.DataFrame(dataset['test'])

            # Veriyi Birleştirip Karıştıralım
            df = pd.concat([train_df, test_df]).sample(frac=1, random_state=42).reset_index(drop=True)

            # Temizlik Fonksiyonu
            def clean_text(text):
                text = re.sub(r'<.*?>', '', text) # HTML taglerini sil
                text = re.sub(r'[^a-zA-Z\s]', '', text) # Özel karakterleri sil
                return text.lower().strip()

            df['text'] = df['text'].apply(clean_text)

            # %80 Eğitim, %20 Test
            X = df['text'].values
            y = df['label'].values

            self.raw_text_train, self.raw_text_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            print(f"✅ Veri Hazır: Eğitim Seti: {len(self.raw_text_train)}, Test Seti: {len(self.raw_text_test)}")

        except Exception as e:
            print(f"❌ Veri yükleme hatası: {e}")

    def sonuc_ekle(self, kategori, model_adi, y_true, y_pred, sure):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        yeni_kayit = pd.DataFrame({
            "Kategori": [kategori],
            "Model": [model_adi],
            "Doğruluk": [acc],
            "Kesinlik": [prec],
            "Duyarlılık": [rec],
            "F1-Skoru": [f1],
            "Eğitim Süresi (sn)": [round(sure, 2)]
        })

        self.results_df = pd.concat([self.results_df, yeni_kayit], ignore_index=True)

        # Karmaşıklık Matrisi (Sadece konsola bilgi, görseli sona saklıyoruz)
        print(f"📌 {model_adi} ({kategori}) Tamamlandı. Doğruluk: {acc:.4f} | F1: {f1:.4f}")

# Tez Yönetim Nesnesini Başlat
tez = TezYonetimi()
tez.veri_yukle_ve_hazirla()

# ==================================================================================================
# 3.3. MAKİNE ÖĞRENMESİ YÖNTEMLERİ (5 MODEL)
# ==================================================================================================
print("\n" + "="*50)
print("🚀 BÖLÜM 3.3: MAKİNE ÖĞRENMESİ")
print("="*50)

print("⚙️ TF-IDF Vektörleştirme yapılıyor...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(tez.raw_text_train)
X_test_tfidf = tfidf.transform(tez.raw_text_test)

def run_ml_model(model, name):
    start = time.time()
    model.fit(X_train_tfidf, tez.y_train)
    preds = model.predict(X_test_tfidf)
    end = time.time()
    tez.sonuc_ekle("Makine Öğrenmesi", name, tez.y_test, preds, end-start)

run_ml_model(LogisticRegression(max_iter=1000), "Lojistik Regresyon")
run_ml_model(MultinomialNB(), "Naive Bayes")
run_ml_model(LinearSVC(), "SVM (LinearSVC)")
run_ml_model(RandomForestClassifier(n_estimators=100, n_jobs=-1), "Rastgele Orman")
run_ml_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), "XGBoost")

del X_train_tfidf, X_test_tfidf
gc.collect()

# ==================================================================================================
# 3.4. DERİN ÖĞRENME YÖNTEMLERİ (5 MODEL)
# ==================================================================================================
print("\n" + "="*50)
print("🚀 BÖLÜM 3.4: DERİN ÖĞRENME")
print("="*50)

# Tokenizer ve Padding
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(tez.raw_text_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(tez.raw_text_train), maxlen=MAX_SEQUENCE_LENGTH)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(tez.raw_text_test), maxlen=MAX_SEQUENCE_LENGTH)

def train_dl_model(model_type, name):
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(SpatialDropout1D(0.2))

    if model_type == 'LSTM':
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.0))
    elif model_type == 'CNN':
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'Bi-LSTM':
        model.add(Bidirectional(LSTM(64, dropout=0.2, return_sequences=False)))
    elif model_type == 'GRU':
        model.add(GRU(100, dropout=0.2))
    elif model_type == 'CNN-LSTM':
        model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100, dropout=0.2))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(f"\n🧠 {name} Modeli Eğitiliyor...")
    start = time.time()
    model.fit(X_train_seq, tez.y_train, epochs=3, batch_size=256, validation_split=0.1,
              callbacks=[EarlyStopping(monitor='val_loss', patience=1)], verbose=1)

    preds_prob = model.predict(X_test_seq)
    preds = (preds_prob > 0.5).astype(int).flatten()
    end = time.time()

    tez.sonuc_ekle("Derin Öğrenme", name, tez.y_test, preds, end-start)

train_dl_model('LSTM', "LSTM")
train_dl_model('CNN', "1D-CNN")
train_dl_model('Bi-LSTM', "Bi-LSTM")
train_dl_model('GRU', "GRU")
train_dl_model('CNN-LSTM', "CNN-LSTM (Hibrit)")

del X_train_seq, X_test_seq
gc.collect()

# ==================================================================================================
# 3.5. STANDARD TRANSFORMER YÖNTEMLERİ (5 MODEL)
# ==================================================================================================
print("\n" + "="*50)
print("🚀 BÖLÜM 3.5: STANDARD TRANSFORMER (BERT & TÜREVLERİ)")
print("="*50)

def run_transformer(model_class, tokenizer_class, pretrained_name, custom_name, category="Transformer"):
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n🌀 {custom_name} ({pretrained_name}) Modeli Hazırlanıyor... [Kategori: {category}]")

    try:
        tokenizer = tokenizer_class.from_pretrained(pretrained_name)
    except:
        tokenizer = tokenizer_class.from_pretrained(pretrained_name, mirror='tuna')

    def encode_data(texts, labels):
        input_ids = []
        attention_masks = []
        for text in texts:
            encoded = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=256,
                padding='max_length',
                return_attention_mask=True,
                truncation=True
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
        return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels)

    # Veri hazırlığı (GPU belleğini idareli kullanmak için batch_size 16)
    train_inputs, train_masks, train_labels = encode_data(tez.raw_text_train, tez.y_train)
    test_inputs, test_masks, test_labels = encode_data(tez.raw_text_test, tez.y_test)

    batch_size = 16
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=batch_size)

    model = model_class.from_pretrained(pretrained_name, num_labels=2)
    model.cuda()

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 3

    start = time.time()

    for epoch in range(epochs):
        print(f"  Epoch {epoch+1}/{epochs} Eğitiliyor...")
        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to('cuda')
            b_input_mask = batch[1].to('cuda')
            b_labels = batch[2].to('cuda')

            model.zero_grad()
            try:
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            except TypeError:
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            loss = outputs[0]
            loss.backward()
            optimizer.step()

    print("  Test işlemi yapılıyor...")
    model.eval()
    predictions, true_labels = [], []

    for batch in test_dataloader:
        batch = tuple(t.to('cuda') for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            try:
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            except TypeError:
                outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs[0].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend(np.argmax(logits, axis=1).flatten())
        true_labels.extend(label_ids.flatten())

    end = time.time()
    tez.sonuc_ekle(category, custom_name, true_labels, predictions, end-start)
    del model, train_dataloader, test_dataloader
    torch.cuda.empty_cache()

# Standart Transformerlar
run_transformer(BertForSequenceClassification, BertTokenizer, 'bert-base-uncased', 'BERT', category="Transformer")
run_transformer(RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base', 'RoBERTa', category="Transformer")
run_transformer(DistilBertForSequenceClassification, DistilBertTokenizer, 'distilbert-base-uncased', 'DistilBERT', category="Transformer")
run_transformer(XLNetForSequenceClassification, XLNetTokenizer, 'xlnet-base-cased', 'XLNet', category="Transformer")
run_transformer(AlbertForSequenceClassification, AlbertTokenizer, 'albert-base-v1', 'ALBERT', category="Transformer")

# ==================================================================================================
# 3.6. ADVANCED TRANSFORMER YÖNTEMLERİ (3 MODEL)
# ==================================================================================================
print("\n" + "="*50)
print("👑 BÖLÜM 3.6: GELİŞMİŞ (ADVANCED) TRANSFORMER MİMARİLERİ")
print("="*50)
print("Not: Bu modeller standart BERT yapısını geliştiren yeni nesil mimarilerdir.")

# Kategori ismi artık: "Advanced Transformer"

# 1. DeBERTa-v3
run_transformer(AutoModelForSequenceClassification, AutoTokenizer, 'microsoft/deberta-v3-base', 'DeBERTa-v3', category="Advanced Transformer")

# 2. ELECTRA
run_transformer(AutoModelForSequenceClassification, AutoTokenizer, 'google/electra-base-discriminator', 'ELECTRA', category="Advanced Transformer")

# 3. MPNet
run_transformer(AutoModelForSequenceClassification, AutoTokenizer, 'microsoft/mpnet-base', 'MPNet', category="Advanced Transformer")

# ==================================================================================================
# 3.7. GELİŞMİŞ AKADEMİK GÖRSELLEŞTİRME VE RAPORLAMA
# ==================================================================================================
print("\n" + "="*50)
print("📊 BÖLÜM 3.7: NİHAİ AKADEMİK GRAFİKLER OLUŞTURULUYOR")
print("="*50)

# Veriyi Kaydet
tez.results_df.to_csv("Tez_Sonuclari_Advanced_Dahil.csv", index=False)

# --------------------------------------------------------------------------------------------------
# 1. KATEGORİ BAZLI DETAY RAPORLARI
# --------------------------------------------------------------------------------------------------
def kategori_raporu_ciz(kategori_adi, dosya_adi, renk_paleti):
    df_cat = tez.results_df[tez.results_df["Kategori"] == kategori_adi].copy()
    if len(df_cat) == 0: return

    df_melted = df_cat.melt(id_vars="Model", value_vars=["Doğruluk", "F1-Skoru"], var_name="Metrik", value_name="Skor")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Model", y="Skor", hue="Metrik", data=df_melted, palette=renk_paleti)
    plt.title(f"{kategori_adi} Yöntemleri Kıyaslaması", fontsize=14, fontweight='bold')
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=15)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(dosya_adi)
    plt.show()

# 4 Kategori (Yeni İsimlendirme ile)
kategori_raporu_ciz("Makine Öğrenmesi", "Rapor_ML.png", "Blues")
kategori_raporu_ciz("Derin Öğrenme", "Rapor_DL.png", "Oranges")
kategori_raporu_ciz("Transformer", "Rapor_Transformer.png", "Purples")
kategori_raporu_ciz("Advanced Transformer", "Rapor_Advanced.png", "Reds") # Sota yerine Advanced

# --------------------------------------------------------------------------------------------------
# 2. ISI HARİTASI (HEATMAP)
# --------------------------------------------------------------------------------------------------
final_table = tez.results_df.sort_values(by="F1-Skoru", ascending=False).set_index("Model")
plt.figure(figsize=(12, 12))
sns.heatmap(final_table[['Doğruluk', 'Kesinlik', 'Duyarlılık', 'F1-Skoru']],
            annot=True, cmap='RdYlGn', fmt='.4f', linewidths=1, linecolor='white', cbar=False)
plt.title("Tüm Modellerin Performans Metrikleri", fontsize=16, pad=20)
plt.tight_layout()
plt.savefig("Grafik_IsiHaritasi.png")
plt.show()

# --------------------------------------------------------------------------------------------------
# 3. DETAYLI 4'LÜ KARŞILAŞTIRMA GRAFİĞİ (GRID VIEW)
# --------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Tüm Modellerin Metrik Bazlı Detaylı Karşılaştırması', fontsize=20, fontweight='bold')

metrics = ["Doğruluk", "Kesinlik", "Duyarlılık", "F1-Skoru"]
grid_colors = ["Blues_d", "Greens_d", "Oranges_d", "Purples_d"]

for i, metric in enumerate(metrics):
    row, col = i // 2, i % 2
    sns.barplot(x=metric, y="Model", data=tez.results_df.sort_values(by=metric, ascending=False),
                ax=axes[row, col], palette=grid_colors[i])
    axes[row, col].set_title(f'{metric} Sıralaması', fontsize=14)
    axes[row, col].set_xlim(0.5, 1.0)
    axes[row, col].grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("Grafik_Grid_Kiyaslama.png")
plt.show()

# --------------------------------------------------------------------------------------------------
# 4. SCATTER PLOT (EĞİTİM SÜRESİ vs BAŞARIM)
# --------------------------------------------------------------------------------------------------
plt.figure(figsize=(12, 8))
# Renk haritası güncellendi
colors_scatter = {'Makine Öğrenmesi': 'blue', 'Derin Öğrenme': 'orange',
                  'Transformer': 'purple', 'Advanced Transformer': 'red'}

sns.scatterplot(data=tez.results_df, x="Eğitim Süresi (sn)", y="F1-Skoru",
                hue="Kategori", style="Kategori", s=200, palette=colors_scatter, alpha=0.8)

for i in range(tez.results_df.shape[0]):
    plt.text(x=tez.results_df["Eğitim Süresi (sn)"][i], y=tez.results_df["F1-Skoru"][i]+0.002,
             s=tez.results_df["Model"][i], fontdict=dict(color='black', size=9))

plt.title("Model Verimlilik Analizi (Süre vs F1-Skoru)", fontsize=16)
plt.xscale('log')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("Grafik_Verimlilik_Analizi.png")
plt.show()

# --------------------------------------------------------------------------------------------------
# 5. RADAR (ÖRÜMCEK) GRAFİĞİ - KATEGORİ LİDERLERİ
# --------------------------------------------------------------------------------------------------
# Her kategorinin en iyisini seç
try:
    best_ml = tez.results_df[tez.results_df['Kategori'] == 'Makine Öğrenmesi'].sort_values(by='F1-Skoru', ascending=False).iloc[0]
    best_dl = tez.results_df[tez.results_df['Kategori'] == 'Derin Öğrenme'].sort_values(by='F1-Skoru', ascending=False).iloc[0]
    best_tr = tez.results_df[tez.results_df['Kategori'] == 'Transformer'].sort_values(by='F1-Skoru', ascending=False).iloc[0]
    best_adv = tez.results_df[tez.results_df['Kategori'] == 'Advanced Transformer'].sort_values(by='F1-Skoru', ascending=False).iloc[0]
except:
    best_adv = None

categories = ['Doğruluk', 'Kesinlik', 'Duyarlılık', 'F1-Skoru']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(9, 9))
ax = plt.subplot(111, polar=True)

def radar_ciz(row, color, label):
    if row is None: return
    values = row[categories].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=label, color=color)
    ax.fill(angles, values, color=color, alpha=0.1)

radar_ciz(best_ml, 'blue', f"ML Lider: {best_ml['Model']}")
radar_ciz(best_dl, 'orange', f"DL Lider: {best_dl['Model']}")
radar_ciz(best_tr, 'purple', f"Std. Transformer Lider: {best_tr['Model']}")
radar_ciz(best_adv, 'red', f"Advanced Lider: {best_adv['Model']}")

plt.xticks(angles[:-1], categories, fontsize=12)
plt.ylim(0.5, 1.0)
plt.title("4 Kuşağın Şampiyonları (Radar Analizi)", fontsize=16, y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.savefig("Grafik_Radar_Kiyaslama.png")
plt.show()

# --------------------------------------------------------------------------------------------------
# 6. SONUÇLARI PAKETLE VE İNDİR
# --------------------------------------------------------------------------------------------------
print("\n📦 SONUÇLAR VE GRAFİKLER PAKETLENİYOR...")

klasor_adi = "Tez_Ciktilar_Final"
if os.path.exists(klasor_adi): shutil.rmtree(klasor_adi)
os.makedirs(klasor_adi, exist_ok=True)

dosyalar = [f for f in os.listdir() if f.endswith(".png") or f.endswith(".csv")]
for f in dosyalar:
    shutil.copy(f, f"{klasor_adi}/{f}")

shutil.make_archive("Tez_Nihai_Paket_Advanced", 'zip', klasor_adi)
files.download("Tez_Nihai_Paket_Advanced.zip")
print("\n✅ İŞLEM TAMAMLANDI. ZIP DOSYASI İNDİRİLİYOR.")
