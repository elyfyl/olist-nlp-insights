import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

def ml_modelini_egit():
    print("1. Temizlenmiş veri okunuyor...")
    # pandas ile temizlediğimiz CSV dosyasını okuyoruz
    df = pd.read_csv("data/cleaned_reviews.csv")
    
    # Nötr (1) yorumları çıkaralım ki model net olarak Negatif ve Pozitif'e odaklansın
    df = df[df['sentiment'] != 1].copy()
    
    # Hedef değişkenimizi (Y) ve metinlerimizi (X) ayırıyoruz
    X = df['review_comment_message']
    y = df['sentiment']
    
    print("2. Veri eğitim ve test setlerine ayrılıyor...")
    # Verinin %80'ini modeli eğitmek için, %20'sini test etmek için bölüyoruz
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("3. TF-IDF ile kelimeler sayısallaştırılıyor...")
    # Sadece en sık geçen 5000 kelimeyi alıyoruz. 
    # Portekizce anlamsız kelimeleri (stop_words) dahil etmemek için manuel bir filtre ekliyoruz.
    portekizce_stop_words = ["o", "a", "e", "do", "da", "de", "que", "em", "um", "uma", "para", "com", "não", "na", "no"]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=portekizce_stop_words)
    
    X_train_sayisal = vectorizer.fit_transform(X_train)
    X_test_sayisal = vectorizer.transform(X_test)
    
    print("4. Lojistik Regresyon Modeli eğitiliyor (Dengesiz sınıf ayarı ile)...")
    # class_weight='balanced' diyerek azınlıkta olan Negatif sınıfa ekstra önem veriyoruz
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_sayisal, y_train)
    
    print("\n--- 5. Model Başarı Raporu ---")
    tahminler = model.predict(X_test_sayisal)
    # Modelin test verisi üzerindeki başarasını (Precision, Recall, F1-Score) yazdırıyoruz
    print(classification_report(y_test, tahminler, target_names=["Negatif", "Pozitif"]))
    
    print("\n--- 6. Negatif Yorumlara Sebep Olan En Kritik 10 Kelime ---")
    # Lojistik Regresyon katsayılarını ve kelime listesini alıyoruz
    kelimeler = vectorizer.get_feature_names_out()
    katsayilar = model.coef_[0]
    
    # Katsayıları sıralayıp en negatif olanları (en küçük değerler) buluyoruz
    en_negatif_indexler = np.argsort(katsayilar)[:10]
    for i in en_negatif_indexler:
        print(f"Kelime: '{kelimeler[i]}' | Etki Puanı: {katsayilar[i]:.2f}")

# Kod çalıştırıldığında fonksiyonumuzu başlatıyoruz
if __name__ == "__main__":
    ml_modelini_egit()