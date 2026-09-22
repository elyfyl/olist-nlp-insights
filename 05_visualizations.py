import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import os

def grafikleri_ciz():

    print("1. Veriler okunuyor...")
    df = pd.read_csv("data/cleaned_reviews.csv")

    # ==========================================
    # GRAFİK 1: SINIF DAĞILIMI (DONUT CHART)
    # ==========================================
    print("2. Sınıf Dağılımı grafiği hazırlanıyor (Mor tonları ile)...")
    
    # Sınıfların sayılarını alıyoruz
    dagilim = df['sentiment'].value_counts()
    etiketler = ['Pozitif (4-5)', 'Negatif (1-2)', 'Nötr (3)']
    
    # Bizim etiketlerimiz 2, 0 ve 1 şeklindeydi, onları eşleştirelim
    degerler = [dagilim.get(2, 0), dagilim.get(0, 0), dagilim.get(1, 0)]
    
    renkler = [ '#d7bde2','#4a235a', "#8a6da7d4"] 

    plt.figure(figsize=(8, 8))
    # Pasta grafiğini çiziyoruz
    plt.pie(degerler, labels=etiketler, colors=renkler, autopct='%1.1f%%', startangle=140, 
            textprops={'fontsize': 12, 'fontweight': 'bold'}, wedgeprops={'edgecolor': 'white'})
    
    # Ortasına beyaz bir daire ekleyerek 'Donut' görünümü veriyoruz
    orta_daire = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(orta_daire)
    
    plt.title("Müşteri Yorumları Sınıf Dağılımı", fontsize=16, fontweight='bold')
    kayit_yolu_1 = 'images/01_sentiment_distribution.png'
    plt.savefig(kayit_yolu_1, dpi=300, bbox_inches='tight')
    plt.close() 
    
    # ==========================================
    # GRAFİK 2: EN KRİTİK 10 KELİME (YATAY BAR)
    # ==========================================
    print("3. Kök Neden Kelime Analizi grafiği hazırlanıyor...")
    
    # Nötr yorumları modelden çıkarıyoruz
    df_ml = df[df['sentiment'] != 1].copy()
    X = df_ml['review_comment_message']
    y = df_ml['sentiment']
    
    # TF-IDF işlemi (Sadece en sık 5000 kelime)
    portekizce_stop_words = ["o", "a", "e", "do", "da", "de", "que", "em", "um", "uma", "para", "com", "não", "na", "no"]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=portekizce_stop_words)
    X_sayisal = vectorizer.fit_transform(X)
    
    # Modeli eğitiyoruz
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_sayisal, y)
    
    # Katsayıları ve kelimeleri çekiyoruz
    kelimeler = vectorizer.get_feature_names_out()
    katsayilar = model.coef_[0]
    
    # En negatif etkileyen 10 kelimeyi buluyoruz (en düşük katsayılar)
    en_negatif_indexler = np.argsort(katsayilar)[:10]
    en_negatif_kelimeler = [kelimeler[i] for i in en_negatif_indexler]
    en_negatif_skorlar = [katsayilar[i] for i in en_negatif_indexler]
    
    # Grafik için veriyi Pandas formatına sokuyoruz
    df_kelimeler = pd.DataFrame({
        'Kelime': en_negatif_kelimeler,
        'Etki Puanı': en_negatif_skorlar
    })
    
    # Çizim işlemi
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Yatay çubuk grafik
    barplot = sns.barplot(
        x='Etki Puanı', 
        y='Kelime', 
        data=df_kelimeler, 
        palette="Reds_r", 
        hue="Kelime", 
        legend=False
    )
    
    plt.title("Negatif Yorumlara Sebep Olan En Kritik 10 Kelime", fontsize=14, fontweight='bold')
    plt.xlabel("Lojistik Regresyon Etki Katsayısı (Daha düşük = Daha Negatif)", fontsize=12)
    plt.ylabel("Kelimeler (Portekizce)", fontsize=12)
    
    kayit_yolu_2 = 'images/02_top_negative_words.png'
    plt.savefig(kayit_yolu_2, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Harika! İki yeni grafik başarıyla oluşturuldu:\n- {kayit_yolu_1}\n- {kayit_yolu_2}")

if __name__ == "__main__":
    grafikleri_ciz()