import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

def basari_raporunu_gorsellestir():
    if not os.path.exists('images'):
        os.makedirs('images')

    print("1. Veriler okunuyor ve model hızlıca eğitiliyor...")
    df = pd.read_csv("data/cleaned_reviews.csv")
    df = df[df['sentiment'] != 1].copy()

    X = df['review_comment_message']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    portekizce_stop_words = ["o", "a", "e", "do", "da", "de", "que", "em", "um", "uma", "para", "com", "não", "na", "no"]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=portekizce_stop_words)
    
    X_train_sayisal = vectorizer.fit_transform(X_train)
    X_test_sayisal = vectorizer.transform(X_test)

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_sayisal, y_train)
    
    tahminler = model.predict(X_test_sayisal)

    print("2. Her bir benzersiz skora özel renk atanarak Isı Haritası çiziliyor...")
    
    rapor_sozlugu = classification_report(y_test, tahminler, target_names=["Negatif", "Pozitif"], output_dict=True)
    df_rapor = pd.DataFrame(rapor_sozlugu).T
    
    df_rapor_gorsel = df_rapor.drop(columns=['support'])
    df_rapor_gorsel = df_rapor_gorsel.drop(['accuracy'], errors='ignore')
    df_rapor_gorsel = df_rapor_gorsel.round(2)
    benzersiz_degerler = np.sort(np.unique(df_rapor_gorsel.values))
    sinirlar = [benzersiz_degerler[0] - 0.01]
    for i in range(len(benzersiz_degerler) - 1):
        orta_nokta = (benzersiz_degerler[i] + benzersiz_degerler[i+1]) / 2.0
        sinirlar.append(orta_nokta)
    sinirlar.append(benzersiz_degerler[-1] + 0.01)

    renkler = [
        "#FDF8E2", # 1. Krem (En düşük değerler için)
        "#E8D8C4", # 2. Taş Rengi
        "#F2C6C2", # 3. Toz Pembe
        "#DFC5D3", # 4. Lila
        "#C2A895", # 5. Açık Vizon
        "#B58254", # 6. Cappuccino
        "#A47764", # 7. Açık Kahve
        "#8C6A5D", # 8. Koyu Vizon
        "#6B4F4F"  # 9. En Koyu Ton (En yüksek değerler için)
    ]
    
    # Sadece elimizdeki farklı sayı adedi kadar renk kullanıyoruz
    kullanilacak_renkler = renkler[:len(benzersiz_degerler)]
    ozel_palet = ListedColormap(kullanilacak_renkler)
    
    # Normalizasyon fonksiyonu ile renkleri sayılara kilitliyoruz
    norm = BoundaryNorm(sinirlar, ozel_palet.N)

    plt.figure(figsize=(8, 5))
    
    # norm=norm parametresini ekledik
    sns.heatmap(df_rapor_gorsel, annot=True, cmap=ozel_palet, norm=norm, fmt=".2f", linewidths=1, cbar=True, annot_kws={"size": 14, "weight": "bold"})
    
    plt.title("Lojistik Regresyon Sınıflandırma Başarı Raporu", fontsize=16, fontweight='bold', pad=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)

    kayit_yolu = 'images/06_classification_report_heatmap.png'
    plt.savefig(kayit_yolu, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Başarı raporu başarıyla görselleştirildi ve '{kayit_yolu}' konumuna kaydedildi!")

if __name__ == "__main__":
    basari_raporunu_gorsellestir()