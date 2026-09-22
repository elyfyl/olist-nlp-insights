import pandas as pd
import os

def veriyi_hazirla():
    print("1. Olist müşteri yorumları veri seti yükleniyor...")
    # Portekizce karakterlerin bozulmaması için utf-8 kullanıyoruz
    df = pd.read_csv("data/olist_order_reviews_dataset.csv", encoding='utf-8')
    
    print(f"Toplam yorum sayısı: {len(df)}")
    
    # Sadece metin içeren yorumları filtrele (boş yorumları at)
    df_text = df.dropna(subset=['review_comment_message']).copy()
    print(f"Metin içeren yorum sayısı: {len(df_text)}")
    
    # Weak Supervision Fonksiyonu
    def sentiment_etiketle(skor):
        if skor <= 2:
            return 0  # Negatif
        elif skor == 3:
            return 1  # Nötr
        else:
            return 2  # Pozitif
            
    print("2. Yıldız puanlarına göre sentiment etiketleri oluşturuluyor...")
    df_text['sentiment'] = df_text['review_score'].apply(sentiment_etiketle)
    
    # Sınıf dağılımını terminale yazdır (Class Imbalance kontrolü)
    dagilim = df_text['sentiment'].value_counts(normalize=True) * 100
    print("\n--- Sınıf Dağılımı (%) ---")
    print(f"Pozitif (4-5 Yıldız): %{dagilim.get(2, 0):.1f}")
    print(f"Negatif (1-2 Yıldız): %{dagilim.get(0, 0):.1f}")
    print(f"Nötr (3 Yıldız): %{dagilim.get(1, 0):.1f}")
    
    # Gerekli sütunları seç ve temizlenmiş veriyi kaydet
    analiz_df = df_text[['order_id', 'review_id', 'review_score', 'review_comment_message', 'sentiment']]
    
    cikis_yolu = "data/cleaned_reviews.csv"
    analiz_df.to_csv(cikis_yolu, index=False, encoding='utf-8')
    print(f"\n✓ NLP için temizlenmiş veri başarıyla kaydedildi: {cikis_yolu}")

if __name__ == "__main__":
    veriyi_hazirla()