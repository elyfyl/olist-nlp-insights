import pandas as pd
import numpy as np

def operasyonel_analiz_yap():
    print("1. Yorum ve Sipariş verileri yükleniyor...")
    # Yorum verimizi ve sipariş operasyon verimizi okuyoruz
    df_reviews = pd.read_csv("data/cleaned_reviews.csv")
    df_orders = pd.read_csv("data/olist_orders_dataset.csv")
    
    print("2. Veriler 'order_id' üzerinden birleştiriliyor...")
    # İki tabloyu ortak sütun olan order_id üzerinden birleştiriyoruz
    df_merged = pd.merge(df_reviews, df_orders, on="order_id", how="inner")
    
    print("3. Gecikme süreleri hesaplanıyor...")
    # Tarih sütunlarını Pandas'ın anlayacağı datetime formatına çeviriyoruz
    df_merged['order_delivered_customer_date'] = pd.to_datetime(df_merged['order_delivered_customer_date'])
    df_merged['order_estimated_delivery_date'] = pd.to_datetime(df_merged['order_estimated_delivery_date'])
    
    # Gerçek teslimat tarihinden, söz verilen tarihi çıkararak gün cinsinden farkı buluyoruz
    df_merged['gecikme_gunu'] = (df_merged['order_delivered_customer_date'] - df_merged['order_estimated_delivery_date']).dt.days
    
    # Eğer gecikme_gunu 0'dan büyükse sipariş gecikmiştir, küçükse erken teslim edilmiştir
    print("\n--- 4. Sentiment Sınıflarına Göre Ortalama Gecikme ---")
    
    # Nötr (1) yorumları analiz dışı bırakıyoruz
    df_analiz = df_merged[df_merged['sentiment'] != 1].copy()
    
    # Negatif (0) ve Pozitif (2) yorumlar için ortalama gecikme gününü hesaplıyoruz
    sonuclar = df_analiz.groupby('sentiment')['gecikme_gunu'].mean().reset_index()
    
    # Sonuçları ekrana yazdırıyoruz
    for index, row in sonuclar.iterrows():
        durum = "Negatif Yorumlar" if row['sentiment'] == 0 else "Pozitif Yorumlar"
        gun = row['gecikme_gunu']
        
        if gun > 0:
            print(f"{durum}: Ortalama {gun:.1f} gün GECİKME yaşanmış.")
        else:
            print(f"{durum}: Ortalama {abs(gun):.1f} gün ERKEN teslim edilmiş.")

    print("\n✓ Operasyonel kök neden analizi tamamlandı.")

if __name__ == "__main__":
    operasyonel_analiz_yap()