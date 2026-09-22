import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def operasyonel_grafik_olustur():
    print("1. Veriler yükleniyor ve işleniyor...")
    df_reviews = pd.read_csv("data/cleaned_reviews.csv")
    df_orders = pd.read_csv("data/olist_orders_dataset.csv")
    
    df_merged = pd.merge(df_reviews, df_orders, on="order_id", how="inner")
    
    df_merged['order_delivered_customer_date'] = pd.to_datetime(df_merged['order_delivered_customer_date'])
    df_merged['order_estimated_delivery_date'] = pd.to_datetime(df_merged['order_estimated_delivery_date'])
    
    # Gecikme gününü hesaplıyoruz (Negatif sayılar erken teslimatı gösterir)
    df_merged['gecikme_gunu'] = (df_merged['order_delivered_customer_date'] - df_merged['order_estimated_delivery_date']).dt.days
    
    # Nötr yorumları çıkar
    df_analiz = df_merged[df_merged['sentiment'] != 1].copy()
    
    # Grafik için etiketleri (0 ve 2) metne çeviriyoruz
    df_analiz['Yorum Türü'] = df_analiz['sentiment'].map({0: 'Negatif Yorumlar', 2: 'Pozitif Yorumlar'})
    
    # Erken teslimat gün sayısını pozitif olarak göstermek için mutlak değer (abs) alıyoruz
    df_analiz['Erken Teslimat (Gün)'] = df_analiz['gecikme_gunu'].abs()

    print("2. Grafik çiziliyor...")
    # Görselin boyutunu ve temasını ayarlıyoruz
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    
    # Çubuk grafiğini oluşturuyoruz
    barplot = sns.barplot(
        x="Yorum Türü", 
        y="Erken Teslimat (Gün)", 
        data=df_analiz, 
        hue="Yorum Türü",
        legend=False,
        palette={"Negatif Yorumlar": "#cc210e", "Pozitif Yorumlar": "#2DBD2F"},
        width=0.4,       # YENİ EKLENEN KISIM: Çubukları inceltir ve aralarındaki boşluğu artırır
        errorbar=None
    )
    
    # Grafiğe başlık ve etiketler ekliyoruz
    plt.title("Müşteri Memnuniyeti ve Teslimat Hızı İlişkisi", fontsize=14, fontweight='bold')
    plt.ylabel("Ortalama Erken Teslimat Süresi (Gün)", fontsize=12)
    plt.xlabel("")
    
    # Çubukların üzerine sayısal değerleri yazdırıyoruz
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.1f'), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'center', 
                         xytext = (0, 9), 
                         textcoords = 'offset points',
                         fontweight='bold')

    print("3. Grafik images klasörüne kaydediliyor...")
    if not os.path.exists('images'):
        os.makedirs('images')
        
    kayit_yolu = 'images/04_delivery_impact.png'
    plt.savefig(kayit_yolu, dpi=300, bbox_inches='tight')
    print(f"✓ Grafik başarıyla '{kayit_yolu}' konumuna kaydedildi! Sütunlar inceltildi.")

if __name__ == "__main__":
    operasyonel_grafik_olustur()