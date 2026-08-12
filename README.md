
# stock-lstm-gru-pytorch

AMZN (Amazon) hissesinin geçmiş kapanış fiyatları kullanılarak, PyTorch ile
uygulanmış **LSTM** ve **GRU** mimarileriyle bir sonraki günün fiyatı /
getirisi tahmin edilmeye çalışılıyor. Proje, temel bir "ham fiyatla tahmin"
denemesinden başlayıp, sırasıyla non-stationarity sorununun tespiti, getiri
(return) bazlı modelleme, çoklu özellik seti, attention mekanizması, early
stopping, walk-forward validation ve hisseler-arası genelleme testine kadar
ilerleyen bir geliştirme sürecini içeriyor.

## Veri Seti

- **Kaynak:** `all_stocks_2006-01-01_to_2018-01-01.csv` (Kaggle — DJIA 30 hisse verisi)
- **Kapsam:** 2006-01-01 – 2018-01-01 arası, 31 farklı hisse (bu projede **AMZN** ile GOOGL kullanıldı)
- **Sütunlar:** `Date, Open, High, Low, Close, Volume, Name`

Veri dosyası büyük olduğu için repoya dahil edilmemiştir — Kaggle'dan indirip
proje kök dizinine (veya `data/` klasörüne) koyman yeterli.

## Kurulum

```bash
git clone https://github.com/<kullanici-adi>/stock-lstm-gru-pytorch.git
cd stock-lstm-gru-pytorch
pip install -r requirements.txt
```

Veri setini indirip proje dizinine yerleştirdikten sonra `AMZN_stock_prediction.ipynb`
dosyasını Jupyter'da açıp baştan sona çalıştırman yeterli.

## Yöntem — Kısa Özet

1. **Veri hazırlığı:** AMZN filtrelemesi, MinMax normalizasyon, sliding window (lookback=20)
2. **Baseline modeller:** Ham `Close` fiyatı üzerinden LSTM ve GRU (2 katman, hidden_size=64)
3. **Bulgu — Non-stationarity:** Ham fiyatla eğitilen modellerde eğitim/test RMSE arasında büyük
   fark gözlendi (fiyat 2006-2017 arasında ~$47'den ~$1170'e çıktığı için model, görmediği
   fiyat aralığında test edilince büyük hata veriyor).
4. **Çözüm — Getiri (Return) bazlı modelleme:** Fiyat yerine günlük yüzde değişim tahmin
   edilerek daha durağan (stationary) bir hedef elde edildi; scaler veri sızıntısını önlemek
   için sadece train setiyle fit edildi.
5. **Genişletmeler:** Çoklu özellik seti (Open/High/Low/Close/Volume/Return) + Dropout,
   attention mekanizmalı LSTM, early stopping + LR scheduling, walk-forward validation,
   naive baseline karşılaştırması, AMZN→GOOGL cross-stock genelleme testi.

## Sonuçlar

**1. Ham fiyat vs. Getiri (Return) bazlı yaklaşım (Test RMSE):**

| Yaklaşım | Model | Train RMSE | Test RMSE |
|---|---|---|---|
| Ham Fiyat | LSTM | 8.1406 | 189.3027 |
| Ham Fiyat | GRU | 4.8023 | 39.8883 |
| Getiri (Return) | LSTM | 0.026887 | 0.016449 |
| Getiri (Return) | GRU | 0.025865 | 0.016402 |

Getiri bazlı yaklaşımda LSTM ve GRU neredeyse aynı performansı gösterdi; ham
fiyat yaklaşımındaki GRU üstünlüğü return uzayında ortadan kalktı.

**2. Çoklu özellik + Dropout (window_size=20, Test RMSE):**

| Model | Test RMSE |
|---|---|
| GRU (çoklu özellik + dropout) | 0.017164 |
| Naive Baseline (dünkü değeri tekrar et) | 0.022831 |
| Attention LSTM | 0.016396 |
| Early Stopping + LR Scheduling GRU | 0.019364 |

Model, naive baseline'ı geçebiliyor — yani sadece "dünkü değeri tekrarla"
stratejisinden daha iyi bir şey öğrenmiş durumda.

**3. Walk-forward validation (4 fold, GRU):**

Ortalama RMSE: **0.023778** (± 0.007656) — fold'lar arası performans, eğitim
verisi büyüdükçe iyileşme eğiliminde (Fold 1: 0.0367 → Fold 4: 0.0169).

**4. Hisseler arası genelleme (AMZN → GOOGL):**

| Model | GOOGL Test RMSE |
|---|---|
| AMZN'de eğitilen model (transfer, yeniden eğitilmeden) | 0.030056 |
| GOOGL'un kendi verisiyle sıfırdan eğitilen model | 0.011734 |

Beklendiği gibi, bir hissede eğitilen model doğrudan başka bir hisseye
uygulandığında, o hissenin kendi verisiyle eğitilen modelden belirgin şekilde
daha kötü performans gösteriyor — hisseye özgü dinamiklerin öğrenilmesi gerektiğini doğruluyor.

## Öğrenilenler & Sonraki Adımlar

- Fiyat serilerinin durağan (stationary) olmaması, ham fiyatla tahmin yapmayı
  yanıltıcı hale getiriyor; getiri/log-getiri bazlı hedefler daha güvenilir.
- Basit bir naive baseline (dünkü değeri tekrarla) her zaman ilk kıyaslama
  noktası olmalı — aksi halde bir modelin gerçekten bir şey "öğrenip
  öğrenmediği" yanlış yorumlanabilir.
- Bir hissede eğitilen model başka bir hisseye genellenemiyor; hisseye özgü
  eğitim gerekiyor.
- Sonraki adım fikirleri: daha fazla teknik gösterge (RSI, hareketli
  ortalamalar), transformer tabanlı mimariler, birden fazla hisseyle ortak
  eğitim (multi-task / multi-ticker).
- **Kritik not:** Hisse senedi fiyat tahmini doğası gereği çok zor bir problem;
  bu proje bir öğrenme/uygulama egzersizi olup gerçek yatırım kararları için
  kullanılmamalıdır (bkz. *AI Snake Oil*, Narayanan & Kapoor).

## Proje Yapısı

```
stock-lstm-gru-pytorch/
├── AMZN_stock_prediction.ipynb   # Ana notebook (veri hazırlığı → model → değerlendirme)
├── requirements.txt
├── README.md
└── data/                          # (opsiyonel) CSV veri dosyası buraya konabilir
```
