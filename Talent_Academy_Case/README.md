# Data Science Intern Case Study (Physical Medicine & Rehabilitation)

Name: **Beyda Durak**  
Email: **beydadurak@gmail.com** 

## Overview
EDA ve preprocessing adımlarıyla dataset model-ready hale getirilmiştir.
Hedef değişken: **TedaviSuresi**

## Exploratory Data Analysis (EDA)
- **Eksik Değerler:**  
  - Cinsiyet: 169 eksik  
  - Kan Grubu: 675 eksik  
  - Alerji: 944 eksik  
- **Dağılımlar:**  
  - Yaş değişkeni sağa çarpık, çoğunluk 35-55 yaş arasındadır. 
    Daha ileri yaşlarda (70+) daha az hasta bulunmaktadır.
  - Tedavi süresi (TedaviSuresi) geniş dağılıma sahip ve bazı uç değerler mevcuttur.
## Data Preprocessing
- Duplicates kaldırıldı.  
- Cinsiyet ve Kan Grubu normalize edildi, eksikler "Bilinmiyor" ile dolduruldu.  
- KronikHastalik, Alerji, Tanilar, UygulamaYerleri ve Bolum kolonları için `*_Sayisi` özellikleri oluşturuldu.  
- Eksik değerler:
  - Sayısal değişkenlerde median ile dolduruldu.  
  - Kategorik değişkenlerde most frequent ile dolduruldu.  
- Sayısal değişkenler StandardScaler ile ölçeklendi.  
- Kategorik değişkenler OneHotEncoder ile dönüştürüldü. 

## Project Structure
- `data/` : Orijinal dataset
- `scripts/main.py` : Analiz ve preprocessing adımları
- `outputs/df_clean.csv` : Temiz dataset
- `outputs/X_ready_matrix.csv` : Model-ready özellik matrisi
- `outputs/y_target.csv` : Hedef değişken
- `outputs/preprocess_pipeline.joblib` : Preprocessing pipeline
- `docs/findings.md` : Analiz özeti (opsiyonel)

## Requirements
```bash
pip install -r requirements.txt
```

## How to Run
```bash
python scripts/main.py
```