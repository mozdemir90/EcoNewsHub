import pandas as pd
import re
import os
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Gerekli NLTK verilerini indir
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Klasörler ve dosya adları
INPUT_FILE = "data/haberler_detayli_lang_tarih1.xlsx"
OUTPUT_FILE = "output/haberler_temizlenmis.xlsx"

stopwords_tr = set(stopwords.words("turkish"))
stopwords_en = set(stopwords.words("english"))

keywords = {
    "dollar": ["dolar", "usd", "kur", "döviz"],
    "gold": ["altın", "gram", "ons"],
    "stock": ["borsa", "hisse", "bist", "endeks"],
    "bitcoin": ["bitcoin", "btc", "kripto", "coin", "ethereum"]
}

up_words = ["art", "yüksel", "rekor", "patlama", "zirve", "ralli", "sıçra", "yukarı"]
down_words = ["düş", "azal", "gerile", "çök", "sert düşüş", "kayıp", "değer kaybı"]

def clean_text(text, lang):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s\dçğıöşüÇĞİÖŞÜ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text, language="turkish" if lang == "tr" else "english")
    stops = stopwords_tr if lang == "tr" else stopwords_en
    tokens = [word for word in tokens if word not in stops]
    return " ".join(tokens)

def detect_labels(text):
    text = text.lower()
    return {
        "label_dollar": int(any(kw in text for kw in keywords["dollar"])),
        "label_gold": int(any(kw in text for kw in keywords["gold"])),
        "label_stock": int(any(kw in text for kw in keywords["stock"])),
        "label_bitcoin": int(any(kw in text for kw in keywords["bitcoin"]))
    }

def estimate_impact(text, topic_keywords):
    text = text.lower()
    if not any(kw in text for kw in topic_keywords):
        return 2
    has_up = any(up in text for up in up_words)
    has_down = any(down in text for down in down_words)
    if has_up and not has_down:
        return 4 if any(word in text for word in ["çok", "sert", "rekor", "patlama"]) else 3
    elif has_down and not has_up:
        return 0 if any(word in text for word in ["sert", "çök", "dibe"]) else 1
    elif has_up and has_down:
        return 2
    else:
        return 2


def standardize_and_fix_dates(df, column='date'):
    """
    Tüm tarihleri standartlaştırır ve eksik tarihleri yukarıdan gelenle doldurur.
    """
    if column not in df.columns:
        print("⚠️  'date' sütunu bulunamadı. Tarihler eklenmeyecek.")
        return df

    df[column] = pd.to_datetime(df[column], errors='coerce', dayfirst=True)
    df[column] = df[column].fillna(method='ffill')
    df[column] = df[column].fillna(pd.Timestamp("2000-01-01"))
    return df

def preprocess_news():
    print(f"📥 Haber dosyası okunuyor: {INPUT_FILE}")
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print(f"HATA: {INPUT_FILE} dosyası bulunamadı. Lütfen dosya adını ve yolunu kontrol edin.")
        return

    if "content" not in df.columns:
        raise ValueError("Excel dosyasında 'content' sütunu bulunamadı.")

    # Tarihleri düzelt
    df = standardize_and_fix_dates(df, column='date')

    # Tekrarları kaldır
    initial_rows = len(df)
    df.drop_duplicates(subset=['content'], keep='first', inplace=True)
    print(f"🔄 Tekrarlayan haberler kaldırıldı. {initial_rows - len(df)} satır silindi.")

    df['content'].fillna('', inplace=True)
    
    clean_texts = []
    langs = []

    for text in df["content"]:
        try:
            lang = detect(text) if len(text) > 10 else "tr" 
        except:
            lang = "unknown"
        langs.append(lang)
        clean = clean_text(text, lang)
        clean_texts.append(clean)

    df["language_detected"] = langs
    df["clean_text"] = clean_texts

    # Boş içerikleri at
    rows_before_cleaning = len(df)
    df = df[(df['clean_text'].str.strip() != '') & (df['clean_text'].str.strip() != 'content')]
    print(f"🧹 Temiz içerik filtresi: {rows_before_cleaning - len(df)} satır silindi.")

    # Etiketler ve etkiler
    labels = []
    impacts = []
    for text in df["clean_text"]:
        labels.append(detect_labels(text))
        impacts.append({
            "impact_dollar": estimate_impact(text, keywords["dollar"]),
            "impact_gold": estimate_impact(text, keywords["gold"]),
            "impact_stock": estimate_impact(text, keywords["stock"]),
            "impact_bitcoin": estimate_impact(text, keywords["bitcoin"])
        })

    for key in ["dollar", "gold", "stock", "bitcoin"]:
        df[f"label_{key}"] = [l[f"label_{key}"] for l in labels]
        df[f"impact_{key}"] = [i[f"impact_{key}"] for i in impacts]

    os.makedirs("output", exist_ok=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if df['date'].notna().any():
        df.loc[df['date'].notna(), 'date'] = df.loc[df['date'].notna(), 'date'].dt.tz_localize(None)

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ Temizlenmiş dosya kaydedildi: {OUTPUT_FILE} ({len(df)} satır)")

if __name__ == "__main__":
    preprocess_news()
