import pandas as pd
import numpy as np
from textblob import TextBlob
import re
import warnings
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
warnings.filterwarnings('ignore')

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("⚠️  NLTK indirmeleri yapılamadı, özetleme sınırlı çalışabilir")

# Varlık anahtar kelimeleri
DOLAR_KELIMELER = [
    'dolar', 'usd', 'amerikan doları', 'dollar', 'usdtry', 'usd/try', 'dolar kuru', 'dolar fiyatı', 'dolar endeksi','merkez bankası','faiz','enflasyon'
]
ALTIN_KELIMELER = [
    'altın', 'gram altın', 'gold', 'altın fiyatı', 'altın ons', 'altın onsu', 'altın endeksi', 'altın gram', 'altın piyasası','gold','war','ceasefire','faiz','savaş','war'
]
BORSA_KELIMELER = [
    'borsa', 'bist', 'bist100', 'bist 100', 'hisse', 'hisse senedi', 'endeks', 'stock', 'stock market', 'equity', 'borsa istanbul','yatırım','faiz','güven','faiz','savaş','war'
]
BITCOIN_KELIMELER = [
    'bitcoin', 'btc', 'kripto', 'kripto para', 'kripto para piyasası', 'btc/usd', 'btc fiyatı', 'btc fiyat', 'btc fiyatı', 'btc fiyatları'
]

# Varlık bazlı pozitif/negatif kelimeler
BORSA_POZITIF = [
    'büyüme', 'rekor', 'yükseliş', 'başarı', 'iyileşme', 'olumlu', 'güçlü', 'pozitif', 'ilerleme', 'arttı', 'kazanç', 'başarılı', 'güçlendi',
    'destek', 'fırsat', 'rally', 'boom', 'bull', 'bullish', 'gain', 'profit', 'increase', 'rise', 'expansion', 'upturn', 'surge', 'recovery',
    'positive', 'up', 'yatırım', 'alım', 'talep', 'yeni rekor', 'yeni zirve', 'yeni yüksek', 'yeni kazanç', 'yeni fırsat', 'yatırımcı', 'hisse', 'endeks', 'borsa', 'bist', 'bist100', 'borsa istanbul'
]
BORSA_NEGATIF = [
    'kriz', 'düşüş', 'düştü', 'kaybetti', 'sorun', 'zorluk', 'risk', 'olumsuz', 'zayıf', 'negatif', 'gerileme', 'düşük', 'azaldı', 'kayıp',
    'başarısız', 'zayıfladı', 'tehlike', 'çöküş', 'dip', 'crash', 'bear', 'bearish', 'loss', 'decline', 'drop', 'fall', 'plunge', 'recession',
    'decrease', 'setback', 'downturn', 'slump', 'crisis', 'down', 'zarar', 'zarar etti', 'zarar açıkladı', 'zarar duyurdu', 'zarar bekleniyor',
    'zarar riski', 'zarar olasılığı', 'zarar tahmin ediliyor', 'tartışma', 'düşüşe geçti', 'zarar gördü', 'kayıp yaşadı'
]
DOLAR_POZITIF = BORSA_NEGATIF  # Dolar için borsa negatifleri pozitif
DOLAR_NEGATIF = BORSA_POZITIF  # Dolar için borsa pozitifleri negatif
ALTIN_POZITIF = DOLAR_POZITIF
ALTIN_NEGATIF = DOLAR_NEGATIF
BITCOIN_POZITIF = [
    'yükseliş', 'artış', 'rekor', 'kazanç', 'büyüme', 'talep', 'pozitif', 'güçlü', 'rally', 'bull', 'bullish', 'gain', 'profit', 'increase', 'rise', 'surge', 'recovery', 'positive', 'up', 'yeni rekor', 'yeni zirve', 'yeni yüksek', 'yeni kazanç', 'yeni fırsat', 'yatırım', 'kripto', 'btc', 'bitcoin', 'kripto para', 'kripto para piyasası'
]
BITCOIN_NEGATIF = [
    'düşüş', 'düştü', 'kaybetti', 'kriz', 'çöküş', 'bear', 'bearish', 'loss', 'decline', 'drop', 'fall', 'plunge', 'recession', 'decrease', 'setback', 'downturn', 'slump', 'crisis', 'down', 'zarar', 'zarar etti', 'zarar açıkladı', 'zarar duyurdu', 'zarar bekleniyor', 'zarar riski', 'zarar olasılığı', 'zarar tahmin ediliyor', 'hack', 'dolandırıcılık', 'regülasyon', 'yasak', 'kapatma', 'iflas', 'hacklenme'
]

# Pozitif ve negatif kelimeler (kısa, sade liste)
POZITIF_KELIMELER = [
    'artış', 'yükseldi', 'kazandı', 'büyüme', 'gelişme', 'iyileşme', 'olumlu', 'güçlü', 'pozitif', 'ilerleme', 'arttı', 'kazanç', 'başarılı', 'güçlendi', 
    'destek', 'fırsat', 'rekor', 'zirve', 'rally', 'boom', 'bull', 'bullish', 'gain', 'profit', 'increase', 'rise', 'growth', 'expansion', 'upturn', 'surge',
     'recovery', 'positive', 'up', 'yükseliş', 'yatırım', 'alım', 'talep', 'yeni rekor', 'yeni zirve', 'yeni yüksek', 'yeni kazanç', 'yeni fırsat','yükseliş',
     'yatırım','fırsat','alım','satım','yatırımcılar','yatırım fırsatı', 'yatırım stratejisi', 'yatırım analizi', 'yatırım tavsiyesi',
    'yatırımcı duyarlılığı', 'yatırımcı güveni', 'yatırımcı davranışı',
    'yatırımcı psikolojisi', 'yatırımcı eğilimleri', 'yatırımcı beklentileri',
    'yatırımcı kararları', 'yatırımcı ilişkileri', 'yatırımcı sunumu',
    'yatırımcı raporu', 'yatırımcı toplantısı', 'yatırımcı konferansı',
    'yatırımcı etkinliği', 'yatırımcı duyurusu', 'yatırımcı bilgilendirmesi',
    'yatırımcı ilişkileri yönetimi', 'yatırımcı ilişkileri stratejisi',
    'yatırımcı ilişkileri analizi', 'yatırımcı ilişkileri raporu',
]
NEGATIF_KELIMELER = [
    'düşüş', 'düştü', 'kaybetti', 'kriz', 'sorun', 'zorluk', 'risk', 'olumsuz', 'zayıf', 'negatif', 'gerileme', 'düşük', 'azaldı', 'kayıp',
     'başarısız', 'zayıfladı', 'tehlike', 'çöküş', 'dip', 'crash','bear', 'bearish', 'loss', 'decline', 'drop', 'fall', 'plunge','savaş','zarar bekleniyor', 'zarar tahmin ediliyor', 'zarar olasılığı', 'zarar riski',
    'zarar riski taşıyor', 'zarar riski yüksek', 'zarar riski artıyor','düşüş', 'düştü', 'kaybetti', 'kriz', 'sorun', 'zorluk', 'risk',
    'olumsuz', 'zayıf', 'negatif', 'gerileme', 'düşük', 'azaldı', 'kayıp', 'başarısız', 'zayıfladı', 'tehlike', 'çöküş', 'dip', 'crash','bear', 'bearish', 'loss', 'decline', 'drop', 'fall', 'plunge','savaş',
]

# Türkçe ve İngilizce artış/azalış anahtar kelimeleri
ARTIS_KELIMELERI_TR = [
    'yüksel', 'art', 'rekor', 'zirve', 'güçlü', 'talep', 'canlan', 'toparlan', 'ivme', 'pozitif', 'ralli', 'patlama', 'sıçrama', 'yeni rekor', 'yeni zirve', 'yeni yüksek', 'değer kazandı', 'güçlü alım', 'pozitif ayrıştı', 'yeni kazanç', 'yeni fırsat', 'canlandı', 'ivme kazandı', 'açık ara', 'patladı', 'rekor kırdı', 'rekor seviyede', 'rekor artış', 'rekor yükseliş', 'hızlı yükseliş', 'tarihi yükseliş', 'aştı', 'güçlü alım', 'talep arttı', 'pozitif görünüm', 'pozitif sinyal', 'yeni hedef', 'canlandı', 'ivme kazandı', 'patladı', 'rekor kırdı', 'rekor seviyede', 'rekor artış', 'rekor yükseliş', 'hızlı yükseliş', 'tarihi yükseliş'
]
AZALIS_KELIMELERI_TR = [
    'düş', 'azal', 'çök', 'gerile', 'değer kaybı', 'sert düşüş', 'çakıldı', 'satış baskısı', 'panik', 'çöküş', 'dibe vurdu', 'negatif ayrıştı', 'sert kayıp', 'sert satış', 'sert gerileme', 'sert azalış', 'sert kayıp', 'sert düşüş', 'sert çöküş', 'sert değer kaybı', 'sert panik', 'sert çakılma', 'sert dibe vurma', 'negatif görünüm', 'negatif sinyal', 'negatif ayrıştı', 'güven kaybı', 'zayıflattı', 'belirsizlik', 'çalkantı', 'kayıp', 'zarar', 'iflas', 'hack', 'regülasyon', 'yasak', 'kapatma', 'çakıldı', 'dibe vurdu', 'panik', 'çöküş', 'çök', 'gerile', 'azal', 'düş', 'satış baskısı', 'negatif ayrıştı'
]
ARTIS_KELIMELERI_EN = [
    'rise', 'increase', 'record', 'surge', 'jump', 'rally', 'gain', 'climb', 'soar', 'peak', 'bull', 'positive', 'demand', 'rebound', 'momentum', 'new high', 'new record', 'strong buy', 'outperform', 'breakout', 'spike', 'boost', 'advance', 'strengthen', 'improve', 'expand', 'recover', 'hit high', 'hit record', 'hit peak', 'all-time high', 'skyrocket', 'rocket', 'explode', 'shoot up', 'break record', 'break high', 'breakout', 'bullish', 'optimism', 'optimistic', 'strong demand', 'strong rally', 'strong gain', 'strong momentum', 'positive signal', 'positive outlook', 'new target', 'new opportunity'
]
AZALIS_KELIMELERI_EN = [
    'fall', 'decrease', 'drop', 'decline', 'crash', 'plunge', 'slump', 'bear', 'negative', 'loss', 'panic', 'sell-off', 'collapse', 'downturn', 'uncertainty', 'weakness', 'losses', 'underperform', 'pullback', 'correction', 'tumble', 'plummet', 'dive', 'sink', 'slide', 'bearish', 'pessimism', 'pessimistic', 'weak demand', 'weakness', 'negative signal', 'negative outlook', 'uncertain', 'volatile', 'volatility', 'instability', 'bankrupt', 'hack', 'regulation', 'ban', 'shutdown', 'hit low', 'hit bottom', 'all-time low', 'bottomed', 'bottom', 'collapse', 'crash', 'sell pressure', 'sell-off', 'sharp drop', 'sharp decline', 'sharp fall', 'sharp loss', 'sharp plunge', 'sharp correction', 'sharp pullback', 'sharp weakness', 'sharp volatility', 'sharp uncertainty'
]

class VarlikBazliHaberAnalizi:
    def __init__(self, excel_dosya_yolu, haber_sutun='content', dil_sutun='language'):
        self.excel_dosya_yolu = excel_dosya_yolu
        self.haber_sutun = haber_sutun
        self.dil_sutun = dil_sutun
        self.df = None
        self.skor_sutunlari = ['dolar_skor', 'altin_skor', 'borsa_skor', 'bitcoin_skor']

    def veri_yukle(self):
        self.df = pd.read_excel(self.excel_dosya_yolu)
        if self.haber_sutun not in self.df.columns:
            raise ValueError(f"'{self.haber_sutun}' sütunu bulunamadı!")
        if self.dil_sutun not in self.df.columns:
            self.df[self.dil_sutun] = 'tr'

    def veri_temizle(self):
        if self.df is None:
            raise ValueError("Veri yüklenmeden temizlik yapılamaz.")
        onceki = len(self.df)
        # Boş değerleri sil
        self.df = self.df.dropna(subset=[self.haber_sutun])
        # Boş veya sadece boşluk olanları sil
        self.df = self.df[self.df[self.haber_sutun].astype(str).str.strip() != '']
        # Çok kısa haberleri sil
        self.df = self.df[self.df[self.haber_sutun].astype(str).str.len() >= 20]
        # Tekrarlayan haberleri sil
        self.df = self.df.drop_duplicates(subset=[self.haber_sutun])
        self.df = self.df.reset_index(drop=True)
        print(f"Veri temizliği: {onceki - len(self.df)} satır silindi. Kalan: {len(self.df)}")

    def dil_tespiti_uygula(self):
        if self.df is None:
            raise ValueError("Veri yüklenmeden dil tespiti yapılamaz.")
        def tespit_et(metin):
            turkce_karakterler = set('çğıöşüÇĞIÖŞÜ')
            if any(char in turkce_karakterler for char in str(metin)):
                return 'tr'
            return 'en'
        mask = self.df[self.dil_sutun].isna() | (self.df[self.dil_sutun].astype(str).str.strip() == '')
        self.df.loc[mask, self.dil_sutun] = self.df.loc[mask, self.haber_sutun].apply(tespit_et)
        print(f"Dil tespiti: {mask.sum()} satır için otomatik dil atandı.")

    def varlik_duygu_skoru(self, metin, kelime_listesi, dil):
        metin = str(metin).lower()
        if not any(k in metin for k in kelime_listesi):
            return np.nan  # O varlıkla ilgili değil
        kelimeler = metin.split()
        pozitif = sum(1 for k in kelimeler if k in POZITIF_KELIMELER)
        negatif = sum(1 for k in kelimeler if k in NEGATIF_KELIMELER)
        if len(kelimeler) == 0:
            return 2  # Nötr
        net = (pozitif - negatif) / len(kelimeler)
        # Skoru 0-5 aralığına ölçekle
        if net <= -0.2:
            return 0
        elif net <= -0.1:
            return 1
        elif net <= 0.1:
            return 2
        elif net <= 0.2:
            return 3
        elif net <= 0.4:
            return 4
        else:
            return 5

    def haber_ozetle(self, metin, dil='tr', max_cumle=2):
        try:
            if not isinstance(metin, str) or len(metin) < 40:
                return metin
            cumleler = sent_tokenize(metin)
            if len(cumleler) <= max_cumle:
                return metin
            # Basit extractive: en uzun cümleleri seç
            secili = sorted(cumleler, key=lambda x: len(x), reverse=True)[:max_cumle]
            # Orijinal sıraya göre sırala
            secili = sorted(secili, key=lambda x: cumleler.index(x))
            return ' '.join(secili)
        except Exception:
            return metin[:200] + '...'

    def tum_varlik_skorlarini_hesapla_ve_ozetle(self):
        if self.df is None:
            raise ValueError("Veri yüklenmeden skor hesaplanamaz. Önce veri_yukle() çağrılmalı.")
        def skorla(metin, dil, kelime_listesi, pozitif_liste, negatif_liste):
            metin = str(metin).lower()
            # Dil kontrolü
            if dil == 'en':
                artis_kelimeleri = ARTIS_KELIMELERI_EN
                azalis_kelimeleri = AZALIS_KELIMELERI_EN
            else:
                artis_kelimeleri = ARTIS_KELIMELERI_TR
                azalis_kelimeleri = AZALIS_KELIMELERI_TR
            # Önce cümle bazlı anahtar kelime kontrolü
            cumle_skor = varlik_cumle_skoru(metin, kelime_listesi, artis_kelimeleri, azalis_kelimeleri)
            if cumle_skor is not None:
                return cumle_skor
            # Yoksa kelime oranı ile devam
            if any(k in metin for k in kelime_listesi):
                kelimeler = metin.split()
                pozitif = sum(1 for k in kelimeler if k in pozitif_liste)
                negatif = sum(1 for k in kelimeler if k in negatif_liste)
                if len(kelimeler) == 0:
                    return 3
                net = (pozitif - negatif) / len(kelimeler)
                # Daha dengeli skor aralığı
                if net <= -0.30:
                    return 0
                elif net <= -0.15:
                    return 1
                elif net <= 0.10:
                    return 2
                elif net <= 0.25:
                    return 3
                elif net <= 0.40:
                    return 4
                else:
                    return 5
            else:
                return 3
        # Özet sütunu
        self.df['ozet'] = self.df.apply(lambda row: self.haber_ozetle(row[self.haber_sutun], row[self.dil_sutun]), axis=1)
        # Varlık skorları (ilgili değilse 3 atanır, varlık bazlı pozitif/negatif listelerle ve cümle bazlı anahtar kelimelerle)
        self.df['dolar_skor'] = self.df.apply(lambda row: skorla(
            row[self.haber_sutun], row[self.dil_sutun], DOLAR_KELIMELER, DOLAR_POZITIF, DOLAR_NEGATIF
        ), axis=1)
        self.df['altin_skor'] = self.df.apply(lambda row: skorla(
            row[self.haber_sutun], row[self.dil_sutun], ALTIN_KELIMELER, ALTIN_POZITIF, ALTIN_NEGATIF
        ), axis=1)
        self.df['borsa_skor'] = self.df.apply(lambda row: skorla(
            row[self.haber_sutun], row[self.dil_sutun], BORSA_KELIMELER, BORSA_POZITIF, BORSA_NEGATIF
        ), axis=1)
        self.df['bitcoin_skor'] = self.df.apply(lambda row: skorla(
            row[self.haber_sutun], row[self.dil_sutun], BITCOIN_KELIMELER, BITCOIN_POZITIF, BITCOIN_NEGATIF
        ), axis=1)
        # Finansal varlıklar arası korelasyon düzeltmesi (yumuşatılmış)
        for idx, row in self.df.iterrows():
            # Dolar ve borsa ters korelasyon
            if row['dolar_skor'] >= 4:
                self.df.at[idx, 'borsa_skor'] = min(row['borsa_skor'], 2)
            if row['borsa_skor'] >= 4:
                self.df.at[idx, 'dolar_skor'] = min(row['dolar_skor'], 2)
            # Dolar ve gram altın doğru korelasyon (ama aynı anda 5 olmasın, biri 5 ise diğeri en fazla 4)
            if row['dolar_skor'] == 5:
                self.df.at[idx, 'altin_skor'] = min(max(row['altin_skor'], 4), 4)
            if row['altin_skor'] == 5:
                self.df.at[idx, 'dolar_skor'] = min(max(row['dolar_skor'], 4), 4)
        # Bitcoin bağımsız, mevcut skoru korunacak

    def ozet_istatistikleri_hazirla(self):
        if self.df is None:
            raise ValueError("Veri yüklenmeden özet istatistik hazırlanamaz. Önce veri_yukle() çağrılmalı.")
        data = []
        for varlik, sutun in zip(['Dolar', 'Altın', 'Borsa', 'Bitcoin'], self.skor_sutunlari):
            skorlar = self.df[sutun].dropna()
            data.append({
                'Varlık': varlik,
                'Ortalama Skor': skorlar.mean() if not skorlar.empty else np.nan,
                'Min Skor': skorlar.min() if not skorlar.empty else np.nan,
                'Max Skor': skorlar.max() if not skorlar.empty else np.nan,
                '0 (Düşüş)': (skorlar == 0).sum(),
                '1': (skorlar == 1).sum(),
                '2 (Nötr-)': (skorlar == 2).sum(),
                '3 (Nötr+)': (skorlar == 3).sum(),
                '4': (skorlar == 4).sum(),
                '5 (Yükseliş)': (skorlar == 5).sum(),
            })
        return pd.DataFrame(data)

    def kaydet(self, cikti_dosya='analiz_sonuclari2.xlsx'):
        if self.df is None:
            raise ValueError("Veri yüklenmeden kaydedilemez. Önce veri_yukle() çağrılmalı.")
        with pd.ExcelWriter(cikti_dosya, engine='openpyxl') as writer:
            # Ana sheet: haber, özet ve skorlar
            ana_kolonlar = [self.haber_sutun, 'ozet', self.dil_sutun, 'dolar_skor', 'altin_skor', 'borsa_skor', 'bitcoin_skor']
            self.df[ana_kolonlar].to_excel(writer, sheet_name='Haber_Skorlari', index=False)
            ozet_df = self.ozet_istatistikleri_hazirla()
            ozet_df.to_excel(writer, sheet_name='Ozet_Istatistikler', index=False)
        print(f"Sonuçlar '{cikti_dosya}' dosyasına kaydedildi.")

    def calistir(self, cikti_dosya='analiz_sonuclari2.xlsx'):
        self.veri_yukle()
        self.veri_temizle()
        self.dil_tespiti_uygula()
        self.tum_varlik_skorlarini_hesapla_ve_ozetle()
        self.kaydet(cikti_dosya)

def varlik_cumle_skoru(metin, varlik_kelimeleri, artis_kelimeleri, azalis_kelimeleri):
    metin = str(metin).lower()
    cumleler = sent_tokenize(metin)
    skorlar = []
    for cumle in cumleler:
        if any(v in cumle for v in varlik_kelimeleri):
            artis = any(a in cumle for a in artis_kelimeleri)
            azalis = any(a in cumle for a in azalis_kelimeleri)
            if artis and not azalis:
                skorlar.append(5)
            elif azalis and not artis:
                skorlar.append(1)
            elif artis and azalis:
                skorlar.append(3)
    if skorlar:
        return max(skorlar)  # En güçlü sinyali kullan
    return None

if __name__ == "__main__":
    analiz = VarlikBazliHaberAnalizi(
        excel_dosya_yolu='haberler_detayli_lang_tarih1.xlsx',
        haber_sutun='content',
        dil_sutun='language'
    )
    analiz.calistir('analiz_sonuclari2.xlsx')