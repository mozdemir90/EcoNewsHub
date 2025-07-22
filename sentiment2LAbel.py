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

# NLTK indirmeleri
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("⚠️  NLTK indirmeleri yapılamadı")

POZITIF_KELIMELER = [
    'artış', 'yükseldi', 'kazandı', 'başarı', 'büyüme', 'gelişme', 'iyileşme',
    'olumlu', 'güçlü', 'pozitif', 'ilerleme', 'yüksek', 'arttı', 'kazanç',
    'başarılı', 'güçlendi', 'destek', 'fırsat', 'rekor', 'zirve', 'tırmanış',
    'rally', 'boom', 'bull', 'bullish', 'gain', 'profit', 'increase', 'rise','savaş',
    'growth', 'expansion', 'upturn', 'surge', 'recovery', 'positive',
    'up', 'bull market', 'economic growth', 'discussion', 'yükseliş','yatırım','yatırımcı','fırsat','alım','satım','yatırımcılar',
    'yatırım fırsatı', 'yatırım stratejisi', 'yatırım analizi', 'yatırım tavsiyesi',
    'yatırımcı duyarlılığı', 'yatırımcı güveni', 'yatırımcı davranışı',
    'yatırımcı psikolojisi', 'yatırımcı eğilimleri', 'yatırımcı beklentileri',
    'yatırımcı kararları', 'yatırımcı ilişkileri', 'yatırımcı sunumu',
    'yatırımcı raporu', 'yatırımcı toplantısı', 'yatırımcı konferansı',
    'yatırımcı etkinliği', 'yatırımcı duyurusu', 'yatırımcı bilgilendirmesi',
    'yatırımcı ilişkileri yönetimi', 'yatırımcı ilişkileri stratejisi',
    'yatırımcı ilişkileri analizi', 'yatırımcı ilişkileri raporu',
]

NEGATIF_KELIMELER = [
    'düşüş', 'düştü', 'kaybetti', 'kriz', 'sorun', 'zorluk', 'risk',
    'olumsuz', 'zayıf', 'negatif', 'gerileme', 'düşük', 'azaldı', 'kayıp',
    'başarısız', 'zayıfladı', 'tehlike', 'çöküş', 'dip', 'crash',
    'bear', 'bearish', 'loss', 'decline', 'drop', 'fall', 'plunge','savaş',
    'recession', 'decrease', 'setback', 'downturn', 'slump', 'crisis',
    'down', 'decline', 'recession', 'bear market', 'economic downturn','tartışma',
    'düşüş', 'düşüşe geçti', 'zarar', 'zarar gördü', 'kayıp yaşadı',
    'kayıplar', 'zarar etti', 'zarar yazdı', 'zarar açıkladı', 'zarar duyurdu',
    'zarar bekleniyor', 'zarar tahmin ediliyor', 'zarar olasılığı', 'zarar riski',
    'zarar riski taşıyor', 'zarar riski yüksek', 'zarar riski artıyor',
    'zarar riski düşüyor', 'zarar riski azalıyor', 'zarar riski kontrol altında',
    'zarar riski yönetiliyor', 'zarar riski azaltılıyor', 'zarar riski önleniyor',
    'zarar riski engelleniyor', 'zarar riski azaltılmaya çalışılıyor',
]

FINANSAL_KELIMELER = [
    'dolar', 'euro', 'altın', 'borsa', 'bitcoin', 'piyasa', 'ekonomi',
    'faiz', 'enflasyon', 'yatırım', 'hisse', 'kur', 'para', 'banka',
    'tcmb', 'fed', 'bist', 'usd', 'eur', 'btc', 'stock', 'market','hazine',
    'bond', 'commodity', 'forex', 'cryptocurrency', 'crypto', 'gold',
    'silver', 'oil', 'petrol', 'commodity', 'inflation', 'interest rate',
    'central bank', 'federal reserve', 'exchange rate', 'currency',
    'investment', 'equity', 'shares', 'capital market', 'financial',
    'financial market', 'trading', 'trader', 'portfolio', 'asset',
    'assets', 'liquidity', 'volatility', 'risk management', 'hedge',
    'derivative', 'futures', 'options', 'mutual fund', 'hedge fund','maliye',
    'finans', 'finansal', 'finansman', 'finansal piyasa', 'finansal analiz',
    'finansal rapor', 'finansal tablo', 'finansal istatistik', 'finansal veri','borsa','borsalar',
    'borsa endeksi', 'borsa yatırım', 'borsa analizi', 'borsa raporu',
    'borsa haberleri', 'borsa yorumları', 'borsa stratejisi', 'borsa eğilimleri',
    'borsa beklentileri', 'borsa tahminleri', 'borsa performansı', 'borsa trendi',
    'borsa hareketleri', 'borsa dalgalanmaları', 'borsa volatilitesi',
    'borsa likiditesi', 'borsa hacmi', 'borsa fiyatları', 'borsa değerleme',
    'borsa endeksi', 'borsa göstergeleri', 'borsa analiz araçları','stock market',
    'stock markets', 'stock index', 'stock investment', 'stock analysis',
    'stock report', 'stock statistics', 'stock data', 'stock exchange',
    'stock trading', 'stock trader', 'stock portfolio', 'stock asset',
    'stock assets', 'stock liquidity', 'stock volatility', 'stock risk management',
    'stock hedge', 'stock derivative', 'stock futures', 'stock options',
    'stock mutual fund', 'stock hedge fund', 'financial market analysis',
]

TR_STOP_WORDS = set([
    've', 'ile', 'bir', 'bu', 'şu', 'o', 'ben', 'sen', 'biz', 'siz', 'onlar',
    'ki', 'da', 'de', 'ta', 'te', 'ya', 'ye', 'mi', 'mu', 'mı', 'mü',
    'için', 'gibi', 'kadar', 'daha', 'çok', 'az', 'en', 'hem', 'ya', 'veya',
    'ama', 'ancak', 'fakat', 'lakin', 'ne', 'nasıl', 'neden', 'niçin', 'niye',
    'hangi', 'hangisi', 'kim', 'kime', 'kimin', 'kimse', 'hiç', 'hiçbir'
])


# Varlıklar. "bitcoin" yerine sadece "btc" kullanalım:
VARLIKLAR = ['dolar', 'gram altın', 'borsa', 'btc']


class KapsamliHaberAnalizi:
    def __init__(self, excel_dosya_yolu):
        self.excel_dosya_yolu = excel_dosya_yolu
        self.df = None
        self.original_df = None
        self.temizlik_istatistikleri = {}

    def veri_yukle(self, haber_sutun, dil_sutun):
        print("\n📂 1. VERİ YÜKLEME")
        print("-" * 30)
        try:
            self.df = pd.read_excel(self.excel_dosya_yolu)
            self.original_df = self.df.copy()
            print(f"✅ Dosya başarıyla yüklendi")
            print(f"   📊 Toplam kayıt: {len(self.df)}")
            print(f"   📋 Sütunlar: {list(self.df.columns)}")

            if haber_sutun not in self.df.columns:
                print(f"❌ Hata: '{haber_sutun}' sütunu bulunamadı!")
                return False

            if dil_sutun not in self.df.columns:
                self.df[dil_sutun] = 'tr'
                print(f"⚠️  Dil sütunu yok, 'tr' atandı")

            self.haber_sutun = haber_sutun
            self.dil_sutun = dil_sutun

            return True
        except Exception as e:
            print(f"❌ Veri yükleme hatası: {e}")
            return False

    def kapsamli_temizlik(self):
        print("\n🧹 2. KAPSAMLI VERİ TEMİZLİĞİ")
        print("-" * 30)
        baslangic_sayisi = len(self.df)

        # 2.1 Temel temizlik
        self._temel_temizlik()

        # 2.2 Metin temizliği
        self._metin_temizligi()

        # 2.3 Dil tespiti
        self._dil_tespiti()

        # 2.4 Özel temizlik
        self._ozel_temizlik()

        # 2.5 Final kontrol
        self._final_kontrol()

        self.temizlik_istatistikleri = {
            'baslangic_sayisi': baslangic_sayisi,
            'bitis_sayisi': len(self.df),
            'temizlenen_kayit': baslangic_sayisi - len(self.df),
            'temizlik_orani': (baslangic_sayisi - len(self.df)) / baslangic_sayisi * 100
        }

        print(f"\n✅ TEMİZLİK TAMAMLANDI")
        print(f"   📉 Temizlenen kayıt: {self.temizlik_istatistikleri['temizlenen_kayit']}")
        print(f"   📊 Temizlik oranı: %{self.temizlik_istatistikleri['temizlik_orani']:.1f}")
        print(f"   📈 Kalan kayıt: {self.temizlik_istatistikleri['bitis_sayisi']}")

    def _temel_temizlik(self):
        print("   🔸 Temel temizlik...")

        onceki = len(self.df)
        self.df = self.df.dropna(subset=[self.haber_sutun])
        print(f"      ✓ Boş değerler: {onceki - len(self.df)} kayıt temizlendi")

        onceki = len(self.df)
        self.df = self.df[self.df[self.haber_sutun].astype(str).str.strip() != '']
        print(f"      ✓ Boş stringler: {onceki - len(self.df)} kayıt temizlendi")

        onceki = len(self.df)
        self.df = self.df[self.df[self.haber_sutun].astype(str).str.len() >= 20]
        print(f"      ✓ Çok kısa metinler: {onceki - len(self.df)} kayıt temizlendi")

        onceki = len(self.df)
        self.df = self.df.drop_duplicates(subset=[self.haber_sutun])
        print(f"      ✓ Duplicate metinler: {onceki - len(self.df)} kayıt temizlendi")

    def _metin_temizligi(self):
        print("   🔸 Metin temizliği...")

        def metin_temizle(metin):
            if pd.isna(metin):
                return ""

            metin = str(metin)
            metin = re.sub(r'<.*?>', '', metin)
            metin = re.sub(r'http[s]?://\S+', '', metin)
            metin = re.sub(r'\S+@\S+', '', metin)
            metin = re.sub(r'[\+]?[(]?[\d\s\-\(\)]{10,}', '', metin)
            metin = re.sub(r'\d{8,}', '', metin)
            metin = re.sub(r'[^\w\s.,!?;:()\"\'çğıöşüÇĞIÖŞÜ\-]', ' ', metin)
            metin = re.sub(r'\s+', ' ', metin)
            metin = re.sub(r'(.)\1{3,}', r'\1\1', metin)
            return metin.strip()

        self.df[self.haber_sutun] = self.df[self.haber_sutun].apply(metin_temizle)
        print(f"      ✓ Metin temizliği tamamlandı")

    def _dil_tespiti(self):
        print("   🔸 Dil tespiti...")

        def dil_tespit(metin):
            turkce_karakterler = set('çğıöşüÇĞIÖŞÜ')
            if any(char in turkce_karakterler for char in metin):
                return 'tr'
            return 'en'

        self.df['tespit_edilen_dil'] = self.df[self.haber_sutun].apply(dil_tespit)

        self.df[self.dil_sutun] = self.df[self.dil_sutun].fillna(
            self.df['tespit_edilen_dil']
        )

        dil_dagilimi = self.df[self.dil_sutun].value_counts()
        print(f"      ✓ Dil dağılımı: {dict(dil_dagilimi)}")

    def _ozel_temizlik(self):
        print("   🔸 Özel temizlik...")

        def buyuk_harf_duzelt(metin):
            if metin.isupper() and len(metin) > 50:
                return metin.lower().capitalize()
            return metin

        self.df[self.haber_sutun] = self.df[self.haber_sutun].apply(buyuk_harf_duzelt)

        def finansal_kelime_var(metin):
            metin_lower = metin.lower()
            return any(kelime in metin_lower for kelime in FINANSAL_KELIMELER)

        self.df['finansal_kelime_var'] = self.df[self.haber_sutun].apply(finansal_kelime_var)

        finansal_oran = self.df['finansal_kelime_var'].sum() / len(self.df) * 100
        print(f"      ✓ Finansal kelime içeren haberlerin oranı: %{finansal_oran:.1f}")

    def _final_kontrol(self):
        print("   🔸 Final kontrol...")

        max_uzunluk = 5000
        onceki = len(self.df[self.df[self.haber_sutun].str.len() > max_uzunluk])

        def metin_kisalt(metin):
            if len(metin) > max_uzunluk:
                return metin[:max_uzunluk] + "..."
            return metin

        self.df[self.haber_sutun] = self.df[self.haber_sutun].apply(metin_kisalt)

        if onceki > 0:
            print(f"      ✓ {onceki} uzun metin kısaltıldı ({max_uzunluk} karakter)")

        self.df = self.df.reset_index(drop=True)
        print(f"      ✓ Index resetlendi")

    def temiz_verileri_kaydet(self, dosya_adi='output/temizlenmis_veriler.xlsx'):
        print(f"\n💾 TEMİZLENMİŞ VERİLER KAYDEDİLİYOR: {dosya_adi}")

        try:
            temiz_df = self.df[[self.haber_sutun, self.dil_sutun, 'finansal_kelime_var']].copy()
            temiz_df.to_excel(dosya_adi, index=False)
            print(f"   ✅ Temizlenmiş veriler başarıyla kaydedildi: {dosya_adi}")
            print(f"   📊 Toplam {len(temiz_df)} temiz kayıt")
            return True
        except Exception as e:
            print(f"   ❌ Kaydetme hatası: {e}")
            return False

    def haberleri_ozetle(self, ozet_uzunlugu=3):
        print("\n📝 3. HABER ÖZETLERİ OLUŞTURULUYOR")
        print("-" * 30)

        def haber_ozetle(metin, dil, max_cumle=3):
            try:
                if len(metin) < 200:
                    return metin

                cumleler = sent_tokenize(metin)

                if len(cumleler) <= max_cumle:
                    return metin

                stop_words = TR_STOP_WORDS if dil == 'tr' else set(stopwords.words('english'))

                kelime_frekanslari = {}
                for cumle in cumleler:
                    for kelime in cumle.lower().split():
                        kelime = kelime.strip('.,!?()[]{}":;')
                        if kelime and kelime not in stop_words:
                            kelime_frekanslari[kelime] = kelime_frekanslari.get(kelime, 0) + 1

                for kelime in FINANSAL_KELIMELER:
                    if kelime in kelime_frekanslari:
                        kelime_frekanslari[kelime] *= 1.5

                cumle_skorlari = {}
                for i, cumle in enumerate(cumleler):
                    for kelime in cumle.lower().split():
                        kelime = kelime.strip('.,!?()[]{}":;')
                        if kelime and kelime not in stop_words:
                            cumle_skorlari[i] = cumle_skorlari.get(i, 0) + kelime_frekanslari.get(kelime, 0)

                if 0 in cumle_skorlari:
                    cumle_skorlari[0] *= 1.5
                if len(cumleler) - 1 in cumle_skorlari:
                    cumle_skorlari[len(cumleler) - 1] *= 1.2

                en_iyi_cumleler = sorted(cumle_skorlari.items(), key=lambda x: x[1], reverse=True)[:max_cumle]
                en_iyi_cumleler = sorted(en_iyi_cumleler, key=lambda x: x[0])
                ozet = ' '.join([cumleler[i] for i, _ in en_iyi_cumleler])
                return ozet

            except Exception as e:
                print(f"      ⚠️  Özet oluşturma hatası: {e}")
                return metin[:200] + "..." if len(metin) > 200 else metin

        self.df['ozet'] = self.df.apply(lambda row: haber_ozetle(row[self.haber_sutun], row[self.dil_sutun], ozet_uzunlugu), axis=1)

        ortalama_ozet_uzunluk = self.df['ozet'].str.len().mean()
        ortalama_haber_uzunluk = self.df[self.haber_sutun].str.len().mean()
        ozet_orani = ortalama_ozet_uzunluk / ortalama_haber_uzunluk * 100 if ortalama_haber_uzunluk else 0

        print(f"   ✅ Özetleme tamamlandı")
        print(f"   📊 Ortalama haber uzunluğu: {ortalama_haber_uzunluk:.0f} karakter")
        print(f"   📊 Ortalama özet uzunluğu: {ortalama_ozet_uzunluk:.0f} karakter")
        print(f"   📊 Ortalama özet oranı: %{ozet_orani:.1f}")

    def duygu_analizi_ayrik(self, row):
        metin = str(row[self.haber_sutun]).lower()
        dil = row[self.dil_sutun]

        def tr_duygu_analizi(metin):
            kelimeler = metin.split()
            poz = sum(1 for k in kelimeler if k in POZITIF_KELIMELER)
            neg = sum(1 for k in kelimeler if k in NEGATIF_KELIMELER)
            if len(kelimeler) == 0:
                return 0
            net = (poz - neg) / len(kelimeler)
            finansal_carpan = 1.5 if any(kelime in metin for kelime in FINANSAL_KELIMELER) else 1.0
            skor = net * finansal_carpan * 3
            return max(-1, min(1, skor))

        def en_duygu_analizi(metin):
            try:
                blob = TextBlob(metin)
                return blob.sentiment.polarity
            except:
                return 0

        skorlar = {}
        for varlik in VARLIKLAR:
            # "bitcoin" var mı diye kontrol etmeye gerek yok, stringde "btc" geçerse skor hesapla
            if varlik == 'btc':
                if 'btc' in metin or 'bitcoin' in metin:
                    if dil == 'tr':
                        skorlar[varlik] = tr_duygu_analizi(metin)
                    else:
                        skorlar[varlik] = en_duygu_analizi(metin)
                else:
                    skorlar[varlik] = 0
            else:
                if varlik in metin:
                    if dil == 'tr':
                        skorlar[varlik] = tr_duygu_analizi(metin)
                    else:
                        skorlar[varlik] = en_duygu_analizi(metin)
                else:
                    skorlar[varlik] = 0
        return skorlar

    def duygu_analizi_yap(self):
        print("\n🎯 4. DUYGU ANALİZİ (VARLIK BAZLI)")
        print("-" * 30)

        varlik_skor_listesi = self.df.apply(self.duygu_analizi_ayrik, axis=1)

        for varlik in VARLIKLAR:
            self.df[f'duygu_{varlik}'] = varlik_skor_listesi.apply(lambda x: x.get(varlik, 0))

        print("   ✅ Varlık bazlı duygu skorları eklendi.")
        print(self.df[[self.haber_sutun] + [f'duygu_{v}' for v in VARLIKLAR]].head())

    def sonuc_dogrulama(self):
        print("\n🔍 5. SONUÇ DOĞRULAMA")
        print("-" * 30)
        # Basit doğrulama örneği
        missing_vals = self.df.isnull().sum()
        if missing_vals.any():
            print("   ⚠️ Eksik değerler mevcut:")
            print(missing_vals[missing_vals > 0])
        else:
            print("   ✅ Eksik değer yok.")

    def detayli_rapor(self):
        print("\n📊 6. DETAYLI RAPOR")
        print("-" * 30)
        print(f"   🔸 Toplam haber: {len(self.df)}")
        print(f"   🔸 Ortalama ozet uzunluğu: {self.df['ozet'].str.len().mean():.1f}")
        for varlik in VARLIKLAR:
            ort_skor = self.df[f'duygu_{varlik}'].mean()
            print(f"   🔸 Ortalama duygu skoru - {varlik}: {ort_skor:.3f}")

    def sonuclari_kaydet(self, cikti_dosya):
        print(f"\n💾 SONUÇLAR KAYDEDİLİYOR: {cikti_dosya}")
        print("-" * 30)
        try:
            ana_sonuc_cols = [self.haber_sutun, 'ozet', 'finansal_kelime_var'] + [f'duygu_{v}' for v in VARLIKLAR]
            ana_sonuc_df = self.df[ana_sonuc_cols].copy()

            diger_sutunlar = [
                'duygu_skoru', 'etiket', 'etiket_aciklama', 'ozet_duygu_skoru',
                'ozet_etiket', 'ozet_etiket_aciklama', 'metin_uzunlugu', 'ozet_uzunlugu',
                'ozet_orani', 'analiz_tarihi'
            ]
            diger_sonuc_df = self.df[[col for col in diger_sutunlar if col in self.df.columns]].copy()

            if 'analiz_tarihi' not in diger_sonuc_df.columns:
                diger_sonuc_df['analiz_tarihi'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            with pd.ExcelWriter(cikti_dosya, engine='openpyxl') as writer:
                ana_sonuc_df.to_excel(writer, sheet_name='Analiz_Sonuclari', index=False)
                diger_sonuc_df.to_excel(writer, sheet_name='Ek_Bilgiler', index=False)

            print("   ✅ Excel dosyası başarıyla kaydedildi")
            print(f"   📋 Sayfalar: Analiz_Sonuclari, Ek_Bilgiler")
            return True
        except Exception as e:
            print(f"   ❌ Kaydetme hatası: {e}")
            return False

    def calistir(self, haber_sutun='haber', dil_sutun='dil', cikti_dosya='analiz_sonuclari.xlsx', ozet_uzunlugu=3):
        print("🚀 KAPSAMLI HABER ANALİZİ BAŞLATIYOR...")
        print("="*60)

        self.haber_sutun = haber_sutun
        self.dil_sutun = dil_sutun

        if not self.veri_yukle(haber_sutun, dil_sutun):
            return False

        self.kapsamli_temizlik()
        self.temiz_verileri_kaydet('output/temiz_haberler.xlsx')
        self.haberleri_ozetle(ozet_uzunlugu=ozet_uzunlugu)
        self.duygu_analizi_yap()
        self.sonuc_dogrulama()
        self.detayli_rapor()
        self.sonuclari_kaydet(cikti_dosya)

        print("\n🎉 ANALİZ TAMAMLANDI!")
        return True

    def ornek_sonuclari_goster(self, n=5):
        print(f"\n👀 ÖRNEK SONUÇLAR (İlk {n} kayıt)")
        print("-" * 80)
        for i in range(min(n, len(self.df))):
            row = self.df.iloc[i]
            print(f"\n📰 Haber {i+1}:")
            print(f"   Metin: {row[self.haber_sutun][:100]}...")
            print(f"   Özet: {row['ozet']}")
            print(f"   Dil: {row[self.dil_sutun]}")
            for varlik in VARLIKLAR:
                print(f"   Duygu {varlik}: {row[f'duygu_{varlik}']:.3f}")
            print("-" * 80)


def main():
    excel_dosya = 'haberler_detayli_lang_tarih1.xlsx'  # Kendi dosyanızı buraya yazın

    print("\n" + "="*70)
    print("📰 HABER ANALİZİ VE ÖZETLEME SİSTEMİ")
    print("="*70)

    analiz = KapsamliHaberAnalizi(excel_dosya)

    if analiz.calistir(
        haber_sutun='content',      # Excelde haber metni sütunu
        dil_sutun='language',       # Excelde dil sütunu
        cikti_dosya='analiz_sonuclari.xlsx',
        ozet_uzunlugu=3
    ):
        analiz.ornek_sonuclari_goster(3)
        print("\n" + "="*70)
        print("🎉 TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
        print("📁 Sonuçlar 'analiz_sonuclari.xlsx' dosyasında kaydedildi")
        print("="*70)
    else:
        print("❌ Analiz başarısız oldu!")


if __name__ == "__main__":
    main()