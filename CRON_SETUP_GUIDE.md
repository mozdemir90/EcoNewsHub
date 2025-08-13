# Cron Job Kurulum Rehberi
==================================================

## 🎯 Cron News Fetcher Nedir?

`cron_news_fetcher.py` scripti, Telegram bot'unu sürekli çalıştırmak yerine, belirli aralıklarla tek seferlik haber kontrolü yapar.

## 📋 Avantajları

✅ **Kaynak Tasarrufu**: Sürekli çalışan bot yerine ihtiyaç anında çalışır
✅ **Güvenilirlik**: Her çalışmada temiz başlangıç
✅ **Kontrol**: Ne zaman çalışacağını tam kontrol edebilirsiniz
✅ **Log Yönetimi**: Her çalışma için ayrı log

## 🔧 Kurulum Adımları

### 1. Cron Job Ekleme

```bash
# Cron job editörünü aç
crontab -e

# Aşağıdaki satırlardan birini ekle:
```

### 2. Farklı Zamanlama Seçenekleri

**Her 30 dakikada bir:**
```bash
*/30 * * * * cd /Users/mozdemir/AiTrain/newsFetch && python3 cron_news_fetcher.py
```

**Her saat başı:**
```bash
0 * * * * cd /Users/mozdemir/AiTrain/newsFetch && python3 cron_news_fetcher.py
```

**Her 2 saatte bir:**
```bash
0 */2 * * * cd /Users/mozdemir/AiTrain/newsFetch && python3 cron_news_fetcher.py
```

**Sadece iş günleri (Pazartesi-Cuma) saat 9-18 arası:**
```bash
0 9-18 * * 1-5 cd /Users/mozdemir/AiTrain/newsFetch && python3 cron_news_fetcher.py
```

**Günde 3 kez (8:00, 14:00, 20:00):**
```bash
0 8,14,20 * * * cd /Users/mozdemir/AiTrain/newsFetch && python3 cron_news_fetcher.py
```

### 3. Cron Job Yönetimi

**Mevcut cron job'ları görüntüle:**
```bash
crontab -l
```

**Cron job'ları düzenle:**
```bash
crontab -e
```

**Tüm cron job'ları sil:**
```bash
crontab -r
```

## 📊 Log Takibi

**Cron job loglarını görüntüle:**
```bash
tail -f logs/cron_news.log
```

**Telegram bot loglarını görüntüle:**
```bash
tail -f logs/telegram_bot.log
```

**Son 50 satır log:**
```bash
tail -50 logs/cron_news.log
```

## 🚀 Test Etme

**Manuel test:**
```bash
python3 cron_news_fetcher.py
```

**Cron job'ın çalışıp çalışmadığını kontrol et:**
```bash
# Sistem cron loglarını kontrol et
sudo tail -f /var/log/cron

# macOS için
sudo log show --predicate 'process == "cron"' --last 1h
```

## ⚠️ Önemli Notlar

1. **Tam Yol Kullanın**: Script'in tam yolunu belirtin
2. **Çalışma Dizini**: `cd` komutu ile doğru dizine geçin
3. **Python Yolu**: `python3` kullanın
4. **Log Dosyaları**: `logs/` klasöründe saklanır
5. **Hata Kontrolü**: Log dosyalarını düzenli kontrol edin

## 🔍 Sorun Giderme

**Cron job çalışmıyor:**
```bash
# Cron servisini kontrol et
sudo systemctl status cron

# macOS için
sudo launchctl list | grep cron
```

**Log dosyası oluşmuyor:**
```bash
# Dizin izinlerini kontrol et
ls -la logs/

# Manuel test et
python3 cron_news_fetcher.py
```

**Bot token hatası:**
```bash
# bot_config.json dosyasını kontrol et
cat bot_config.json
```

## 📱 Telegram Bildirimleri

Cron job çalıştığında:
- Yeni haberler varsa Telegram'a gönderilir
- Yeni haber yoksa log'a yazılır, mesaj gönderilmez
- Her çalışma `logs/cron_news.log` dosyasına kaydedilir

## 🎯 Önerilen Zamanlama

**Geliştirme/Test için:**
```bash
*/15 * * * * cd /Users/mozdemir/AiTrain/newsFetch && python3 cron_news_fetcher.py
```

**Üretim için:**
```bash
0 */2 * * * cd /Users/mozdemir/AiTrain/newsFetch && python3 cron_news_fetcher.py
```

**Yoğun haber dönemleri için:**
```bash
*/30 * * * * cd /Users/mozdemir/AiTrain/newsFetch && python3 cron_news_fetcher.py
```
