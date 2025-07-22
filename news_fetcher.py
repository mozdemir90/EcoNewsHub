import requests
from bs4 import BeautifulSoup
import pandas as pd
from langdetect import detect
import time
import re

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
}

def get_bloomberght_articles():
    BASE_URL = "https://www.bloomberght.com"
    print("Bloomberght haberleri çekiliyor...")
    try:
        response = requests.get(BASE_URL, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        articles = []
        for a_tag in soup.find_all("a", class_="md:min-h-16 inline-flex items-center"):
            title_tag = a_tag.find("span", class_="line-clamp-2")
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            relative_link = a_tag.get("href")
            full_link = relative_link if relative_link.startswith("http") else BASE_URL + relative_link

            content, date = get_bloomberght_detail(full_link)
            lang = detect_language(content)
            articles.append({
                "source": "Bloomberght",
                "title": title,
                "url": full_link,
                "date": date,
                "content": content,
                "language": lang
            })
            time.sleep(1)

        return articles
    except Exception as e:
        print("Error - Bloomberght:", e)
        return []

def get_bloomberght_detail(url):
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        div = soup.find("div", class_="article-wrapper mb-4 mt-5")
        paragraphs = div.find_all("p") if div else []
        content = " ".join(p.get_text(strip=True) for p in paragraphs)

        date_text = ""
        info_div = soup.find("div", attrs={"data-type": "news-info"})
        if info_div:
            text_xs_div = info_div.find("div", class_="text-xs")
            if text_xs_div:
                for inner_div in text_xs_div.find_all("div"):
                    if "Giriş:" in inner_div.text:
                        date_text = inner_div.text.replace("Giriş:", "").strip()
                        break
        return content, date_text
    except Exception as e:
        print(f"Bloomberght detail error: {url} - {e}")
        return "", ""

def get_cnn_articles():
    BASE_URL = "https://edition.cnn.com"
    URL = "https://edition.cnn.com/business"
    print("CNN Business haberleri çekiliyor...")
    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        articles = []
        for a_tag in soup.find_all("a", class_="container__link container__link--type-article container_lead-plus-headlines__link"):
            title_tag = a_tag.find("span", class_="container__headline-text")
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            relative_link = a_tag.get("href")
            full_link = relative_link if relative_link.startswith("http") else BASE_URL + relative_link

            content, date = get_cnn_detail(full_link)
            lang = detect_language(content)
            articles.append({
                "source": "CNN Business",
                "title": title,
                "url": full_link,
                "date": date,
                "content": content,
                "language": lang
            })
            time.sleep(1)

        return articles
    except Exception as e:
        print("Error - CNN:", e)
        return []

def get_cnn_detail(url):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        article_body = soup.find("div", class_="article__content") or soup.find("div", attrs={"data-zn-id": "body-text"})
        paragraphs = article_body.find_all("p") if article_body else []
        content = " ".join(p.get_text(strip=True) for p in paragraphs)

        date_meta = soup.find("meta", {"itemprop": "datePublished"})
        if date_meta and date_meta.get("content"):
            date = date_meta["content"]
        else:
            match = re.search(r"/(\d{4}/\d{2}/\d{2})/", url)
            date = match.group(1) if match else ""

        return content, date
    except Exception as e:
        print(f"Error parsing CNN detail: {url} - {e}")
        return "", ""

def get_hurriyet_articles():
    BASE_URL = "https://www.hurriyetdailynews.com"
    URL = BASE_URL + "/economy/"
    print("Hürriyet Daily News (Economy) çekiliyor...")

    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        articles = []
        modules = soup.find_all("div", class_="module news-single-complete")

        for module in modules:
            a_tag = module.find("a", href=True)
            if not a_tag:
                continue

            link = a_tag["href"]
            full_url = link if link.startswith("http") else BASE_URL + link

            title_tag = a_tag.find("h3")
            title = title_tag.get_text(strip=True) if title_tag else ""

            content, date = get_hurriyet_detail(full_url)
            lang = detect_language(content)
            articles.append({
                "source": "Hürriyet Daily News",
                "title": title,
                "url": full_url,
                "date": date,
                "content": content,
                "language": lang
            })
            time.sleep(1)

        return articles
    except Exception as e:
        print("Error - Hürriyet Daily News:", e)
        return []

def get_hurriyet_detail(url):
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        content_div = soup.find("div", class_="content")
        paragraphs = content_div.find_all("p") if content_div else []
        content = " ".join(p.get_text(strip=True) for p in paragraphs)

        time_tag = soup.find("time")
        date = time_tag["datetime"] if time_tag and time_tag.has_attr("datetime") else ""

        return content, date
    except Exception as e:
        print(f"Hürriyet detail error ({url}):", e)
        return "", ""

def get_bbc_news():
    url = "https://www.bbc.com/business"
    base_url = "https://www.bbc.com"
    print("BBC News çekiliyor...")
    try:
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.content, "html.parser")
        links = [a['href'] for a in soup.find_all("a", href=True) if a['href'].startswith("/news/articles/")]

        data = []
        for link in links:
            news_url = base_url + link
            try:
                news_r = requests.get(news_url, headers=headers)
                news_soup = BeautifulSoup(news_r.content, "html.parser")
                
                title_tag = news_soup.find("h1")
                title = title_tag.text.strip() if title_tag else ""

                date_tag = news_soup.find("time")
                date = date_tag["datetime"] if date_tag and date_tag.has_attr("datetime") else ""

                paragraphs = news_soup.find_all("p")
                content = " ".join([p.text.strip() for p in paragraphs])
                lang = detect_language(content)
                data.append({
                    "source": "BBC",
                    "url": news_url,
                    "title": title,
                    "content": content,
                    "date": date,
                    "language": lang
                })
            except Exception as e:
                print(f"Error parsing {news_url}: {e}")
            time.sleep(1)

        return data
    except Exception as e:
        print("Error - BBC:", e)
        return []

def get_anadolu_ajansi():
    url = "https://www.aa.com.tr/tr/ekonomi"
    print("Anadolu Ajansı haberleri çekiliyor...")
    try:
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.content, "html.parser")

        article_blocks = soup.find_all("div", class_="row konu-ust-icerik container p-0") + \
                         soup.find_all("div", class_="row konu-ust-mansetalti konu-ust-icerik container p-0")

        links = []
        for block in article_blocks:
            anchors = block.find_all("a", href=True)
            for a in anchors:
                href = a["href"]
                if href.startswith("https://www.aa.com.tr/tr/ekonomi/") and href.count("/") > 5:
                    links.append(href)

        data = []
        for news_url in links:
            try:
                news_r = requests.get(news_url, headers=headers)
                news_soup = BeautifulSoup(news_r.content, "html.parser")

                title_tag = news_soup.find("div", class_="detay-spot-category").find("h1")
                title = title_tag.text.strip() if title_tag else ""

                date_tag = news_soup.find("span", class_="tarih")
                date = date_tag.get_text(strip=True) if date_tag else ""

                content_div = news_soup.find("div", class_="detay-icerik")
                paragraphs = content_div.find_all("p") if content_div else []
                content = " ".join(p.get_text(strip=True) for p in paragraphs)
                lang = detect_language(content)
                if title and content:
                    data.append({
                        "source": "Anadolu Ajansı",
                        "url": news_url,
                        "title": title,
                        "content": content,
                        "date": date,
                        "language": lang
                    })

            except Exception as e:
                print(f"Error parsing {news_url}: {e}")
            time.sleep(1)

        return data
    except Exception as e:
        print("Error - Anadolu Ajansı:", e)
        return []

def get_additional_haberler_articles():
    base_url = "https://www.haberler.com"
    url = base_url + "/ekonomi/"
    print("Ekstra Haberler.com haberleri çekiliyor...")

    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        
        soup = BeautifulSoup(r.content, "html.parser")

        articles = []
        card_container = soup.find("div", class_="new3card-container")
        if card_container:
            for card in card_container.find_all("div", class_="new3card"):
                a_tag = card.find("a", href=True)
                if not a_tag:
                    continue
                
                href = a_tag["href"]
                full_url = base_url + href if href.startswith("/") else href
                title = a_tag.get("title", "").strip()
                
                date_div = card.find("div", class_="hbbiText")
                date = date_div.get_text(strip=True) if date_div else "Tarih bulunamadı"
                lang = detect_language(title)
                content = fetch_haberler_content(full_url)
                if content:
                    articles.append({
                    "source": "Haberler.com",
                    "url": full_url,
                    "title": title,
                    "content": content,
                    "date": date,
                    "language": lang
                })
                        
        return articles
    except Exception as e:
        print(f"Haberler.com ana sayfa hatası: {e}")
        return []

def fetch_haberler_content(news_url):
    try:
        news_r = requests.get(news_url, headers=headers)
        news_r.raise_for_status()
        news_soup = BeautifulSoup(news_r.content, "html.parser")

        content_div = news_soup.find("div", id="news")
        paragraphs = content_div.find_all("p") if content_div else []
        content = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        return content  # Sadece metin döndürülüyor, dict değil
    except Exception as e:
        print(f"Error fetching content from {news_url}: {e}")
        return "Content not found"

def get_ntv_para_articles():
    base_url = "https://www.ntv.com.tr"
    url = base_url + "/ntvpara"
    print("NTV Para haberleri çekiliyor...")

    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()

        soup = BeautifulSoup(r.content, "html.parser")

        articles = []
        # Look for the bottom frame containing additional articles
        bottom_frame = soup.find("div", class_="ntvpara-bottom-frame")
        if bottom_frame:
            # Iterate through multiple items
            for item in bottom_frame.find_all("div", class_="category-multiple-item"):
                card = item.find("div", class_="card")
                if not card:
                    continue

                a_tag = card.find("a", href=True)
                href = a_tag["href"]
                full_url = base_url + href if href.startswith("/") else href
                title = a_tag.get("title", "").strip()
                lang = detect_language(title)
                content = fetch_article_content(full_url)

                # Find date inside the individual news page
                news_r = requests.get(full_url, headers=headers)
                news_r.raise_for_status()
                news_soup = BeautifulSoup(news_r.content, "html.parser")
                
                # Try to find date in both left and right div
                date = "Tarih bulunamadı"
                
                # First try left div
                date_div_left = news_soup.find("div", class_="news-info-text--left")
                if date_div_left:
                    date = date_div_left.get_text(strip=True)
                else:
                    # If not found in left, try right div
                    date_div_right = news_soup.find("div", class_="news-info-text--right")
                    if date_div_right:
                        date = date_div_right.get_text(strip=True)
                    else:
                        # If neither found, try more generic selectors
                        date_divs = news_soup.find_all("div", class_=lambda x: x and "news-info" in x)
                        for div in date_divs:
                            text = div.get_text(strip=True)
                            # Check if the text contains date-like patterns
                            if any(keyword in text.lower() for keyword in ["dakika", "saat", "gün", "ay", "yıl", "bugün", "dün", ":", "/"]):
                                date = text
                                break

                if content:
                    articles.append({
                        "source": "NTV Para",
                        "url": full_url,
                        "title": title,
                        "content": content,
                        "date": date,
                        "language": lang
                    })

        return articles

    except Exception as e:
        print(f"NTV ana sayfa hatası: {e}")
        return []

def fetch_article_content(news_url):
    try:
        news_r = requests.get(news_url, headers=headers)
        news_r.raise_for_status()

        news_soup = BeautifulSoup(news_r.content, "html.parser")

        content_div = news_soup.find("div", class_="content-news-tag-selector")
        paragraphs = content_div.find_all("p") if content_div else []
        content = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        return content if content else "İçerik bulunamadı"
    except Exception as e:
        return "İçerik bulunamadı"

def fetch_dunya_news():
    print("Dünya Gazetesi haberleri çekiliyor...")
    url = "https://www.dunya.com"
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}
    articles = []

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Manşet slider'dan haberler
        slider_items = soup.select("div#headline a.swiper-slide")
        for item in slider_items:
            title = item.get("title", "").strip()
            link = item.get("href", "")
            if title and link:
                detail = fetch_dunya_detail(link)
                articles.append({
                    "title": title,
                    "url": link,
                    "content": detail["content"],
                    "date": detail["published_time"],
                    "language": detail["language"],
                    "source": "dunya.com"
                })

        # Ek olarak diğer haberleri de alabiliriz (örnek)
        wrapper_items = soup.select("div.middle-side_wrapper a")
        for item in wrapper_items:
            title = item.get("title", "").strip()
            link = item.get("href", "")
            if title and link and link.startswith("https://www.dunya.com") and not any(a["url"] == link for a in articles):
                detail = fetch_dunya_detail(link)
                articles.append({
                    "title": title,
                    "url": link,
                    "content": detail["content"],
                    "date": detail["published_time"],
                    "language": detail["language"],
                    "source": "dunya.com"
                })

    except Exception as e:
        print(f"[HATA] dunya.com ana sayfa hatası: {e}")

    return articles

# İlk tanım (bu daha sade ve eksiksiz)
def fetch_dunya_detail(url):

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # İçerik
        content_div = soup.select_one("div.content-text")
        paragraphs = []
        if content_div:
            for tag in content_div.find_all(["p", "h2"]):
                text = tag.get_text(strip=True)
                if text and "adpro" not in tag.get("class", []):
                    paragraphs.append(text)
        content = "\n".join(paragraphs)

        # Yayın zamanı
        time_tag = soup.select_one("div.item-date time")
        published_time = time_tag.get("datetime", "").strip() if time_tag else ""

        # Dil
        language = detect(content) if content else "unknown"

        return {
            "content": content,
            "published_time": published_time,
            "language": language
        }

    except Exception as e:
        print(f"[HATA] dunya.com detay sayfası işlenemedi: {url} -> {e}")
        return {
            "content": "",
            "published_time": "",
            "language": "unknown"
        }


def fetch_cnbc_content(news_url):
    try:
        news_r = requests.get(news_url, headers=headers)
        news_r.raise_for_status()
        
        news_soup = BeautifulSoup(news_r.content, "html.parser")

        # Extract the main article body
        article_body_div = news_soup.find("div", class_="ArticleBody-articleBody")
        paragraphs = article_body_div.find_all("p") if article_body_div else []
        content = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        return {
            "content": content
        }
        
    except Exception as e:
        print(f"Error fetching content from {news_url}: {e}")
        return {
            "content": "No content found"
        }

def get_cnbc_articles():
    home_url = "https://www.cnbc.com/world/?region=world"
    processed_urls = set()

    try:
        response = requests.get(home_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        lang= detect_language(response.text)
        articles = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.startswith("http") and "/2025/" in href and href not in processed_urls:
                full_url = href
                title = a_tag.get_text(strip=True)
                time_tag = soup.select_one("time[data-testid='published-timestamp']")
                published_time = time_tag.get("datetime", "") if time_tag else "Zaman bulunamadı"
                article_data = fetch_cnbc_content(full_url)
                articles.append({
                    "source": "CNBC",
                    "url": full_url,
                    "title": title,
                    "content": article_data["content"],
                    "date": published_time,  # Adjust date extraction if needed
                    "language": lang
                })
                processed_urls.add(href)  # Mark as processed
                time.sleep(1)  # Be polite to the server

        return articles
    except Exception as e:
        print(f"Error fetching CNBC homepage: {e}")
        return []

# Gather and save all articles
if __name__ == "__main__":
    all_articles = []
    all_articles.extend(get_bloomberght_articles())
    all_articles.extend(get_cnn_articles())
    all_articles.extend(get_hurriyet_articles())
    all_articles.extend(get_bbc_news())
    all_articles.extend(get_anadolu_ajansi())
    all_articles.extend(get_ntv_para_articles())
    all_articles.extend(get_additional_haberler_articles())
    all_articles.extend(fetch_dunya_news())
    all_articles.extend(get_cnbc_articles())

    df = pd.DataFrame(all_articles)
    df.to_excel("haberler_detayli_lang_tarih.xlsx", index=False)
    print("Tüm haberler 'haberler_detayli_lang_tarih.xlsx' dosyasına kaydedildi.")