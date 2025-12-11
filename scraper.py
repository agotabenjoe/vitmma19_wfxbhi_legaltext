from bs4 import BeautifulSoup
import csv

def extract_article_paragraphs(html, article_id="post-68"):
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article", id=article_id) or soup.select_one(f"article.{article_id}")
    paragraphs = []
    if article:
        for p in article.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            if text:
                paragraphs.append(text)
    return paragraphs

def main(html_path, csv_path):
    with open(html_path, encoding="utf-8") as f:
        html = f.read()
    paragraphs = extract_article_paragraphs(html)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        # BOM az elejére az ékezetek miatt (Excel)
        f.write('\ufeff')
        writer = csv.writer(f)
        writer.writerow(["paragraph", "ease_of_understanding"])
        for p in paragraphs:
            writer.writerow([p, ""])

if __name__ == "__main__":
    # Fix: nem általános, fix elérési utak
    html_path = "data_collection/kepregeny_depo.html"
    csv_path = "data_collection/output.csv"
    main(html_path, csv_path)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
