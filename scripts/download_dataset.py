#!/usr/bin/env python3
"""
scripts/download_dataset.py
---------------------------
Project Gutenberg kitap kataloğunu indirir ve SQLite FTS5 veritabanına
kaydeder. Arama işlemlerinin tamamen yerel, anlık ve çevrimdışı çalışmasını sağlar.

Kullanım:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --force
    python scripts/download_dataset.py --download-popular 10
"""

import argparse
import csv
import io
import logging
import os
import sqlite3
import sys
import time
import urllib.request

# Proje kök dizinini ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BOOKS_DIR, CATALOG_DB_PATH, DATASET_CACHE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_dataset")

PRIMARY_CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
FALLBACK_CATALOG_URL = "https://huggingface.co/datasets/Fhrozen/gutenberg8k/resolve/main/gutenberg_metadata.csv"

# En çok okunan klasik kitapların Gutenberg ID'leri (isteğe bağlı önden indirme için)
POPULAR_BOOK_IDS = [
    ("1342", "Pride and Prejudice", "Jane Austen"),
    ("84", "Frankenstein; Or, The Modern Prometheus", "Mary Wollstonecraft Shelley"),
    ("1661", "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    ("345", "Dracula", "Bram Stoker"),
    ("11", "Alice's Adventures in Wonderland", "Lewis Carroll"),
    ("2701", "Moby Dick; Or, The Whale", "Herman Melville"),
    ("98", "A Tale of Two Cities", "Charles Dickens"),
    ("174", "The Picture of Dorian Gray", "Oscar Wilde"),
    ("76", "Adventures of Huckleberry Finn", "Mark Twain"),
    ("2591", "Grimms' Fairy Tales", "Jacob Grimm and Wilhelm Grimm"),
    ("1232", "The Prince", "Niccolò Machiavelli"),
    ("28054", "The Brothers Karamazov", "Fyodor Dostoyevsky"),
    ("2600", "War and Peace", "Leo Tolstoy"),
    ("43", "The Strange Case of Dr. Jekyll and Mr. Hyde", "Robert Louis Stevenson"),
    ("1952", "The Yellow Wallpaper", "Charlotte Perkins Gilman"),
]


def download_catalog_content(progress_callback=None) -> bytes:
    """Katalog CSV dosyasını indirir (önce Gutenberg, gerekirse HF fallback)."""
    headers = {"User-Agent": "Mozilla/5.0 (ContextRAG/1.0; contact@example.com)"}

    urls = [
        ("Gutenberg Resmi Katalogu (~20 MB, 75k+ kitap)", PRIMARY_CATALOG_URL),
        ("HuggingFace Yedek Katalog (~1.5 MB, 15k kitap)", FALLBACK_CATALOG_URL),
    ]

    for label, url in urls:
        logger.info(f"Katalog indiriliyor: {label}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as resp:
                total_size = int(resp.headers.get("content-length", 0))
                downloaded = 0
                chunks = []

                while True:
                    chunk = resp.read(1024 * 512)  # 512KB chunks
                    if not chunk:
                        break
                    chunks.append(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size or 21253758, f"{label} indiriliyor...")

                content = b"".join(chunks)
                logger.info(f"Katalog başarıyla indirildi: {len(content) / (1024 * 1024):.2f} MB")
                return content
        except Exception as e:
            logger.warning(f"URL başarısız ({url}): {e}")

    raise RuntimeError("Katalog dosyası indirilemedi. Lütfen internet bağlantınızı kontrol edin.")


def build_catalog_database(csv_bytes: bytes, db_path: str = CATALOG_DB_PATH) -> int:
    """CSV verisini okur ve SQLite FTS5 veritabanını oluşturur."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Geçici dosyaya yazıp işlem bitince atomik değiştirme yapalım
    temp_db_path = db_path + ".tmp"
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)

    conn = sqlite3.connect(temp_db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS books")
    cur.execute("DROP TABLE IF EXISTS books_fts")

    cur.execute(
        """
        CREATE TABLE books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id TEXT UNIQUE,
            title TEXT,
            author TEXT,
            subjects TEXT,
            bookshelves TEXT,
            language TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE VIRTUAL TABLE books_fts USING fts5(
            title,
            author,
            subjects,
            bookshelves,
            content='books',
            content_rowid='id'
        )
        """
    )

    logger.info("CSV ayrıştırılıyor ve veritabanı indeksleniyor...")
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8", errors="ignore")))

    batch = []
    count = 0
    seen_ids = set()

    for row in reader:
        # Gutenberg catalog vs HF format uyumluluğu
        doc_type = row.get("Type", "Text")
        if doc_type != "Text":
            continue  # Sesli kitapları ve diğer formatları atla

        book_id = (row.get("Text#") or row.get("Link", "").split("/")[-1]).strip()
        title = (row.get("Title") or "").strip()
        author = (row.get("Authors") or row.get("Author") or "").strip()
        subjects = (row.get("Subjects") or "").strip()
        bookshelves = (row.get("Bookshelves") or row.get("Bookshelf") or "").strip()
        lang = (row.get("Language") or "en").strip()

        if not title or not book_id or book_id in seen_ids:
            continue

        seen_ids.add(book_id)
        batch.append((book_id, title, author, subjects, bookshelves, lang))
        count += 1

        if len(batch) >= 5000:
            cur.executemany(
                "INSERT INTO books (book_id, title, author, subjects, bookshelves, language) VALUES (?, ?, ?, ?, ?, ?)",
                batch,
            )
            batch = []

    if batch:
        cur.executemany(
            "INSERT INTO books (book_id, title, author, subjects, bookshelves, language) VALUES (?, ?, ?, ?, ?, ?)",
            batch,
        )

    logger.info("FTS arama indeksi optimize ediliyor...")
    cur.execute("INSERT INTO books_fts(books_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()

    # Atomik taşıma
    if os.path.exists(db_path):
        os.remove(db_path)
    os.rename(temp_db_path, db_path)

    logger.info(f"✅ Başarılı! {count:,} kitap indekslendi. Boyut: {os.path.getsize(db_path)/(1024*1024):.2f} MB")
    return count


def download_book_text(book_id: str, title: str = "", author: str = "") -> str:
    """Tek bir kitabın metnini yerel disk cache'ine kaydeder."""
    os.makedirs(BOOKS_DIR, exist_ok=True)
    local_path = os.path.join(BOOKS_DIR, f"{book_id}.txt")

    if os.path.isfile(local_path) and os.path.getsize(local_path) > 100:
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    headers = {"User-Agent": "Mozilla/5.0 (ContextRAG/1.0)"}
    urls = [
        f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
    ]

    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                text = resp.read().decode("utf-8", errors="ignore")
                if len(text) > 200:
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    logger.info(f"Kitap indirildi ve kaydedildi: [{book_id}] {title} ({len(text):,} karakter)")
                    return text
        except Exception:
            continue

    raise FileNotFoundError(f"Kitap metni indirilemedi (ID: {book_id})")


def download_popular_books(count: int = 10):
    """En popüler klasik kitapları önden diske indirir."""
    logger.info(f"{count} adet popüler klasik kitap diske önceden indiriliyor...")
    success = 0
    for b_id, title, author in POPULAR_BOOK_IDS[:count]:
        try:
            download_book_text(b_id, title, author)
            success += 1
            time.sleep(0.5)  # Gutenberg sunucusuna saygılı hız
        except Exception as e:
            logger.warning(f"Kitap indirilemedi [{b_id}] {title}: {e}")
    logger.info(f"✅ {success}/{count} popüler kitap yerel hafızaya (`data/books/`) kaydedildi.")


def main():
    parser = argparse.ArgumentParser(description="Gutenberg kitap kataloğunu önden indirip indeksler.")
    parser.add_argument("--force", action="store_true", help="Mevcut indeks varsa bile sıfırdan yeniden indir ve kur.")
    parser.add_argument("--download-popular", type=int, default=0, help="N adet popüler kitabı da önden metin olarak indir.")
    args = parser.parse_args()

    if os.path.exists(CATALOG_DB_PATH) and not args.force:
        logger.info(f"Katalog veritabanı zaten mevcut: {CATALOG_DB_PATH}")
        with sqlite3.connect(CATALOG_DB_PATH) as conn:
            cnt = conn.cursor().execute("SELECT count(*) FROM books").fetchone()[0]
        logger.info(f"Toplam kayıtlı kitap: {cnt:,}")
    else:
        logger.info("Katalog kurulumu başlatılıyor...")
        t0 = time.time()
        csv_data = download_catalog_content()
        total = build_catalog_database(csv_data)
        logger.info(f"Kurulum {time.time()-t0:.1f} saniyede tamamlandı ({total:,} kitap).")

    if args.download_popular > 0:
        download_popular_books(args.download_popular)

    logger.info("\n🎉 Sistem hazır! Uygulamayı açıp anında arama yapabilirsiniz:")
    logger.info("   streamlit run ui/app.py")


if __name__ == "__main__":
    main()
