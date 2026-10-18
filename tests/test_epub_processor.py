"""
tests/test_epub_processor.py
----------------------------
EPUB dosya ayrıştırma, metadata, bölüm tespiti ve API endpoint testleri.
"""

import io
import zipfile
import pytest
from fastapi.testclient import TestClient
from api.main import app
from src.epub_processor import extract_text_from_epub_bytes, get_full_text

client = TestClient(app)


def create_in_memory_epub(
    title: str = "Test EPUB Book",
    author: str = "Test Author",
    ch1_text: str = "Sherlock Holmes sat in his armchair looking out the window onto Baker Street.",
    ch2_text: str = "Dr. Watson walked into the room carrying a medical journal.",
) -> bytes:
    """Birim testler için bellek içinde geçerli bir EPUB dosyası oluşturur."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>""",
        )
        z.writestr(
            "content.opf",
            f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="BookID" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>{title}</dc:title>
    <dc:creator>{author}</dc:creator>
    <dc:identifier id="BookID">test-12345</dc:identifier>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="ch1"/>
    <itemref idref="ch2"/>
  </spine>
</package>""",
        )
        z.writestr(
            "toc.ncx",
            f"""<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:uid" content="test-12345"/></head>
  <docTitle><text>{title}</text></docTitle>
  <navMap>
    <navPoint id="np-1" playOrder="1">
      <navLabel><text>Chapter 1: The Investigation</text></navLabel>
      <content src="ch1.xhtml"/>
    </navPoint>
    <navPoint id="np-2" playOrder="2">
      <navLabel><text>Chapter 2: The Conclusion</text></navLabel>
      <content src="ch2.xhtml"/>
    </navPoint>
  </navMap>
</ncx>""",
        )
        z.writestr(
            "ch1.xhtml",
            f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 1</title></head>
<body>
<h1>Chapter 1: The Investigation</h1>
<p>{ch1_text}</p>
</body>
</html>""",
        )
        z.writestr(
            "ch2.xhtml",
            f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 2</title></head>
<body>
<h1>Chapter 2: The Conclusion</h1>
<p>{ch2_text}</p>
</body>
</html>""",
        )
    return buf.getvalue()


class TestEpubProcessor:
    def test_extract_text_and_metadata_from_epub_bytes(self):
        epub_bytes = create_in_memory_epub()
        data = extract_text_from_epub_bytes(epub_bytes, "test.epub")

        assert data["title"] == "Test EPUB Book"
        assert data["author"] == "Test Author"
        assert data["total_pages"] >= 2
        assert len(data["pages"]) >= 2
        assert len(data["chapters"]) == 2

        # Bölüm başlıkları
        assert "Chapter 1: The Investigation" in data["chapters"][0]["title"]
        assert "Chapter 2: The Conclusion" in data["chapters"][1]["title"]

        # Sayfa içeriği ve metadata eşlemesi
        p1 = data["pages"][0]
        assert "Sherlock Holmes" in p1["text"]
        assert p1.get("source_chapter") == "Chapter 1: The Investigation"

        # Tam metin
        full_text = get_full_text(data["pages"])
        assert "Sherlock Holmes" in full_text
        assert "Dr. Watson" in full_text

    def test_extract_corrupted_epub_raises(self):
        with pytest.raises(Exception):
            extract_text_from_epub_bytes(b"not an epub file content", "corrupted.epub")


class TestEpubApiEndpoints:
    def test_generate_from_epub_invalid_extension(self):
        files = {"file": ("test.txt", b"plain text", "text/plain")}
        resp = client.post("/api/v1/quiz/generate-from-epub", files=files)
        assert resp.status_code == 400
        assert "Yalnızca .epub" in resp.json()["detail"]

    def test_generate_from_epub_missing_file(self):
        resp = client.post("/api/v1/quiz/generate-from-epub")
        assert resp.status_code == 422
