from madde.ingest import parse as P

MADDE_PAGES = [
    "PLANLI ALANLAR İMAR YÖNETMELİĞİ\nBİRİNCİ BÖLÜM\nAmaç\n"
    "MADDE 1 - (1) Bu Yönetmeliğin amacı; plan şartlarını belirlemektir.\n"
    "Kapsam\nMADDE 2 – (1) Bu Yönetmelik, imar planı bulunan alanları kapsar.\n2",
    "Tanımlar\nMADDE 3 - (1) Bu Yönetmelikte geçen;\n"
    + "".join(f"{b}) Terim {b}: " + "uzun tanım metni " * 40 + ",\n" for b in "abc")
    + "(2) İkinci fıkra.\nGEÇİCİ MADDE 1 - (1) Geçiş hükmü.",
]


def test_madde_articles_labels_pages_and_headings():
    chunks = P.parse_madde(MADDE_PAGES)
    by = {c.article: c for c in chunks}
    assert by["1"].heading == "Amaç"
    assert by["1"].body.startswith("MADDE 1 -")
    assert "(1) Bu Yönetmeliğin amacı" in by["1"].body
    assert by["2"].page == 1
    assert by["Geçici 1"].page == 2


def test_long_article_splits_into_fikra_and_bent():
    labels = [c.article for c in P.parse_madde(MADDE_PAGES)]
    assert "3(1)(a)" in labels and "3(1)(c)" in labels
    assert "3(2)" in labels
    assert len(labels) == len(set(labels))


def test_page_numbers_and_toc_lines_are_dropped():
    pages = [
        "İÇİNDEKİLER\n1.1. KAPSAM ........................ 1\n7\n"
        "1.1. KAPSAM\n1.1.1 – Bu Yönetmelik binaları kapsar.\n"
    ]
    chunks = P.parse_numbered(pages)
    assert [c.article for c in chunks] == ["1.1.1"]
    assert chunks[0].heading == "KAPSAM"
    assert chunks[0].body == "Bu Yönetmelik binaları kapsar."


def test_numbered_rejects_table_numbers_and_backwards_jumps():
    pages = [
        "2.3. TASARIM SPEKTRUMU\n2.3.1 – Spektrum tanımı.\n0.4 0.8 1.2\n1.5 tablo hücresi\n"
        "2.3.2 – İkinci kural.\n5.1 – ileri atlama, tablo değeri\n3.1. YENİ BÖLÜM\n3.1.1 – Kural.",
    ]
    chunks = P.parse_numbered(pages)
    assert [c.article for c in chunks] == ["2.3.1", "2.3.2", "3.1.1"]
    assert "1.5 tablo hücresi" in chunks[0].body
    assert "5.1" in chunks[1].body
    assert chunks[2].heading == "YENİ BÖLÜM"


def test_numbered_ignores_inline_references_and_table_rows():
    pages = [
        "16.7. KAZIKLAR\n16.7.1 – Kazık kuralı.\n16.8’de verilen esaslar uyarınca.\n"
        "12.0 18.0 22.6\n16.8. YÜZEYSEL TEMELLER\n16.8.1 – Temel kuralı."
    ]
    labels = [c.article for c in P.parse_numbered(pages)]
    assert labels == ["16.7.1", "16.8.1"]


def test_toc_page_is_skipped_even_without_dot_leaders():
    toc = "İÇİNDEKİLER\n" + "".join(f"5.{i}. BAŞLIK {i} ............ {i}\n" for i in range(1, 7))
    toc += "5.8.3. Uzun başlık satırı ikinci satıra taşan\n"
    body = "1.1. KAPSAM\n1.1.1 – Kural."
    assert [c.article for c in P.parse_numbered([toc, body])] == ["1.1.1"]


def test_annexes_and_part_headers_in_madde_layout():
    pages = [
        "MADDE 1 - (1) Kural.\nİKİNCİ BÖLÜM\nMADDE 2 - (1) Son madde.\n"
        "EK-1 Yangın dayanım tablosu\n(1) tablo notu\n(1) tekrar eden numara\n"
        "EK-1’de belirtilen değerler"
    ]
    chunks = P.parse_madde(pages)
    assert [c.article for c in chunks] == ["1", "2", "Ek-1"]
    assert "BÖLÜM" not in chunks[0].body
    assert chunks[2].heading == "Yangın dayanım tablosu"
    assert "EK-1’de belirtilen" in chunks[2].body


def test_numbered_annexes_follow_their_chapter():
    pages = [
        "5.9. SON\n5.9.1 – Kural.\nEK 5A – MODELLER\n5A.1. BETON\n5A.1.1 – Model.\n"
        "6.1. YENİ\n6.1.1 – Kural."
    ]
    assert [c.article for c in P.parse_numbered(pages)] == ["5.9.1", "5A.1.1", "6.1.1"]


def test_fire_code_annex_formats():
    pages = [
        "MADDE 171 - (1) Son.\nEk-1/B, Ek-2/Ç, Ek-3/C değiştirildi\nEk-1\n"
        "Ek-1/A  Düşük Tehlike Kullanım Alanları\nsatır\nEk-1) 15000(3)\n"
        "Ek - 2 / C  Yanıcılık Sınıfı A1\nsatır"
    ]
    chunks = P.parse_madde(pages)
    assert [c.article for c in chunks] == ["171", "Ek-1", "Ek-1/A", "Ek-2/C"]
    assert chunks[2].heading == "Düşük Tehlike Kullanım Alanları"
    assert "Ek-1) 15000" in chunks[2].body
