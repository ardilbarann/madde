import pytest

from madde.grounding import Answer, Claim, check, normalize, numbers, quote_in_text
from madde.search import Hit

BODY = (
    "(2) Yüksek binalarda kaçış yollarının ve merdivenlerin genişliği 120 cm’den az olamaz.\n"
    "Bina Kullanım Sınıfları ’nın tanımı 3.1.1’de verilmiştir. Oran 1,5 olarak alınır."
)
HITS = [Hit(1, "yangin", "33(2)", "Kaçış yolu sayısı ve genişliği", BODY, 40)]


def claim(statement, quote, article="33(2)", regulation_id="yangin"):
    return Claim(statement=statement, regulation_id=regulation_id, article=article, quote=quote)


def answer(*claims, summary="Kaçış merdiveni genişliği sınırlıdır.", found=True):
    return Answer(found=found, summary=summary, claims=list(claims))


GOOD = claim(
    "Yüksek binalarda merdiven genişliği 120 cm'den az olamaz.",
    "merdivenlerin genişliği 120 cm'den az olamaz",
)


def test_normalize_ignores_spacing_quotes_and_turkish_case():
    assert normalize("Sınıfları ’nın  TANIMI") == normalize("sınıfları'nın tanımı")
    assert normalize("YANGIN") == normalize("yangın")


def test_numbers_treat_comma_and_dot_alike():
    assert numbers("1,5 kat, 120 cm, madde 3.1.1") == {"1.5", "120", "3.1.1"}


@pytest.mark.parametrize(
    ("quote", "ok"),
    [
        ("merdivenlerin genişliği 120 cm'den az olamaz", True),
        ("Bina Kullanım Sınıfları'nın tanımı", True),  # PDF spacing artifact in the source
        ("Yüksek binalarda ... 120 cm’den az olamaz", True),  # ellipsis skips text
        ("120 cm’den az olamaz ... Yüksek binalarda kaçış", False),  # parts out of order
        ("merdivenlerin genişliği 150 cm'den az olamaz", False),  # altered number
        ("az olamaz", False),  # too short to prove anything
    ],
)
def test_quote_in_text(quote, ok):
    assert quote_in_text(quote, BODY) is ok


def test_grounded_answer_passes():
    assert check(answer(GOOD, summary="En az 120 cm."), HITS) == []


def test_citing_an_article_that_was_not_retrieved_fails():
    bad = claim(GOOD.statement, GOOD.quote, article="34")
    [failure] = check(answer(bad), HITS)
    assert failure.claim == 0 and "not among the sources" in failure.reason


def test_fabricated_quote_fails():
    bad = claim("Genişlik 120 cm olmalıdır.", "merdiven genişliği en az 120 cm olmalıdır ve")
    [failure] = check(answer(bad), HITS)
    assert "verbatim" in failure.reason


def test_number_missing_from_quote_fails():
    bad = claim("Merdiven genişliği 1.20 m'den az olamaz.", GOOD.quote)
    [failure] = check(answer(bad), HITS)
    assert "1.20" in failure.reason


def test_article_label_numbers_are_allowed_in_the_statement():
    ok = claim("Madde 33(2)'ye göre genişlik 120 cm'den az olamaz.", GOOD.quote)
    assert check(answer(ok), HITS) == []


def test_summary_may_only_use_justified_numbers():
    [failure] = check(answer(GOOD, summary="En az 150 cm."), HITS)
    assert failure.claim is None and "150" in failure.reason


def test_found_flag_must_agree_with_claims():
    assert check(answer(found=False, summary="Kaynaklarda yok."), HITS) == []
    assert check(answer(GOOD, found=False), HITS) != []
    assert check(answer(found=True), HITS) != []


def test_numbers_from_the_question_are_allowed():
    echo = claim("51.50 m'den yüksek binalarda da genişlik 120 cm'den az olamaz.", GOOD.quote)
    assert check(answer(echo), HITS) != []
    assert check(answer(echo), HITS, question="51.50 m yüksekliğinde bina için?") == []
