"""Official sources. Texts are public-domain official texts (FSEK art. 31); not shipped in git."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    url: str
    published_on: date
    fmt: str  # "madde" (MADDE n - (1) ...) or "numbered" (TBDY-style 2.3.4.1 -)
    filename: str


SOURCES = [
    Source(
        id="tbdy-2018",
        title="Türkiye Bina Deprem Yönetmeliği (Ek: Deprem Etkisi Altında Binaların Tasarımı İçin Esaslar)",
        url="https://www.resmigazete.gov.tr/eskiler/2018/03/20180318M1-2-1.pdf",
        published_on=date(2018, 3, 18),
        fmt="numbered",
        filename="tbdy.pdf",
    ),
    Source(
        id="imar",
        title="Planlı Alanlar İmar Yönetmeliği",
        url="https://www.mevzuat.gov.tr/MevzuatMetin/yonetmelik/7.5.23722.pdf",
        published_on=date(2017, 7, 3),
        fmt="madde",
        filename="imar.pdf",
    ),
    Source(
        id="yangin",
        title="Binaların Yangından Korunması Hakkında Yönetmelik",
        url="https://www.mevzuat.gov.tr/MevzuatMetin/21.5.200712937.pdf",
        published_on=date(2007, 12, 19),
        fmt="madde",
        filename="yangin.pdf",
    ),
]
