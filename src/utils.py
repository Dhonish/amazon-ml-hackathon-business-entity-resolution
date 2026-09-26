import re
import unicodedata

import pandas as pd


def normalize_text(x):

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""

    x = str(x).lower()

    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))

    x = x.replace("&", " and ")

    x = re.sub(r"[^a-z0-9\s]", " ", x)

    x = re.sub(r"\s+", " ", x).strip()

    return x


def normalize_name(x):
    x = normalize_text(x)

    replacements = {
        "private limited": "pvt ltd",
        "private": "pvt",
        "limited": "ltd",
        "corporation": "corp",
        "company": "co"
    }

    for a, b in replacements.items():
        x = x.replace(a, b)

    return x


def normalize_address(x):
    x = normalize_text(x)

    replacements = {
        "road": "rd",
        "street": "st",
        "avenue": "ave",
        "boulevard": "blvd",
        "lane": "ln",
        "building": "bldg",
        "apartment": "apt"
    }

    for a, b in replacements.items():
        x = x.replace(a, b)

    return x