import re
from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio, partial_ratio, WRatio

DIGIT_RE = re.compile(r"\d+")


def jaccard(a, b):
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def get_numbers(text):
    return DIGIT_RE.findall(text)


def jaccard_numbers(a_text, b_text):
    nums1 = set(get_numbers(a_text))
    nums2 = set(get_numbers(b_text))
    if not nums1 or not nums2:
        return 0.0
    return len(nums1 & nums2) / len(nums1 | nums2)


def make_features(a, b):
    name1 = a["name_norm"]
    name2 = b["name_norm"]

    addr1 = a["address_norm"]
    addr2 = b["address_norm"]

    country1 = a["country_norm"]
    country2 = b["country_norm"]

    nums1 = get_numbers(name1 + " " + addr1)
    nums2 = get_numbers(name2 + " " + addr2)

    features = {
        "name_ratio": ratio(name1, name2) / 100.0,
        "name_token_set": token_set_ratio(name1, name2) / 100.0,
        "name_token_sort": token_sort_ratio(name1, name2) / 100.0,
        "name_partial": partial_ratio(name1, name2) / 100.0,
        "name_wratio": WRatio(name1, name2) / 100.0,
        "name_jaccard": jaccard(name1, name2),
        "name_exact": int(bool(name1) and name1 == name2),

        "address_ratio": ratio(addr1, addr2) / 100.0,
        "address_token_set": token_set_ratio(addr1, addr2) / 100.0,
        "address_token_sort": token_sort_ratio(addr1, addr2) / 100.0,
        "address_partial": partial_ratio(addr1, addr2) / 100.0,
        "address_wratio": WRatio(addr1, addr2) / 100.0,
        "address_jaccard": jaccard(addr1, addr2),
        "address_exact": int(bool(addr1) and addr1 == addr2),

        "country_same": int(bool(country1) and country1 == country2),

        "num_match_jaccard": jaccard_numbers(name1 + " " + addr1, name2 + " " + addr2),
        "num_exact_match": int(bool(nums1) and nums1 == nums2),
        "has_non_ascii_name": int(not name1.isascii() or not name2.isascii()),

        "name_len_diff": float(abs(len(name1) - len(name2))),
        "address_len_diff": float(abs(len(addr1) - len(addr2))),
    }

    return features