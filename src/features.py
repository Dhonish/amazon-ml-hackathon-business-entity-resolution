from rapidfuzz.fuzz import ratio, token_set_ratio, WRatio


def jaccard(a, b):
    a = set(a.split())
    b = set(b.split())

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def make_features(a, b):
    name1 = a["name_norm"]
    name2 = b["name_norm"]

    addr1 = a["address_norm"]
    addr2 = b["address_norm"]

    country1 = a["country_norm"]
    country2 = b["country_norm"]

    features = {}

    features["name_ratio"] = ratio(name1, name2) / 100
    features["name_token_ratio"] = token_set_ratio(name1, name2) / 100
    features["name_wratio"] = WRatio(name1, name2) / 100
    features["name_jaccard"] = jaccard(name1, name2)

    features["address_ratio"] = ratio(addr1, addr2) / 100
    features["address_token_ratio"] = token_set_ratio(addr1, addr2) / 100
    features["address_wratio"] = WRatio(addr1, addr2) / 100
    features["address_jaccard"] = jaccard(addr1, addr2)

    features["country_same"] = int(
        country1 != "" and country1 == country2
    )

    features["name_exact"] = int(
        name1 != "" and name1 == name2
    )

    features["address_exact"] = int(
        addr1 != "" and addr1 == addr2
    )

    return features