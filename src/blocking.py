try:
    from .utils import normalize_text
except ImportError:
    from utils import normalize_text


STOPWORDS = {
    "the", "and", "of", "for", "a", "an",
    "inc", "incorporated", "llc", "ltd",
    "limited", "corp", "corporation",
    "company", "co", "private", "pvt",
    "plc", "llp"
}


def get_words(x):
    x = normalize_text(x)

    return [
        w for w in x.split()
        if len(w) >= 3 and w not in STOPWORDS
    ]


def get_blocks(name, address, country):

    country = normalize_text(country)

    name_words = get_words(name)
    address_words = get_words(address)

    blocks = set()

    if not country:
        return blocks

    full_name = normalize_text(name)

    # 1. Exact normalized name
    if full_name:
        blocks.add(
            country + "_EXACT_" + full_name
        )

    if not name_words:
        return blocks

    # 2. First + last name word
    first = name_words[0]
    last = name_words[-1]

    blocks.add(
        country + "_FL_" +
        first[:6] + "_" +
        last[:6]
    )

    # 3. First two meaningful words
    if len(name_words) >= 2:

        w1 = name_words[0]
        w2 = name_words[1]

        blocks.add(
            country + "_N2_" +
            w1[:6] + "_" +
            w2[:6]
        )

        blocks.add(
            country + "_N2_" +
            w2[:6] + "_" +
            w1[:6]
        )

    # 4. Name + address combination
    if address_words:

        addr_first = address_words[0]

        blocks.add(
            country + "_NA_" +
            first[:5] + "_" +
            addr_first[:6]
        )

        blocks.add(
            country + "_NA_" +
            last[:5] + "_" +
            addr_first[:6]
        )

    # 5. Address first + last combination
    if len(address_words) >= 2:

        addr_first = address_words[0]
        addr_last = address_words[-1]

        blocks.add(
            country + "_AA_" +
            addr_first[:5] + "_" +
            addr_last[:5]
        )

    return blocks