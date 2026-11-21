"""Clean and deduplicate Defactify Parquet splits."""

import re
import pyarrow.parquet as pq

ERROR_PATTERNS = (
    "Error communicating with",
    "HTTPSConnectionPool",
    "Max retries exceeded",
    "I apologize, but",
    "As an AI language model",
)

BOILERPLATE_PATTERNS = (
    "Vote reporting is effectively complete",
    "Results by county",
    "Disabling auto-updates may improve",
    "Visit Our2024",
    "Centers for Disease Control and Prevention",
    "test positivity rates",
    "daily hospital admissions",
    "The data in these charts has been archived",
    "First Draft provides",
    "Come back throughout the day for the latest news",
    "Get Well's Running email",
    "ArtsBeat, a blog about arts and culture",
    "Find our latest recaps and our full archive",
    "Source: Election results and race calls",
    "new video loaded:",
    "Updated vaccine is recommended",
    "An updated vaccineis recommended",
)

SKIP_PREFIXES = (
    "supported by",
    "what to read",
    "advertisement",
    "title:",
    "by ",
    "*by ",
    "published ",
    "updated ",
    "read more",
    "sign up for",
    "subscribe ",
    "correction:",
    "follow ",
    "transcript",
    "comments",
)


def is_corrupted_or_boilerplate(text):
    if not text or len(text.strip()) < 350:
        return True
    for pat in ERROR_PATTERNS + BOILERPLATE_PATTERNS:
        if pat in text:
            return True
    return False


def clean_article_to_sentences(text):
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", " - ").replace("\u2013", "-")
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    kept = []
    for ln in raw_lines:
        if ln.startswith("#") or ln.startswith("*"):
            trimmed = ln.lstrip("#* ").rstrip("* ").strip()
            if len(trimmed.split()) < 14 or not re.search(r"[.!?][\"']?$", trimmed):
                continue
            ln = trimmed
        low = ln.lower()
        if any(low.startswith(pref) for pref in SKIP_PREFIXES):
            continue
        if "|" in ln and re.match(r"^[A-Za-z &/\-]{2,28}\|", ln):
            continue
        if re.match(r"^By[A-Z][a-z]+", ln) or "[Your Name]" in ln or "[Author" in ln or "[Date]" in ln:
            continue
        ln = re.sub(r"\*\*(.*?)\*\*", r"\1", ln)
        ln = re.sub(r"\*(.*?)\*", r"\1", ln)
        ln = re.sub(r"^[A-Z][A-Za-z .,]{2,28}\s[-]+\s+", "", ln)
        if len(ln.split()) < 8 or not re.search(r"[.!?][\"']?$", ln):
            continue
        kept.append(ln)

    body = " ".join(kept)
    # Repair fused sentence and colon breaks from scraped NYT HTML ("book.Learn" -> "book. Learn").
    body = re.sub(r"([a-z0-9\"'])([.!?])([\"']?[A-Z])", r"\1\2 \3", body)
    body = re.sub(r"([a-z0-9]):([A-Z])", r"\1: \2", body)
    protected = re.sub(
        r"\b(Mr|Mrs|Ms|Dr|Prof|Sen|Rep|Gov|Gen|St|Jr|Sr|vs|U\.S|U\.K)\.",
        r"\1<DOT>",
        body,
    )
    candidates = re.split(r"(?<=[.!?])\s+(?=[\"']?[A-Z])", protected)
    sentences = []
    for sent in candidates:
        sent = re.sub(r"\s+", " ", sent.replace("<DOT>", ".")).strip()
        wc = len(sent.split())
        if 8 <= wc <= 42 and re.search(r"[.!?][\"']?$", sent) and not re.search(r"[a-z][.!?][A-Z]", sent):
            sentences.append(sent)
    return sentences


def load_cleaned_split(parquet_path):
    table = pq.read_table(parquet_path).to_pydict()
    stats = {
        "raw_rows": len(table["Text"]),
        "corrupted_or_boilerplate": 0,
        "too_few_sentences": 0,
        "duplicate_articles": 0,
        "kept_articles": 0,
    }
    seen_sigs = set()
    records = []
    for idx, (txt, label_a, label_b) in enumerate(zip(table["Text"], table["Label_A"], table["Label_B"])):
        if is_corrupted_or_boilerplate(txt):
            stats["corrupted_or_boilerplate"] += 1
            continue
        sents = clean_article_to_sentences(txt)
        if len(sents) < 6:
            stats["too_few_sentences"] += 1
            continue
        sig = (str(label_b), sents[0][:90], sents[1][:90])
        if sig in seen_sigs:
            stats["duplicate_articles"] += 1
            continue
        seen_sigs.add(sig)
        stats["kept_articles"] += 1
        records.append(
            {
                "row_id": idx,
                "story_group": idx // 7,
                "label_a": int(label_a),
                "label_b": str(label_b),
                "sentences": sents,
            }
        )
    return records, stats
