"""Construct Slice H and Slice E(k) passages with unique human donors."""

import random
import re
from seam_detector.schema import Label, LineRecord, Passage

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "that", "this", "it", "its", "their", "his", "her", "they", "them", "he", "she",
    "we", "you", "have", "has", "had", "not", "no", "into", "over", "after", "before",
}


def content_tokens(sentences):
    toks = re.findall(r"[a-z]{4,}", " ".join(sentences).lower())
    return {t for t in toks if t not in STOPWORDS}


def crop_window(sentences, rng, min_s=6, max_s=16, min_words=150, max_words=400):
    target_n = min(len(sentences), rng.randint(8, 14))
    window = list(sentences[:target_n])
    while len(window) < min(max_s, len(sentences)) and sum(len(s.split()) for s in window) < min_words:
        window.append(sentences[len(window)])
    while len(window) > min_s and sum(len(s.split()) for s in window) > max_words:
        window.pop()
    total_words = sum(len(s.split()) for s in window)
    if not (min_s <= len(window) <= max_s and min_words <= total_words <= max_words):
        return None
    return window


def build_split_passages(
    records,
    split_name,
    n_slice_h,
    n_per_k,
    edit_rates=(0, 25, 50, 75),
    seed=221,
    min_s=6,
    max_s=16,
    min_words=150,
    max_words=400,
):
    rng = random.Random(seed)
    human_pool = [r for r in records if r["label_a"] == 0]
    ai_pool = [r for r in records if r["label_a"] == 1]
    rng.shuffle(human_pool)
    rng.shuffle(ai_pool)

    slice_h_candidates = []
    used_row_ids = set()
    for art in human_pool:
        win = crop_window(art["sentences"], rng, min_s, max_s, min_words, max_words)
        if win is not None:
            slice_h_candidates.append((art, win))
            used_row_ids.add(art["row_id"])
        if len(slice_h_candidates) >= n_slice_h:
            break

    donor_pool = [
        (art, content_tokens(art["sentences"][:14]))
        for art in human_pool
        if art["row_id"] not in used_row_ids
    ]
    used_donor_ids = set()

    passages = []
    for i, (_, sents) in enumerate(slice_h_candidates):
        lines = [LineRecord(t, s, Label.HUMAN, source_model="Human_Story", was_rewritten=False) for t, s in enumerate(sents)]
        passages.append(Passage(f"{split_name}_H_{i:04d}", "H", 100, split_name, lines))

    by_model = {}
    for r in ai_pool:
        by_model.setdefault(r["label_b"], []).append(r)
    models = sorted(by_model.keys())
    model_cursors = {m: 0 for m in models}

    for k in edit_rates:
        built_for_k = 0
        attempts = 0
        while built_for_k < n_per_k and attempts < 10 * n_per_k:
            attempts += 1
            m = models[built_for_k % len(models)]
            art = by_model[m][model_cursors[m] % len(by_model[m])]
            model_cursors[m] += 1
            ai_sents = crop_window(art["sentences"], rng, min_s, max_s, min_words, max_words)
            if ai_sents is None:
                continue
            n = len(ai_sents)
            num_human = round((k / 100.0) * n)

            rewrite_positions = set()
            if num_human > 0:
                if num_human >= 2 and rng.random() < 0.65:
                    span_len = max(1, num_human // 2)
                    start = rng.randint(0, n - span_len)
                    rewrite_positions.update(range(start, start + span_len))
                remaining = [idx for idx in range(n) if idx not in rewrite_positions]
                need = num_human - len(rewrite_positions)
                if need > 0:
                    rewrite_positions.update(rng.sample(remaining, need))

            avail_donors = [
                (d_art, d_toks)
                for d_art, d_toks in donor_pool
                if d_art["row_id"] not in used_donor_ids and len(d_art["sentences"]) >= n
            ]
            if not avail_donors:
                raise RuntimeError(f"Ran out of unique human donor articles in split {split_name}")
            sample_donors = rng.sample(avail_donors, min(50, len(avail_donors)))
            ai_kw = content_tokens(ai_sents)
            best_donor, _ = max(sample_donors, key=lambda pair: len(ai_kw & pair[1]))
            if num_human > 0:
                used_donor_ids.add(best_donor["row_id"])
            donor_sents = best_donor["sentences"]

            lines = []
            for t in range(n):
                if t in rewrite_positions:
                    lines.append(LineRecord(t, donor_sents[t], Label.HUMAN, source_model="Human_Story", was_rewritten=True))
                else:
                    lines.append(LineRecord(t, ai_sents[t], Label.AI, source_model=m, was_rewritten=False))

            total_words = sum(len(ln.text.split()) for ln in lines)
            if not (min_words <= total_words <= max_words):
                continue

            passages.append(Passage(f"{split_name}_E{k}_{built_for_k:04d}", f"E_{k}", k, split_name, lines))
            built_for_k += 1
    return passages
