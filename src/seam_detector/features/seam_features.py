"""Adjacent-line seam shift features psi(c_{t-1}, c_t) (5 dimensions)."""

import math
from collections import Counter
import numpy as np
from seam_detector.features.line_features import FEATURE_NAMES

SEAM_FEATURE_NAMES = [
    "delta_smoothness",
    "delta_length",
    "char_3gram_cosine_dist",
    "delta_clause_density",
    "prev_short_or_quote_flag",
]

SMOOTH_IDX = FEATURE_NAMES.index("bigram_smoothness_proxy")
WORDS_IDX = FEATURE_NAMES.index("word_count")
COMMA_IDX = FEATURE_NAMES.index("comma_semicolon_rate")
QUOTE_IDX = FEATURE_NAMES.index("reporting_quote_flag")
SHORT_IDX = FEATURE_NAMES.index("short_sentence_flag")


def char_3gram_cosine_dist(s1, s2):
    a = s1.lower()
    b = s2.lower()
    if len(a) < 3 or len(b) < 3:
        return 1.0
    c1 = Counter(a[i : i + 3] for i in range(len(a) - 2))
    c2 = Counter(b[i : i + 3] for i in range(len(b) - 2))
    dot = sum(v * c2.get(k, 0) for k, v in c1.items())
    n1 = math.sqrt(sum(v * v for v in c1.values()))
    n2 = math.sqrt(sum(v * v for v in c2.values()))
    if n1 == 0.0 or n2 == 0.0:
        return 1.0
    return 1.0 - (dot / (n1 * n2))


def extract_seam_features(sentences, raw_line_feats):
    n = len(sentences)
    psi = np.zeros((n, len(SEAM_FEATURE_NAMES)), dtype=np.float64)
    for t in range(1, n):
        d_sm = abs(raw_line_feats[t, SMOOTH_IDX] - raw_line_feats[t - 1, SMOOTH_IDX])
        d_len = abs(raw_line_feats[t, WORDS_IDX] - raw_line_feats[t - 1, WORDS_IDX]) / 20.0
        cos_d = char_3gram_cosine_dist(sentences[t - 1], sentences[t])
        d_clause = abs(raw_line_feats[t, COMMA_IDX] - raw_line_feats[t - 1, COMMA_IDX])
        prev_short_quote = 1.0 if (raw_line_feats[t - 1, SHORT_IDX] > 0.5 or raw_line_feats[t - 1, QUOTE_IDX] > 0.5) else 0.0
        psi[t] = [d_sm, d_len, cos_d, d_clause, prev_short_quote]
    return psi
