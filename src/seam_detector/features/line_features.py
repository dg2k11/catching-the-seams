"""Per-line linguistic cues and bigram reference predictability features phi(c_t)."""

import math
import re
from collections import Counter
import numpy as np

FEATURE_NAMES = [
    "ref_bigram_ll",
    "bigram_smoothness_proxy",
    "word_count",
    "len_dev_from_passage",
    "mean_word_chars",
    "type_token_ratio",
    "function_word_ratio",
    "comma_semicolon_rate",
    "dash_colon_paren_flag",
    "digit_specificity_flag",
    "proper_noun_ratio",
    "reporting_quote_flag",
    "nominalization_ratio",
    "antithesis_flag",
    "participial_clause_flag",
    "llm_discourse_opener_flag",
    "passive_voice_flag",
    "short_sentence_flag",
]

FUNCTION_WORDS = {
    "the", "a", "an", "of", "to", "in", "for", "on", "with", "at", "by", "from",
    "as", "is", "was", "are", "were", "be", "been", "that", "which", "who", "this",
    "these", "those", "it", "its", "and", "or", "but", "if", "while", "although",
}
REPORTING_PAT = re.compile(r"([\"']|\bsaid\b|\btold\b|\bwrote\b|\baccording to\b|\breported\b)", re.I)
NOMINAL_PAT = re.compile(r"(tion|ment|ness|ity|ance|ence)s?$", re.I)
ANTITHESIS_PAT = re.compile(r"(\bnot\s+[^,;.]{2,40},\s*but\b|\brather than\b)", re.I)
PARTICIPIAL_PAT = re.compile(r"(,\s+[a-z]{4,}ing\b|^[A-Z][a-z]{3,}ing\b.*?,)")
DISCOURSE_OPENER_PAT = re.compile(
    r"^(Moreover|Furthermore|Additionally|Consequently|Overall|Ultimately|Notably|"
    r"Significantly|Importantly|In conclusion|In summary|Meanwhile|Nevertheless),",
    re.I,
)
PASSIVE_PAT = re.compile(r"\b(was|were|been|is|are|be)\s+([a-z]+ly\s+)?[a-z]{3,}ed\b", re.I)


class ReferenceBigramLM:
    """Laplace/interpolated bigram language model fit on disjoint Human NYT articles."""

    def __init__(self):
        self.unigram = Counter()
        self.bigram = Counter()
        self.total_tokens = 0
        self.vocab_size = 1

    def fit(self, sentence_lists):
        for sents in sentence_lists:
            for s in sents:
                toks = ["<s>"] + re.findall(r"[a-z0-9']+", s.lower()) + ["</s>"]
                self.unigram.update(toks)
                self.total_tokens += len(toks)
                for a, b in zip(toks[:-1], toks[1:]):
                    self.bigram[(a, b)] += 1
        self.vocab_size = max(1, len(self.unigram))

    def token_logprobs(self, s):
        toks = ["<s>"] + re.findall(r"[a-z0-9']+", s.lower()) + ["</s>"]
        lps = []
        for a, b in zip(toks[:-1], toks[1:]):
            p_uni = (self.unigram[b] + 1.0) / (self.total_tokens + self.vocab_size)
            c_prev = self.unigram[a]
            if c_prev > 0:
                p_bi = (self.bigram[(a, b)] + 0.5) / (c_prev + 0.5 * self.vocab_size)
                p = 0.7 * p_bi + 0.3 * p_uni
            else:
                p = p_uni
            lps.append(math.log(max(p, 1e-12)))
        return lps

    def score_sentence(self, s):
        lps = self.token_logprobs(s)
        if not lps:
            return -8.0, 0.0
        mean_lp = sum(lps) / len(lps)
        std_lp = math.sqrt(sum((x - mean_lp) ** 2 for x in lps) / len(lps)) + 1e-4
        return mean_lp, (mean_lp + 6.5) / std_lp


def extract_line_features(sentences, ref_lm):
    lengths = [len(s.split()) for s in sentences]
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0
    feats = np.zeros((len(sentences), len(FEATURE_NAMES)), dtype=np.float64)

    for t, s in enumerate(sentences):
        words = s.split()
        alpha_toks = re.findall(r"[A-Za-z']+", s)
        lower_toks = [w.lower() for w in alpha_toks]
        wc = len(words) or 1
        n_alpha = len(lower_toks) or 1

        mean_lp, smooth = ref_lm.score_sentence(s)
        mean_chars = sum(len(w) for w in alpha_toks) / n_alpha
        ttr = len(set(lower_toks)) / n_alpha
        func_ratio = sum(1 for w in lower_toks if w in FUNCTION_WORDS) / n_alpha
        comma_semi = (s.count(",") + s.count(";")) / wc
        dash_colon = 1.0 if ("-" in s or ":" in s or "(" in s) else 0.0
        has_digit = 1.0 if re.search(r"\d", s) else 0.0
        proper_mid = sum(1 for w in words[1:] if re.match(r"^[A-Z][a-zA-Z.-]+$", w)) / max(1, wc - 1)
        has_reporting = 1.0 if REPORTING_PAT.search(s) else 0.0
        nom_ratio = sum(1 for w in lower_toks if len(w) >= 7 and NOMINAL_PAT.search(w)) / n_alpha
        has_antithesis = 1.0 if ANTITHESIS_PAT.search(s) else 0.0
        has_part = 1.0 if PARTICIPIAL_PAT.search(s) else 0.0
        has_opener = 1.0 if DISCOURSE_OPENER_PAT.search(s) else 0.0
        has_passive = 1.0 if PASSIVE_PAT.search(s) else 0.0
        is_short = 1.0 if wc <= 12 else 0.0

        feats[t] = [
            mean_lp,
            smooth,
            float(wc),
            (wc - mean_len) / (mean_len + 1.0),
            mean_chars,
            ttr,
            func_ratio,
            comma_semi,
            dash_colon,
            has_digit,
            proper_mid,
            has_reporting,
            nom_ratio,
            has_antithesis,
            has_part,
            has_opener,
            has_passive,
            is_short,
        ]
    return feats
