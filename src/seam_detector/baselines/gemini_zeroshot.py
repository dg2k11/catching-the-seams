"""Baseline B2: zero-shot Gemini sentence-level baseline reserved for Milestone M4."""


def predict_passage_zeroshot(sentences, model_name="gemini-2.5-flash"):
    raise RuntimeError(
        f"Baseline B2 ({model_name}) requires an external Gemini client and is scheduled for M4."
    )
