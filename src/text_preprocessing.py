import re


WHITESPACE_PATTERN = re.compile(r"\s+")
PUNCTUATION_PATTERN = re.compile(r"[^a-z0-9\s+#]")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
LINE_SPLIT_PATTERN = re.compile(r"[\n\r]+|(?<=\.)\s+|(?<=!)\s+|(?<=\?)\s+|(?<=:)\s+|(?<=;)\s+")


def preprocess_text(text: str) -> str:
    """Clean text conservatively while preserving skill evidence terms."""

    if not text:
        return ""

    cleaned = text.lower()
    cleaned = PUNCTUATION_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


def normalise_skill_name(skill: str) -> str:
    """Normalize a skill name for dictionary lookup and comparison."""

    return preprocess_text(skill)


def split_text_into_chunks(text: str) -> list[str]:
    """Split text into short evidence chunks for chunk-level scoring."""

    if not text:
        return []

    chunks = [part.strip(" -\t") for part in LINE_SPLIT_PATTERN.split(str(text)) if part.strip(" -\t")]
    return chunks or [str(text).strip()]


def extract_evidence_excerpt(description: str, reference_text: str, skill: str) -> str:
    """Return the sentence with the strongest lexical overlap to the skill evidence."""

    if not description:
        return ""

    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(description) if sentence.strip()]
    if not sentences:
        return description.strip()

    reference_tokens = set(preprocess_text(reference_text).split())
    reference_tokens.update(preprocess_text(skill).split())

    best_sentence = sentences[0]
    best_score = -1
    for sentence in sentences:
        sentence_tokens = set(preprocess_text(sentence).split())
        score = len(sentence_tokens & reference_tokens)
        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence
