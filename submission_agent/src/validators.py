import re
from typing import Any


MIN_CAPTION_WORDS = 6
MAX_CAPTION_WORDS = 40
MAX_CAPTION_SENTENCES = 2

TECH_TERMS = {
  "ai",
  "algorithm",
  "api",
  "app",
  "buffer",
  "bug",
  "cache",
  "code",
  "cpu",
  "database",
  "debug",
  "file system",
  "gpu",
  "loading",
  "render",
  "server",
  "software",
  "thread",
}

STYLE_LABELS = {
  "formal",
  "sarcastic",
  "humorous_tech",
  "humorous_non_tech",
}

REFUSAL_PATTERNS = (
  re.compile(r"\bas an ai\b", re.IGNORECASE),
  re.compile(r"\bi cannot\b", re.IGNORECASE),
  re.compile(r"\bi can't\b", re.IGNORECASE),
  re.compile(r"\bunable to\b", re.IGNORECASE),
)

REASONING_LEAK_PATTERNS = (
  re.compile(r"\b(?:let me (?:analyze|reason|plan)|plan my caption|trusted narrative|chronology from|analysis of (?:the )?evidence)\b", re.IGNORECASE),
  re.compile(r"(?:^|\n)\s*\d+[.)]\s+"),
)

MARKDOWN_PATTERNS = (
  re.compile(r"^\s{0,3}#{1,6}\s+"),
  re.compile(r"^\s*[-*+]\s+"),
  re.compile(r"\*\*[^*]+\*\*"),
  re.compile(r"__[^_]+__"),
  re.compile(r"\[[^\]]+\]\([^)]+\)"),
)

JSON_FRAGMENT_PATTERNS = (
  re.compile(r"^\s*[\{\[]"),
  re.compile(r"[\}\]]\s*$"),
  re.compile(r"[\"']\s*[A-Za-z_][A-Za-z0-9_ -]*\s*[\"']\s*:"),
)

INCOMPLETE_ENDING_PATTERN = re.compile(
  r"\b(?:who|which|that|because|although|while|when|after|before|until|unless|and|or|but|with|without|to|for|from|into|onto|like|than|so|just|as if|as though)\s*[.!?][\"']?$",
  re.IGNORECASE,
)


def validate_caption(caption: Any, style: str) -> list[str]:
  if not isinstance(caption, str):
    return ["caption must be a string"]

  text = caption.strip()
  if not text:
    return ["caption must be non-empty"]

  reasons = []
  word_count = len(re.findall(r"\b[\w'-]+\b", text))
  if word_count < MIN_CAPTION_WORDS:
    reasons.append("caption is too short")
  if word_count > MAX_CAPTION_WORDS:
    reasons.append("caption is too long")
  sentence_count = max(1, len(split_sentences(text)))
  if sentence_count > MAX_CAPTION_SENTENCES:
    reasons.append("caption has too many sentences")
  if "```" in text or "`" in text or _contains_markdown(text):
    reasons.append("caption must not contain markdown")
  if _contains_json_fragment(text):
    reasons.append("caption must not contain JSON fragments")
  if any(pattern.search(text) for pattern in REFUSAL_PATTERNS):
    reasons.append("caption must not contain refusal text")
  if any(pattern.search(text) for pattern in REASONING_LEAK_PATTERNS):
    reasons.append("caption must not expose reasoning or planning")
  if not re.search(r"[.!?][\"']?$", text) or INCOMPLETE_ENDING_PATTERN.search(text):
    reasons.append("caption must be a complete sentence")
  if _has_style_prefix(text, style):
    reasons.append("caption must not start with a style label")
  return reasons


def _contains_json_fragment(text: str) -> bool:
  return any(pattern.search(text) for pattern in JSON_FRAGMENT_PATTERNS)


def split_sentences(value: object) -> list[str]:
  text = str(value or "").strip()
  if not text:
    return []
  marker = "\x00"
  protected = re.sub(r"(\d)\.(\d)", rf"\1{marker}\2", text)
  protected = re.sub(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|vs|etc)\.",
    lambda match: match.group(0).replace(".", marker),
    protected,
    flags=re.IGNORECASE,
  )
  protected = re.sub(
    r"(?:\b[A-Z]\.\s*){2,}",
    lambda match: match.group(0).replace(".", marker),
    protected,
  )
  protected = re.sub(r"\b([A-Z])\.(?=\s+[A-Z][A-Za-z'-]+)", rf"\1{marker}", protected)
  sentences = re.findall(r"[^.!?]+[.!?]+(?:[\"'](?=\s|$))?|[^.!?]+$", protected)
  return [sentence.replace(marker, ".").strip() for sentence in sentences if sentence.strip()]


def _contains_markdown(text: str) -> bool:
  return any(pattern.search(text) for pattern in MARKDOWN_PATTERNS)


def _has_style_prefix(text: str, style: str) -> bool:
  if ":" not in text:
    return False

  label = text.split(":", 1)[0]
  if len(label) > 32:
    return False

  normalized_label = _normalize_label(label)
  normalized_style = _normalize_label(style)
  return normalized_label == normalized_style or normalized_label in STYLE_LABELS


def _normalize_label(value: str) -> str:
  return re.sub(r"[\s-]+", "_", value.strip().lower())


def _contains_tech_term(text: str) -> bool:
  lowered = text.lower()
  for term in TECH_TERMS:
    if " " in term:
      if term in lowered:
        return True
    elif re.search(rf"\b{re.escape(term)}\b", lowered):
      return True
  return False
