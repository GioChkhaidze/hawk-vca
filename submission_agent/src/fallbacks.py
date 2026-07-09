import re


GENERIC_SUMMARY = "The specific subjects and actions are unclear."


def fallback_caption(style: str, factual_summary: object = None) -> str:
  return clean_summary(factual_summary)


def clean_summary(value: object, max_words: int = 42) -> str:
  text = " ".join(str(value or GENERIC_SUMMARY).split()).strip()
  if not text or text[:1] in "{[" or "`" in text or re.search(r"[\"']?factual_summary[\"']?\s*:", text):
    text = GENERIC_SUMMARY
  words = text.split()
  if len(words) < 6:
    padding = ["in", "the", "visible", "scene", "shown", "here."]
    words.extend(padding[:6 - len(words)])
  return " ".join(words[:max_words]).strip()
