import re

from validators import split_sentences


GENERIC_SUMMARY = "The specific subjects and actions are unclear."
STYLE_SUFFIXES = {
  "sarcastic": "with the situation apparently taking itself very seriously",
  "humorous_tech": "while the visible action completes its runtime",
  "humorous_non_tech": "with all the confidence of a carefully planned outing",
}
MAX_CAPTION_WORDS = 40


def fallback_caption(style: str, factual_summary: object = None) -> str:
  suffix = STYLE_SUFFIXES.get(style)
  if suffix is None:
    return clean_summary(factual_summary, max_words=MAX_CAPTION_WORDS)
  suffix_words = len(re.findall(r"\b[\w'-]+\b", suffix))
  summary = clean_summary(factual_summary, max_words=MAX_CAPTION_WORDS - suffix_words)
  return f"{summary.rstrip('.!?')}, {suffix}."


def clean_summary(value: object, max_words: int = 42) -> str:
  text = " ".join(str(value or GENERIC_SUMMARY).split()).strip()
  if not text or text[:1] in "{[" or "`" in text or re.search(r"[\"']?factual_summary[\"']?\s*:", text):
    text = GENERIC_SUMMARY

  sentences = split_sentences(text)
  sentence = sentences[0] if sentences else text
  words = sentence.split()
  if len(words) < 6:
    padding = ["in", "the", "visible", "scene", "shown", "here."]
    words.extend(padding[:6 - len(words)])
  truncated = len(words) > max_words
  result = " ".join(words[:max_words]).strip()
  if truncated:
    result = result.rstrip(" ,;:-.!?") + "."
  elif result[-1:] not in ".!?":
    result += "."
  return result
