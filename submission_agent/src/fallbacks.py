import re


GENERIC_SUMMARY = "The specific subjects and actions are unclear."
STYLE_SUFFIXES = {
  "sarcastic": "apparently making this a moment of truly historic importance",
  "humorous_tech": "as if the visible action were a process finally clearing the main queue",
  "humorous_non_tech": "like an ordinary errand receiving a completely unnecessary amount of ceremony",
}


def fallback_caption(style: str, factual_summary: object = None) -> str:
  max_summary_words = 46 if style == "formal" else 36
  summary = clean_summary(factual_summary, max_words=max_summary_words)
  suffix = STYLE_SUFFIXES.get(style)
  if suffix is None:
    return summary
  return f"{summary.rstrip('.!?')}, {suffix}."


def clean_summary(value: object, max_words: int = 42) -> str:
  text = " ".join(str(value or GENERIC_SUMMARY).split()).strip()
  if not text or text[:1] in "{[" or "`" in text or re.search(r"[\"']?factual_summary[\"']?\s*:", text):
    text = GENERIC_SUMMARY

  sentence_match = re.match(r"^.+?[.!?](?=\s|$)", text)
  sentence = sentence_match.group(0).strip() if sentence_match else text
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
