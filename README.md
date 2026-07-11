# Hawk VCA

Video captioning agent for AMD ACT II Track 2.

Hawk VCA downloads each complete clip once, builds a timestamped scene-aware storyboard, conditionally extracts
speech evidence, and sends bounded perception and style intents to a private model proxy. The container contains no
provider API keys.

## V9.3.1 pipeline

```text
complete video download
  +-- timestamped storyboard -> Qwen3-VL factual report
  +-- native video ----------> Gemini 3.1 Flash-Lite factual report
  +-- local speech gate -----> conditional transcript evidence
                              -> GLM semantic reconciliation
                              -> four parallel GLM style captions
```

V9.3.1 preserves the dominant visible event while allowing scene-grounded formal, sarcastic, humorous technical, and
humorous non-technical narration. Soft style validation can trigger one repair, but a structurally safe provider draft
is retained so a stylistic false positive cannot degrade into a generic local caption. Promotional and montage
material is summarized without treating genuine extra footage as hallucinated.

## Contract

Input: `/input/tasks.json`

```json
[
  {
    "task_id": "v1",
    "video_url": "https://example.com/video.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

Output: `/output/results.json`

```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

## Run

```powershell
New-Item -ItemType Directory -Force -Path output
docker run --rm --platform linux/amd64 `
  -v "${PWD}/submission_agent/examples:/input:ro" `
  -v "${PWD}/output:/output" `
  ghcr.io/giochkhaidze/hawk-vca:v9.3.1
```

## Build

```powershell
docker build --platform linux/amd64 `
  --build-arg CAPTION_PROXY_URL=https://your-proxy.example `
  --build-arg CAPTION_PROXY_ACCESS_ID=replace-with-your-access-id `
  -t hawk-vca:v9.3.1 `
  submission_agent
```

`CAPTION_PROXY_ACCESS_ID` is an observable proxy access identifier, not a provider secret. Provider credentials and
private proxy policy are not stored in this repository or image.

## License

MIT
