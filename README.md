# Hawk VCA

Video captioning agent for AMD ACT II Track 2.

Hawk VCA downloads each complete clip once, builds a chronological scene-aware storyboard, conditionally extracts
speech evidence, and sends bounded perception and style intents to a private model proxy. The container contains no
provider API keys.

## V9.5.4 pipeline

```text
complete video download
  +-- timestamped storyboard -> Qwen3-VL 30B Thinking factual report
  +-- native video ----------> Gemini 3.5 Flash factual report
  +-- local speech gate -----> optional transcript evidence
                              -> GLM-5.2 semantic reconciliation
                              -> shared compact caption basis
                              -> four parallel GLM-5.2 style captions
```

V9.5.4 retains the V9.5.3 caption policy: one complete sentence by default and a second only when a meaningful
transition needs it. Captions are
bounded to 40 words. All four styles preserve one reconciled semantic proposition while integrating formal, sarcastic,
humorous technical, or humorous non-technical wording into the description itself.

The V9.5.4 runtime refactor adds adaptive storyboard preprocessing, earlier deadline-aware degradation, progressive
atomic result journaling, and one centralized caption-validity boundary. These changes reduce avoidable preprocessing
and finalization work without changing the requested caption semantics or public input/output contract.

Low-motion clips retain their available storyboard evidence instead of unnecessarily abandoning the ensemble. Exact
identity, score, event, location, object, and outcome claims are conservatively generalized when the visual reports
do not support the same precision.

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
  ghcr.io/giochkhaidze/hawk-vca:v9.5.4
```

## Build

```powershell
docker build --platform linux/amd64 `
  --build-arg CAPTION_PROXY_URL=https://your-proxy.example `
  --build-arg CAPTION_PROXY_ACCESS_ID=replace-with-your-access-id `
  -t hawk-vca:v9.5.4 `
  submission_agent
```

`CAPTION_PROXY_ACCESS_ID` is an observable proxy access identifier, not a provider secret. Provider credentials and
private proxy policy are not stored in this repository or image.

## License

MIT
