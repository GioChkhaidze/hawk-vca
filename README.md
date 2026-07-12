# Hawk VCA

Video captioning agent for AMD ACT II Track 2.

Hawk VCA downloads each complete clip once, builds an adaptive chronological storyboard, conditionally extracts speech
evidence, and sends bounded perception and style intents to a private model proxy. The public container contains no
provider API keys.

## V9.6 pipeline

```text
complete video
  +-- ordered storyboard -> Qwen3-VL factual report
  +-- native video ------> Gemini 3.5 Flash factual report
  +-- speech gate -------> optional Whisper transcript
                          -> GLM-5.2 reconciliation
                          -> complete factual narrative and exclusions
                          -> four parallel evidence-led style captions
```

V9.6 renders directly from the complete reconciled factual narrative. There is no intermediate compressed caption
basis and no fixed word or sentence target. Sustained actions remain concise, while genuinely changing sequences may
retain their meaningful beginning, progression, and visible ending. The 300-word and eight-sentence limits are runaway
safety ceilings rather than targets.

Formal captions use concrete chronological description. Sarcastic and humorous captions transform the visible action
through one coherent, scene-specific premise instead of appending a detachable joke. GLM-5.2 is the primary renderer;
Qwen 3.7 Plus and MiniMax M3 provide bounded provider recovery.

The runtime globally bounds style-request concurrency, retries transient proxy failures once, preserves the strongest
safe provider response, journals results atomically, and retains an emergency schema-safe caption only after all remote
recovery paths are exhausted.

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
  ghcr.io/giochkhaidze/hawk-vca:v9.6
```

## Build

```powershell
docker build --platform linux/amd64 `
  --build-arg CAPTION_PROXY_URL=https://your-proxy.example `
  --build-arg CAPTION_PROXY_ACCESS_ID=replace-with-your-access-id `
  -t hawk-vca:v9.6 `
  submission_agent
```

`CAPTION_PROXY_ACCESS_ID` is an observable access identifier, not a provider secret. Provider credentials remain in the
private Worker.

## License

MIT
