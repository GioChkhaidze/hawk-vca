# Hawk VCA

Video captioning agent for AMD ACT II Track 2.

Hawk VCA downloads each complete clip once, builds a scene-aware storyboard, conditionally extracts speech evidence,
and sends bounded perception and style intents to a private model proxy. The container never contains provider keys or
model-provider credentials.

## Pipeline

```text
video download
  +-- scene-aware storyboard -> Gemma visual draft/audit
  +-- local speech gate -> conditional Whisper transcript
                         -> native-video Gemini verification
                         -> GLM semantic reconciliation
                         -> four parallel GLM style captions
```

The runtime preserves task order, writes progress atomically, and retains complete conservative fallbacks if a provider
or deadline fails.

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
  ghcr.io/giochkhaidze/hawk-vca:v6
```

## Build

```powershell
docker build --platform linux/amd64 `
  --build-arg CAPTION_PROXY_URL=https://your-proxy.example `
  --build-arg CAPTION_PROXY_ACCESS_ID=replace-with-your-access-id `
  -t hawk-vca:v6 `
  submission_agent
```

`CAPTION_PROXY_ACCESS_ID` is an observable proxy access identifier, not a provider secret. Provider credentials and
private proxy policy are not stored in this repository or image.

## License

MIT
