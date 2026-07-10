# Hawk VCA

Video captioning agent for AMD ACT II.

Hawk VCA analyzes complete video clips and returns captions in four styles:
`formal`, `sarcastic`, `humorous_tech`, and `humorous_non_tech`.

## Pipeline

```text
video -> perception -> facts -> style renderer -> caption
```

Perception and rendering are separate. The renderer receives a constrained fact packet instead of the video. The
runtime preserves task order, writes progress atomically, and stops before the 600-second judging limit.

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

Create an `output` directory, then run:

```text
docker run --rm --platform linux/amd64 -v "${PWD}/submission_agent/examples:/input:ro" -v "${PWD}/output:/output" ghcr.io/giochkhaidze/hawk-vca:v4
```

## Build

```text
docker build --platform linux/amd64 --build-arg CAPTION_PROXY_URL=https://your-proxy.example --build-arg CAPTION_PROXY_ACCESS_ID=replace-with-your-access-id -t hawk-vca:local submission_agent
```

The proxy implementation and provider credentials are private. The proxy access ID is baked into the image and is not
a provider key.

## License

MIT
