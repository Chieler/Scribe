# captionlm

Local captions that preserve terminology from a trusted source document.
Give it a recording plus the notes, slides, reading, or glossary it is about;
it emits SRT captions and plain text without sending either artifact to a server.

captionlm is designed for Apple Silicon and MLX. It does **not** fine-tune a
model: it compiles the document's term list into a context graph at
transcription time.

## Use it

See [SETUP.md](SETUP.md), then put a recording and its document in `dropoff/`
and run:

```bash
venv/bin/python scripts/serve.py
```

Open http://localhost:8756, or use the batch command:

```bash
venv/bin/python scripts/caption_dropoff.py dropoff
```

Developers can benchmark against a source transcript at
http://localhost:8756/benchmark. Drop `recording.m4a` with
`recording.reference.txt`; the source is used only for normalized WER scoring.

For one file from the command line, choose plain text explicitly:

```bash
venv/bin/python -m captionlm.cli recording.m4a --format txt
```

A document with the same basename as a recording applies only to that
recording. An unpaired document is treated as a shared glossary. A recording
with no document is transcribed normally, with no contextual bias.

## Choose a model

Two acoustic models, picked per job, not installed together:

| Model | Flag / preset | Size | Speed |
|---|---|---|---|
| 110M hybrid (default) | `110m` | 438 MB | ~59x realtime |
| 1.1B hybrid | `1.1b` | 4.0 GB | ~30x realtime |

Both download from Hugging Face on first use and cache under
`~/.cache/huggingface`. To fetch one ahead of time instead of waiting on
first run:

```bash
hf download mlx-community/parakeet-tdt_ctc-110m
hf download mlx-community/parakeet-tdt_ctc-1.1b
```

Pick per entry point:

- Browser (`scripts/serve.py`): Small/Large buttons in the UI.
- Batch (`scripts/caption_dropoff.py`): `--model mlx-community/parakeet-tdt_ctc-1.1b`
- CLI (`captionlm.cli`): `--model mlx-community/parakeet-tdt_ctc-1.1b`

See [docs/RESULTS.md](docs/RESULTS.md) for the WER/speed trade-off behind
the choice.

## Operating policy

- Short, document-matched recordings use domain bias at `cb_weight=3.0`.
- Recordings longer than five minutes use the conservative `cb_weight=1.0`.
- This is automatic in both the browser and batch product paths.
- The 110M hybrid model is the default (438 MB); select the 1.1B hybrid model
  (4 GB) when accuracy is worth the added download and runtime.

The evidence and limits behind these choices are in
[docs/RESULTS.md](docs/RESULTS.md). The implementation/data-flow is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
