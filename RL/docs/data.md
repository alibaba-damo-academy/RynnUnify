# Data contract

Use one JSON object per line:

```json
{"id":"example-1","question":"What is the object's final speed?","reference":"20 cm/s","task":"phys","images":[],"videos":["media/observed.mp4"],"native":false}
```

Required fields are `id`, `question`, `reference`, and `task`. Optional `images`/`videos` default to empty lists; `native` defaults to false. Media paths are local files, resolved relative to the JSONL file. Every ID must be unique before final-batch cyclic padding. Use `task` from `choice`, `phys`, `distance`, `point`, `trajectory`, `text`, `mess`.

- Routable rows: one/two observed images OR one observed video. Never provide future reference media as observations.
- Native benchmark rows: set `native=true`; the public adapter currently supports choice tasks and eight direct trajectories. Three-or-more-image rows must use this mode.
- `reference` is the scoring label and is never sent to the actor or generator.
- Reference generation requests, SFT future videos and old reasoning responses are not part of this schema.
- Checkpoints and datasets are not bundled; obtain them under their respective access and license terms.


Validate a dataset without loading any model:

```bash
python scripts/check_data.py /path/to/train.jsonl
```
