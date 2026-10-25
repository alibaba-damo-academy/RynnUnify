# Evaluation

## Learned routing and branch controls

Use the new evaluator with the same normalized dataset and interaction contract as training:

```bash
rynn-rl-evaluate \
  --model runs/experiment/final \
  --data /path/to/test.jsonl \
  --output runs/eval-learned \
  --mode learned \
  --generator-url http://127.0.0.1:18900/generate_video \
  --judge-url http://127.0.0.1:8001/v1 --judge-model your-judge-model
```

Modes are `learned`, `direct`, and `generate`. Learned routing greedily selects between the D/G logits under the original input. Native choice questions remain direct in every mode. Responses are greedy; metrics report mean **training reward**, generation rate, and per-question output, not official benchmark accuracy. Compare modes on the same checkpoint, examples, processor budget, generator, and judge.
