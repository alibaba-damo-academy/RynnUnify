import argparse
from collections import Counter
from rynn_rl.data import load_questions

p = argparse.ArgumentParser(description="Validate normalized BCPO JSONL and local observed media")
p.add_argument("data")
args = p.parse_args()
rows = load_questions(args.data)
print(
    {
        "questions": len(rows),
        "tasks": dict(Counter(q.task for q in rows)),
        "routable": sum(q.routable for q in rows),
        "native": sum(q.native for q in rows),
    }
)
