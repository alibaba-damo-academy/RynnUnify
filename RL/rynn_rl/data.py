from dataclasses import dataclass, field
from pathlib import Path
import json

TASKS = {"choice", "phys", "distance", "point", "trajectory", "text", "mess"}


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    reference: str
    task: str
    images: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    native: bool = False

    @property
    def routable(self):
        return not self.native


def load_questions(path):
    path = Path(path).resolve()
    rows, ids = [], set()
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        row = Question(**{k: raw[k] for k in Question.__dataclass_fields__ if k in raw})
        if not all(isinstance(v, str) and v.strip() for v in (row.id, row.question, row.reference)):
            raise ValueError(f"Line {line_no}: id, question, reference must be nonempty strings")
        if row.id in ids or row.task not in TASKS or not isinstance(row.native, bool):
            raise ValueError(f"Line {line_no}: duplicate ID, unknown task, or invalid native flag")
        if not isinstance(row.images, list) or not isinstance(row.videos, list):
            raise ValueError("images/videos must be arrays of local filenames")
        for paths in (row.images, row.videos):
            for i, name in enumerate(paths):
                p = Path(name).expanduser()
                p = p if p.is_absolute() else path.parent / p
                if not p.is_file():
                    raise FileNotFoundError(p)
                paths[i] = str(p.resolve())
        if row.videos and row.images or len(row.videos) > 1:
            raise ValueError("Use either images or one observed video per question")
        if len(row.images) >= 3 and not row.native:
            raise ValueError("Native multi-image questions must set native=true")
        if row.native and row.task != "choice":
            raise ValueError("This public native adapter supports choice tasks only")
        ids.add(row.id)
        rows.append(row)
    if not rows:
        raise ValueError("Empty dataset")
    return rows
