# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os

from relax.utils.logging_utils import get_logger
from relax.utils.misc import SingletonMeta


logger = get_logger(__name__)


class _MLTrackerAdapter(metaclass=SingletonMeta):
    _ml_tracker = None
    _initialized = False

    def __init__(self, args):
        assert args.use_ml_tracker, f"{args.use_ml_tracker=}"
        try:
            import ml_tracker
        except ImportError:
            logger.warning(
                "ml_tracker is not installed. Install with: pip install ml-tracker "
                "(--extra-index-url http://yum.tbsite.net/aliyun-pypi/simple). "
                "Metrics will not be uploaded to Nebula experiment tracking."
            )
            return

        self._ml_tracker = ml_tracker
        project = getattr(args, "ml_tracker_project", None) or os.getenv("ML_TRACKER_PROJECT", None)

        if self._ml_tracker.run is None:
            self._ml_tracker.init(project=project)

        combined_dict = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))}
        self._ml_tracker.config.update(combined_dict, allow_val_change=True)

        if getattr(self._ml_tracker, "define_metric", None):
            self._ml_tracker.define_metric("train/step")
            self._ml_tracker.define_metric("train/*", step_metric="train/step", step_sync=True)
            self._ml_tracker.define_metric("rollout/step")
            self._ml_tracker.define_metric("rollout/*", step_metric="rollout/step", step_sync=True)
            self._ml_tracker.define_metric("eval/step")
            self._ml_tracker.define_metric("eval/*", step_metric="eval/step", step_sync=True)
            self._ml_tracker.define_metric("perf/*", step_metric="rollout/step", step_sync=True)
            self._ml_tracker.define_metric("passrate/*", step_metric="rollout/step", step_sync=True)

        self._initialized = True
        logger.info(f"ml_tracker initialized, project={project}")

    def log(self, data, step):
        if not self._initialized:
            return
        self._ml_tracker.log(data, step=step)

    def finish(self):
        if self._initialized and self._ml_tracker.run is not None:
            self._ml_tracker.finish()
