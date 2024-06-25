from mlflow.utils.time import get_current_time_millis
from mlflow.tracking.fluent import _get_or_start_run
from mlflow.tracking.client import MlflowClient
from mlflow.entities.metric import Metric
from typing import List


def log_metrics_over_steps(label: str, metrics: List[float]):
    """Edit method log_metrics to allow batch logging with different steps
    source: https://mlflow.org/docs/latest/_modules/mlflow/tracking/fluent.html#log_metrics"""
    run_id = _get_or_start_run().info.run_id
    timestamp = get_current_time_millis()
    metrics_arr = [Metric(label, value, timestamp, i) for i, value in enumerate(metrics)]
    print(f'Pushing {len(metrics_arr)} metrics for {label}')
    return MlflowClient().log_batch(
        run_id=run_id, metrics=metrics_arr, params=[], tags=[], synchronous=True
    )
