import os
import json

def test_metrics_file_creation():
    metrics_path = "data/metrics.json"
    assert os.path.exists(metrics_path), "Il file data/metrics.json non esiste."

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "f1_weighted" in metrics
