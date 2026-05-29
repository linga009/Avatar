import os
import csv
import tempfile
from experiments.metrics_logger import MetricsLogger


def test_logger_creates_csv_with_headers():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.csv")
        logger = MetricsLogger(path)
        logger.close()
        with open(path) as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert "tick" in headers
        assert "r_mean" in headers
        assert "chi" in headers
        assert "emotion" in headers


def test_logger_writes_row():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.csv")
        logger = MetricsLogger(path)
        logger.log(tick=1, r_mean=0.45, fe_delta=-0.1, chi=0.3, tau=0.5,
                   K=0.3, unity=0.8, emotion="curiosity", intensity=0.7,
                   query="test query", prediction_error=1.5,
                   discovery=False, topic_diversity=3)
        logger.close()
        with open(path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["tick"] == "1"
        assert row["r_mean"] == "0.45"
        assert row["emotion"] == "curiosity"
