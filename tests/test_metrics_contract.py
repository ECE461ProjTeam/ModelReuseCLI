from unittest.mock import patch
from tests.base import BaseCLITestCase
from model import Model, Code, Dataset
import logging
logger = logging.getLogger('cli_logger')


class MetricsPipelineTests(BaseCLITestCase):
    # def setUp(self):
    #     # Set up a model with linked code and dataset
    #     self.model = Model(url="https://huggingface.co/org/model", id="org/model")
    #     self.model.linkCode(Code("https://github.com/org/repo"))
    #     self.model.linkDataset(Dataset("https://huggingface.co/datasets/org/data"))

    def test_all_metrics_comprehensive(self):
        """
        Test all metrics in the Model object.
        Ensure all metrics are calculated and fall within valid ranges.
        """
        with patch("metrics.size_score.size_score", return_value={
            "raspberry_pi": 0.5,
            "jetson_nano": 0.6,
            "desktop_pc": 0.7,
            "aws_server": 0.8
        }), \
        patch("metrics.ramp_up_time.ramp_up_time", return_value=0.75), \
        patch("metrics.performance_claims.performance_claims", return_value=0.85), \
        patch("metrics.dataset_and_code_score.dataset_and_code_score", return_value=0.65), \
        patch("metrics.code_quality.code_quality", return_value=0.9), \
        patch("metrics.bus_factor.bus_factor", return_value=0.55), \
        patch("metrics.license.license", return_value=1.0), \
        patch("metrics.dataset_quality.compute_dataset_quality", return_value=0.95):
            
            # Run the pipeline
            self.model.calcMetricsParallel()

            # Validate all metrics
            for metric, value in self.model.metrics.items():
                if isinstance(value, (int, float)):
                    self.assertScore01(value)  # Ensure value is between 0 and 1
                elif isinstance(value, dict):  # For size_score
                    for platform, score in value.items():
                        self.assertScore01(score)

    def test_all_latencies_comprehensive(self):
        """
        Test all latencies in the Model object.
        Ensure all latencies are recorded and are non-negative.
        """
        with patch("metrics.size_score.size_score", return_value={
            "raspberry_pi": 0.5,
            "jetson_nano": 0.6,
            "desktop_pc": 0.7,
            "aws_server": 0.8
        }), \
        patch("metrics.ramp_up_time.ramp_up_time", return_value=0.75), \
        patch("metrics.performance_claims.performance_claims", return_value=0.85), \
        patch("metrics.dataset_and_code_score.dataset_and_code_score", return_value=0.65), \
        patch("metrics.code_quality.code_quality", return_value=0.9):
            
            # Run the pipeline
            self.model.calcMetricsParallel()

            # Validate all latencies
            for latency, value in self.model.latencies.items():
                self.assertNonNegative(value)  # Ensure latency is non-negative


    def setUp(self):
        # Set up a real model with real code and dataset links
        self.model = Model(url="https://huggingface.co/google/bert-base-uncased", id="google/bert-base-uncased")
        self.model.linkCode(Code("https://github.com/google-research/bert"))
        self.model.linkDataset(Dataset("https://huggingface.co/datasets/bookcorpus/bookcorpus"))

    def test_real_model_metrics(self):
        """
        Test all metrics in the Model object using real data.
        """
        # Run the pipeline
        self.model.calcMetricsParallel()

        logger.debug("\nMetrics:")
        for metric, value in self.model.metrics.items():
            logger.debug(f"  {metric}: {value}")

        # Debug: Print all latencies
        logger.debug("\nLatencies:")
        for latency, value in self.model.latencies.items():
            logger.debug(f"  {latency}: {value}")

        # Validate all metrics
        for metric, value in self.model.metrics.items():
            if isinstance(value, (int, float)):
                self.assertScore01(value)  # Ensure value is between 0 and 1
            elif isinstance(value, dict):  # For size_score
                for platform, score in value.items():
                    self.assertScore01(score)

        # Validate all latencies
        for latency, value in self.model.latencies.items():
            self.assertNonNegative(value)  # Ensure latency is non-negative