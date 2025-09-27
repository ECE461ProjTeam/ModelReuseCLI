from tests.base import BaseCLITestCase
from model import Model, Code, Dataset

class ModelInterfaceTests(BaseCLITestCase):
    def setUp(self):
        self.code = Code("https://github.com/org/repo")
        self.data = Dataset("https://huggingface.co/datasets/org/data")
        self.model = Model(url="https://huggingface.co/org/model", id="org/model")
        self.model.linkCode(self.code)
        self.model.linkDataset(self.data)

    def test_code_dataset_getters_align_with_uml(self):
        self.assertEqual(self.code.getURL(), "https://github.com/org/repo")
        self.assertIsInstance(self.code.getName(), str)
        self.assertIsInstance(self.code.getMetadata(), dict)

        self.assertEqual(self.data.getURL(), "https://huggingface.co/datasets/org/data")
        self.assertIsInstance(self.data.getName(), str)
        self.assertIsInstance(self.data.getMetadata(), dict)

    def test_model_core_fields_align_with_uml(self):
        self.assertTrue(hasattr(self.model, "url"))
        self.assertTrue(hasattr(self.model, "id"))
        self.assertIsInstance(self.model.metadata, dict)
        self.assertIsNotNone(self.model.code)
        self.assertIsNotNone(self.model.dataset)

    def test_metrics_dict_has_expected_keys(self):
        expected = {
            "ramp_up_time", "bus_factor", "performance_claims", "license",
            "size_score", "dataset_and_code_score", "dataset_quality", "code_quality",
        }
        self.assertTrue(expected.issubset(set(self.model.metrics.keys())))

