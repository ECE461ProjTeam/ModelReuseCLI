import threading
import time
from metrics.size_score import size_score
from metrics.ramp_up_time import ramp_up_time
import json
from typing import Dict, Union
class Code:
    def __init__(self, url: str) -> None:
        self._url = url
        self._name = ""
        self._metadata = {}
        self._path_to_cloned = ""
        self._code_quality = 0

    def getURL(self) -> str:
        return self._url

    def getName(self) -> str:
        return self._name

    def getMetadata(self) -> dict:
        return self._metadata

    def getPathToCloned(self) -> str:
        return self._path_to_cloned

    def getCodeQuality(self) -> int:
        return self._code_quality


class Dataset:
    def __init__(self, url: str) -> None:
        self._url = url
        self._name = ""
        self._metadata = {}
        self._path_to_cloned = ""
        self._dataset_quality = 0

    def getURL(self) -> str:
        return self._url

    def getName(self) -> str:
        return self._name

    def getMetadata(self) -> dict:
        return self._metadata

    def getPathToCloned(self) -> str:
        return self._path_to_cloned

    def getDatasetQuality(self) -> int:
        return self._dataset_quality


class Model:
    def __init__(self, url: str = "", id: str = "") -> None:
        self.url = url
        self.id: str = id
        self.name = ""
        self.code = None  # instance of Code class
        self.dataset = None  # instance of Dataset class
        self.metadata = {}
        self.metrics = {
            "net_score": 0,
            "ramp_up_time": 0, 
            "bus_factor": 0, 
            "performance_claims": 0, 
            "license": 0, 
            "size_score": {
                "raspberry_pi": 0,
                "jetson_nano": 0,
                "desktop_pc": 0,
                "aws_server": 0
            }, 
            "dataset_and_code_score": 0, 
            "dataset_quality": 0, 
            "code_quality": 0
        }
        self.latencies = {
            "net_score_latency": 0,
            "ramp_up_time_latency": 0,
            "bus_factor_latency": 0,
            "performance_claims_latency": 0,
            "license_latency": 0,
            "size_score_latency": 0,
            "dataset_and_code_score_latency": 0,
            "dataset_quality_latency": 0,
            "code_quality_latency": 0
        }
        self.hfAPIData = {}
        self.gitAPIData = {}

    # Evaluate model
    def evaluate(self) -> Dict[str, Union[int, float, str, Dict[str, float]]]:
        t = int(time.perf_counter_ns() / 1e6)
        self.calcMetricsParallel()
        self.calcNetScore()
        self.latencies["net_score_latency"] = int(time.perf_counter_ns() / 1e6 - t)
        res =  {
            "name": self.name,
            "category": "MODEL",
        }
        res.update(self.metrics)
        res.update(self.latencies)
        return res

    def calcMetricsParallel(self) -> None:
        threads = []
        funcs = {
            "ramp_up_time": self.calcRampUp,
            "bus_factor": self.calcBusFactor,
            "performance_claims": self.calcPerformanceClaims,
            "license": self.calcLicense,
            "size_score": self.calcSize,
            "dataset_and_code_score": self.calcDatasetCode,
            "dataset_quality": self.calcDatasetQuality,
            "code_quality": self.calcCodeQuality,
        }
        for key in funcs:
            t = threading.Thread(target=funcs[key])
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def calcSize(self) -> None:
        # Time in milliseconds
        t = int(time.perf_counter_ns() / 1e6)
        self.metrics["size_score"] = size_score(self.id)
        self.latencies["size_score_latency"] = int(time.perf_counter_ns() / 1e6 - t)

    def calcRampUp(self) -> None:
        t = time.perf_counter()
        score = ramp_up_time(self.id)  # returns float 
        self.metrics["ramp_up_time"] = score
        self.latencies["ramp_up_time_latency"] = time.perf_counter() - t

    def calcBusFactor(self) -> None:
        self.metrics["bus_factor"] = 1

    def calcPerformanceClaims(self) -> None:
        self.metrics["performance_claims"] = 1

    def calcLicense(self) -> None:
        self.metrics["license"] = 1

    def calcDatasetCode(self) -> None:
        self.metrics["dataset_and_code_score"] = 1

    def calcDatasetQuality(self) -> None:
        self.metrics["dataset_quality"] = 1

    def calcCodeQuality(self) -> None:
        self.metrics["code_quality"] = 1

    def calcNetScore(self) -> None:
        self.metrics['net_score'] = 0.08 * (0.05 * self.metrics["size_score"]["raspberry_pi"] + \
                                0.15 * self.metrics["size_score"]["jetson_nano"] + \
                                0.3 * self.metrics["size_score"]["desktop_pc"] + \
                                0.5 * self.metrics["size_score"]["aws_server"]) + \
                        0.12 * self.metrics["license"] + \
                        0.2 * self.metrics["ramp_up_time"] + \
                        0.05 * self.metrics["bus_factor"] + \
                        0.1 * self.metrics["dataset_and_code_score"] + \
                        0.15 * self.metrics["dataset_quality"] + \
                        0.1 * self.metrics["code_quality"] + \
                        0.2 * self.metrics["performance_claims"]
        self.metrics['net_score'] = round(self.metrics['net_score'], 2)

    def linkCode(self, code: Code):
        self.code = code
    
    def linkDataset(self, dataset: Dataset):
        self.dataset = dataset
    

if __name__ == "__main__":
    model = Model(id = "microsoft/DialoGPT-medium")
    model.calcMetricsParallel()
    output = {}
    output.update(model.metrics)
    output.update(model.latencies)

    print(json.dumps(output, indent=4))
    
    model = Model(id = "deepseek-ai/DeepSeek-R1")
    model.calcMetricsParallel()
    output = {}
    output.update(model.metrics)
    output.update(model.latencies)

    print(json.dumps(output, indent=4))
    
