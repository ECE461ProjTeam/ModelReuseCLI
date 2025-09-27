import threading
import time
import json
from typing import Dict, Union
from apis.gemini import get_gemini_key
# from clone_bridge import clone_with_isogit
from metrics.performance_claims import performance_claims
from metrics.dataset_and_code_score import dataset_and_code_score
from metrics.size_score import size_score
from metrics.ramp_up_time import ramp_up_time
from metrics.dataset_quality import compute_dataset_quality
from metrics.bus_factor import bus_factor
from metrics.code_quality import code_quality
from metrics.license import license_score


class Code:
    def __init__(self, url: str) -> None:
        self._url = url
        self._name = ""
        self._metadata = {}
        self._path_to_cloned = ""
        self._code_quality = 0
        self.type = ""

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
        self.id = id
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
        t = int(time.perf_counter_ns() / 1e6)
        score = ramp_up_time(self.id)  # returns float 
        self.metrics["ramp_up_time"] = round(score, ndigits=2)
        self.latencies["ramp_up_time_latency"] = int(time.perf_counter_ns() / 1e6 - t)

    def calcBusFactor(self) -> None:
        t = int(time.perf_counter_ns() / 1e6)
        if self.code:
            code_type = self.code.type if self.code else None
            code_id = self.code._url[self.code._url.index(f"{code_type}.com")+11:] if self.code else None
            self.metrics["bus_factor"] = bus_factor(code_id)
        else:
            self.metrics["bus_factor"] = 0.0

        self.latencies["bus_factor_latency"] = int(time.perf_counter_ns() / 1e6 - t)


    def calcPerformanceClaims(self) -> None:
        t = int(time.perf_counter_ns() / 1e6)
        self.metrics["performance_claims"] = performance_claims(self.id)
        self.latencies["performance_claims_latency"] = int(time.perf_counter_ns() / 1e6 - t)


    def calcLicense(self) -> None:
        t = int(time.perf_counter_ns() / 1e6)
        self.metrics["license"] = license_score(self.id)
        self.latencies["license_latency"] = int(time.perf_counter_ns() / 1e6 - t)


    def calcDatasetCode(self) -> None:
        t = int(time.perf_counter_ns() / 1e6)
        code_type = self.code.type if self.code else None
        code_id = self.code._url[self.code._url.index(f"{code_type}.com")+11:] if self.code else None
        dataset_id = self.dataset._name if self.dataset else None
        self.metrics["dataset_and_code_score"] = dataset_and_code_score(dataset_id, code_id, code_type)
        self.latencies["dataset_and_code_score_latency"] = int(time.perf_counter_ns() / 1e6 - t)

    def calcDatasetQuality(self) -> None:
        t = int(time.perf_counter_ns() / 1e6)
        if self.dataset:
            self.metrics["dataset_quality"] = compute_dataset_quality(self.dataset._url)
        else:
            self.metrics["dataset_quality"] = 0
        self.latencies["dataset_quality_latency"] = int(time.perf_counter_ns() / 1e6 - t)


    def calcCodeQuality(self) -> None:
        target = ""
        if self.code and getattr(self.code, "_url", ""):
            target = self.code._url  # Git URL
        elif self.code and getattr(self.code, "_path_to_cloned", ""):
            target = self.code._path_to_cloned  # local path if you set it elsewhere
        else:
            return
        t = int(time.perf_counter_ns() / 1e6)
        self.metrics["code_quality"] = code_quality(target)
        self.latencies["code_quality_latency"] = int(time.perf_counter_ns() / 1e6 - t)


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
    

# if __name__ == "__main__":
#     model = Model(id = "microsoft/DialoGPT-medium")
#     model.calcMetricsParallel()
#     output = {}
#     output.update(model.metrics)
#     output.update(model.latencies)

#     print(json.dumps(output, indent=4))
    
#     model = Model(id = "deepseek-ai/DeepSeek-R1")
#     model.calcMetricsParallel()
#     output = {}
#     output.update(model.metrics)
#     output.update(model.latencies)

#     print(json.dumps(output, indent=4))

    
