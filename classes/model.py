import threading
import time


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
    def __init__(self, url: str) -> None:
        self.url = url
        self.name: str = ""
        self.code = None  # instance of Code class
        self.dataset = None  # instance of Dataset class
        self.metadata = {}
        self.metrics = {"ramp_up_time": 0, "bus_factor": 0, "performance_claims": 0, "license": 0,
                        "size_score": 0, "dataset_and_code_score": 0, "dataset_quality": 0, "code_quality": 0}
        self.netScore = 0.0
        self.hfAPIData = {}
        self.gitAPIData = {}

    def calcMetricsParallel(self) -> None:
        threads = []
        funcs = {
            "ramp_up_time": self.calcRampUp, "bus_factor": self.calcBusFactor,
            "performance_claims": self.calcPerformanceClaims, "license": self.calcLicense,
            "size_score": self.calcSize, "dataset_and_code_score": self.calcDatasetCode,
            "dataset_quality": self.calcDatasetQuality, "code_quality": self.calcCodeQuality
        }
        for key in funcs:
            t = threading.Thread(target=funcs[key])
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def calcSize(self) -> None:
        self.metrics["size_score"] = 1

    def calcRampUp(self) -> None:
        self.metrics["ramp_up_time"] = 1

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

    def linkCode(self, code: Code):
        self.code = code


if __name__ == "__main__":
    model = Model("url")
    t = time.time()
    model.calcMetricsParallel()
    print(model.metrics)
    print(time.time() - t)
