from huggingface_hub import HfFileSystem
from apis.hf_client import HFClient
from apis.git_api import *


def dataset_and_code_score(dataset_id: str, code_id: str, code_type: str) -> float:
    fs = HfFileSystem()
    client = HFClient()
    headers = set_git_headers()
    
    score = 0
    code_files = []
    
    if (dataset_id):
        score += 0.125
        try:
            dataset_files_raw = fs.ls("datasets/" + dataset_id)
            ds_files = [file["name"][file["name"].index(dataset_id) + len(dataset_id) + 1: ] for file in dataset_files_raw]
            if "README.md" in ds_files:
                score += 0.25
        except:
            pass

    if(code_id):
        score += 0.125
        if code_type == "github":
            files_request = make_request(f"https://api.github.com/repos/{code_id}/git/trees/master?recursive=1", headers).json()
            code_files = [item["path"] for item in files_request["tree"]]
        elif code_type == "gitlab":
            files_request = make_request(f"https://gitlab.example.com/api/v4/projects/{code_id}/repository/tree", headers).json()
            code_files = []
            # idk how to do requests to gitlab!!
        
        if "requirements.txt" in code_files:
            score += 0.1
        for file in code_files:
            if "test" in file:
                score += 0.15
                break
        if "README.md" in code_files:
            score += 0.25
    
    return score

