from apis.hf_client import HFClient
from math import log10
from typing import Dict

#Changed raspberry pi limit to 1B
PLATFORM_SIZE_LIMITS = {"raspberry_pi": 1, "jetson_nano": 7, "desktop_pc": 34, "aws_server": 1200}
LOWER_SIZE_LIMIT = 0.117 

def size_score(model_id: str) -> Dict[str, float]:
    """Calculate a size-based score for a given model.

    Args:
        model_id (str): The id of the model to evaluate.
    Returns:
        dict: A score between 0 and 1 based on the model's size & platform.
    """
    client = HFClient()
    model_info = client.model_info(model_id)
    # if 'safetensors' not in model_info.keys():
    #     return {plat: 0 for plat in PLATFORM_SIZE_LIMITS}
    if 'safetensors' not in model_info or model_info['safetensors'] is None or model_info['safetensors'].total is None:
        return {plat: 0 for plat in PLATFORM_SIZE_LIMITS}
    parameters = model_info['safetensors'].total/1e9
    result = PLATFORM_SIZE_LIMITS
    for plat in PLATFORM_SIZE_LIMITS:
        if parameters > PLATFORM_SIZE_LIMITS[plat] or parameters < LOWER_SIZE_LIMIT:
            result[plat] = 0
        else:
            result[plat] = round(log10(parameters/LOWER_SIZE_LIMIT)/log10(PLATFORM_SIZE_LIMITS[plat]/LOWER_SIZE_LIMIT), ndigits=3)
        
    return result
    
if __name__ == "__main__":
    
    model_id = "deepseek-ai/DeepSeek-R1"
    score = size_score(model_id)
    print(score)

    # Test case for a model with safetensors=None
    model_id = "facebook/blenderbot-400M-distill"
    score = size_score(model_id)
    print(f"Score for model with no safetensors ({model_id}): {score}")