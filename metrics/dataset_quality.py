from utils.prompt_key import get_prompt_key
from apis.hf_client import HFClient
from apis.gemini import prompt_gemini
from apis.purdue_genai import prompt_purdue_genai
import re
import logging


logger = logging.getLogger('cli_logger')


def compute_dataset_quality(dataset_url: str) -> float:
    '''
    Compute the quality score of 1 dataset by checking the size, descriptions, and other relevant info.
    Ask Gemini/Purdue GenAI Studio to score the datasets based on the information given. Score from 0 to 1.

    Args:
        dataset_id (str): The dataset identifier on Hugging Face
    Returns:
        score (float): The dataset quality score from 0 to 1
    '''
    # hf_client = HFClient()
    
    # try:
    #     dataset_card = hf_client.dataset_card_text(dataset_id)
    # except Exception:
    #     logger.warning(f"Unable to fetch dataset card for {dataset_id}. It may not exist or it is not on Hugging Face.")
    #     return 0.0

    # # If dataset card is , return 0.0
    # if not dataset_card:
    #     return 0.0

    api_key = get_prompt_key()
    if not api_key:
        return 0.0

    dataset_quality_prompt = f"""Analyze the following dataset information by opening the link below and reading the dataset card
{dataset_url}
Based on the information, evaluate the dataset's quality on a scale from 0.0 to 1.0. Consider:
- Completeness and clarity of the documentation. [score out of 0.4 points]
- Size and diversity of the dataset. [score out of 0.2 points]
- Presence of data splits (train, validation, test). [score out of 0.2 points]
- Clear licensing and citation information. [score out of 0.2 points]
In your explanation, you must ALWAYS specify how many points were scored on each criteria (out of the total points listed).
The first line of your answer must be in the following format: <score between 0-1>: <explanation and score breakdown>
DO NOT deviate from the format."""

    num_retries = 0
    max_retries = 3
    score = 0.0

    while num_retries < max_retries:
        try:
            if 'purdue_genai' in api_key:
                dq_check = prompt_purdue_genai(dataset_quality_prompt, api_key['purdue_genai'])
            elif 'gemini' in api_key:
                dq_check = prompt_gemini(dataset_quality_prompt, api_key['gemini'])
        except:
            logger.error("compute dataset quality: Could not get LLM response")
            break
        match = re.match(r"([0-1](?:\.\d+)?):(.*)", dq_check, re.DOTALL)
        if match:
            score = float(match.group(1))
            explanation = match.group(2).strip()
            logger.debug(f"Dataset Quality Explanation for {dataset_url}: {explanation}")
            break
        num_retries += 1
    else:
        logger.error("Could not parse the dataset quality score from the response.")
        raise ValueError("Could not parse the dataset quality score from the response.")
    return score

# if __name__ == "__main__":
#     # Example usage
#     print(compute_dataset_quality("rajpurkar/squad"))