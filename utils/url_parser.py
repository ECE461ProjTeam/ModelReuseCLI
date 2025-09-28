"""
URL Parser for ModelReuseCLI
Handles parsing URL files and creating Model, Code, Dataset objects
"""

import re
from typing import List, Tuple, Dict
from model import Model, Code, Dataset
import logging

logger = logging.getLogger('cli_logger')

import apis.hf_client as hf_client



def classify_url(url: str) -> str:
    """
    Classify a URL by its type
    
    Args:
        url (str): The URL to classify
        
    Returns:
        str: 'code', 'dataset', 'model', or 'unknown'
    """
    logger.debug(f"Classifying URL: {url}")
    if not url or not url.strip():
        return 'unknown'
    
    url = url.strip()
    
    # GitHub patterns
    # hugging face code space ex: huggingface.co/spaces/abidlabs/en2fr
    # GitHub code pattern
    if re.search(r'github\.com', url, re.IGNORECASE):
        return 'github'

    # GitLab code pattern
    if re.search(r'gitlab\.[^/]+', url, re.IGNORECASE):
        return 'gitlab'

    # HuggingFace Spaces (code) pattern
    if re.search(r'huggingface\.co/spaces/', url, re.IGNORECASE):
        return 'hfspace'
    
    # HuggingFace dataset patterns
    if re.search(r'huggingface\.co/datasets/', url, re.IGNORECASE):
        return 'dataset'
    
    # HuggingFace model patterns (exclude spaces and datasets explicitly)
    if (re.search(r'huggingface\.co/', url, re.IGNORECASE) and 
        not re.search(r'huggingface\.co/(spaces|datasets)/', url, re.IGNORECASE)):
        return 'model'
    
    return 'unknown'


def extract_name_from_url(url: str) -> str:
    """
    Extract a name from a URL
    
    Args:
        url (str): The URL
        
    Returns:
        str: Extracted name or empty string if extraction fails
    """
    logger.debug(f"Extracting name from URL: {url}")
    if not url:
        return ""
    
    # code pattern: github/gitlab/hfspace
    github_match = re.search(r'github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/.*)?$', url, re.IGNORECASE)
    if github_match:
        owner, repo = github_match.groups()
        return repo.replace('.git', '')

    gitlab_match = re.search(r'(?:git@|https?://)gitlab\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+)(?:\.git)?$', url, re.IGNORECASE)
    if gitlab_match:
        return gitlab_match.group('repo')

    hfcode_match = re.search(r'^https?://(?:www\.)?huggingface\.co/spaces/(?P<owner>[^/]+)/(?P<space>[^/]+)(?:/.*)?$', url, re.IGNORECASE)
    if hfcode_match:
        return hfcode_match.group('space')

    # HuggingFace pattern: extract model/dataset name
    hf_match = re.search(r'huggingface\.co/(?:datasets/)?([^/]+)/([^/]+?)(?:/.*)?$', url, re.IGNORECASE)
    if hf_match:
        namespace, name = hf_match.groups()
        return namespace, name
    
    return ""


def populate_code_info(code: Code, code_type: str) -> None:
    """
    Populate Code object with additional information from GitHub API
    
    Args:
        code (Code): Code object to populate
    """
    # Extract name from URL
    code._name = extract_name_from_url(code._url)
    code.type = code_type
    # TODO: Add GitHub API calls to populate metadata
    # Example implementation for metrics teams:
    # from apis.git_api import get_contributors, get_commit_history
    # owner, repo = extract_github_owner_repo(code._url)
    # code._metadata = {
    #     'contributors': get_contributors(owner, repo),
    #     'commits': get_commit_history(owner, repo),
    #     'bus_factor_data': {...}
    # }


def populate_dataset_info(dataset: Dataset) -> None:
    """
    Populate Dataset object with additional information from HuggingFace API
    
    Args:
        dataset (Dataset): Dataset object to populate
    """
    # Extract name from URL
    owner, name = extract_name_from_url(dataset._url)
    dataset._name = owner + "/" + name
    # TODO: Add HuggingFace API calls to populate metadata
    # Example implementation for metrics teams:
    # from apis.hf_client import HFClient
    # hf_client = HFClient()
    # dataset._metadata = {
    #     'dataset_info': hf_client.dataset_info(dataset._name),
    #     'downloads': ..., 'license': ..., 'size': ...
    # }


def populate_model_info(model: Model) -> None:
    """
    Populate Model object with additional information from HuggingFace API
    
    Args:
        model (Model): Model object to populate
    """
    # Extract name from URL
    owner, model.name = extract_name_from_url(model.url)
    model.id = owner + "/" + model.name
    # TODO: Add HuggingFace API calls to populate hfAPIData
    # Example implementation for metrics teams:
    # from apis.hf_client import HFClient
    # hf_client = HFClient()
    # model.hfAPIData = {
    #     'model_info': hf_client.model_info(model.name),
    #     'downloads': ..., 'license': ..., 'size': ...
    # }


def parse_URL_file(file_path: str) -> Tuple[List[Model], Dict[str, Dataset]]:
    """
    Parse a URL file and create Model objects with linked Code and Dataset objects.
    Also return a registry of all datasets encountered.
    
    Args:
        file_path (str): Path to the URL file
        
    Returns:
        Tuple[List[Model], Dict[str, Dataset]]: List of Model objects and dataset registry
    """
    models = []
    models_to_check = []  # Models with empty dataset links
    dataset_registry = {}  # Track all datasets by name
    
    try:
        logger.info(f"Parsing URL file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()

                # Parse the CSV line
                parts = [part.strip() for part in line.split(',')]
                
                # Ensure we have exactly 3 parts
                if len(parts) != 3:
                    logger.warning(f"Warning: Line {line_num} does not have exactly 3 columns: {line}")
                    continue
                
                code_link, dataset_link, model_link = parts
                
                # Create Code object only if URL exists
                code = None
                if code_link:
                    code_type = classify_url(code_link)
                    # print(code_type)
                    if code_type == 'github' or code_type == 'gitlab' or code_type == 'hfspace':
                        code = Code(code_link)
                        populate_code_info(code, code_type)
                        populate_code_info(code, code_type)
                    else:
                        logger.warning(f"Warning: Code link on line {line_num} is not a GitHub URL: {code_link}")
                
                # Create Dataset object only if URL exists
                dataset = None
                if dataset_link:
                    dataset_type = classify_url(dataset_link)
                    if dataset_type == 'dataset':
                        dataset = Dataset(dataset_link)
                        populate_dataset_info(dataset)
                        dataset_registry[dataset._name] = dataset  # Add to registry
                    else:
                        logger.warning(f"Warning: Dataset link on line {line_num} is not a HuggingFace dataset URL: {dataset_link}")
                
                # Create Model object (always required)
                if not model_link:
                    logger.warning(f"Warning: Model link is missing on line {line_num}")
                    continue
                
                model_type = classify_url(model_link)
                if model_type != 'model':
                    logger.warning(f"Warning: Model link on line {line_num} is not a HuggingFace model URL: {model_link}")
                    continue
                
                # Create and populate Model object
                model = Model(model_link)
                populate_model_info(model)
                
                # Link Code and Dataset to Model (can be None/void)
                if code:
                    model.linkCode(code)
                
                if dataset:
                    model.linkDataset(dataset)
                else:
                    # If no dataset, add to the list to check later
                    models_to_check.append(model)
                
                models.append(model)


        # After parsing, process models with missing datasets using Gemini
        if models_to_check and dataset_registry:
            logger.debug(f"Checking empty dataset link for {len(models_to_check)} models...")
            for model in models_to_check:
                if not model.id:
                    logger.debug(f"  Skipping model with no ID")
                    continue
            
            # Debug: Print dataset registry contents
            logger.debug(f"Dataset registry contains {len(dataset_registry)} datasets:")
            for registry_key, dataset_obj in dataset_registry.items():
                logger.debug(f"  Registry key: '{registry_key}' -> URL: {dataset_obj._url}")
            
            for model in models_to_check:
                if not model.id:
                    logger.debug(f"  Skipping model with no ID")
                    continue

                logger.debug(f"  Analyzing model '{model.id}'...")
                model_card = hf_client.HFClient().model_card_text(model.id)
                
                chosen_dataset = find_empty_dataset(model.id, model_card, dataset_registry)
                
                if chosen_dataset:
                    logger.debug(f"  Linking '{model.id}' to dataset: '{chosen_dataset}'")
                    model.linkDataset(dataset_registry[chosen_dataset])
                else:
                    logger.debug(f"  No relevant dataset found for '{model.id}'")
                    pass
                
                
    except FileNotFoundError:
        logger.error(f"Error: File {file_path} not found")
        return [], {}
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return [], {}
    
    return models, dataset_registry


def find_empty_dataset(model_id: str, model_card: str, dataset_registry: Dict[str, Dataset]) -> str:    
    """
    Use model ID and model card text to find a relevant dataset from the registry
    
    Args:
        model_id (str): Model identifier
        model_card (str): Model card text
        dataset_registry (Dict[str, Dataset]): Registry of all datasets

    Returns:
        str: The name of the chosen dataset, or an empty string if none found
    """
    
    if not dataset_registry:
        return ""

    training_data_section = extract_training_data_section(model_card)
    logger.debug(f"  Training data section found: {bool(training_data_section)}")
    if training_data_section:
        logger.debug(f"  Training data content (first 300 chars): '{training_data_section[:300]}'")

    if not training_data_section:
        return ""

    # Check for placeholder text that indicates no real training data info
    training_data_lower = training_data_section.lower().strip()
    if (training_data_lower.startswith("[more information needed]") or 
        training_data_lower.startswith("<!-- this should link to a dataset")):
        logger.debug(f"  Training data section contains placeholder text, skipping LLM analysis")
        return ""

    from utils.prompt_key import get_prompt_key
    api_keys = get_prompt_key()
    
    if "purdue_genai" in api_keys:
        # Use Purdue GenAI
        from apis.purdue_genai import prompt_purdue_genai
        
        available_datasets = list(dataset_registry.keys())
        
        prompt = f"""Match training data to dataset. Return only the exact dataset name or "NONE".

Available datasets: {', '.join(available_datasets)}

Training data: {training_data_section[:500]}

Return format: Just the dataset name (e.g., "bookcorpus/bookcorpus") or "NONE". No explanation."""

        logger.debug("Sending training data section to LLM...")
        try:
            response = prompt_purdue_genai(prompt, api_keys["purdue_genai"])
            response = response.strip()
            
            # Check if response matches any dataset in registry
            if response in available_datasets or response in dataset_registry:
                logger.debug(f"  Purdue GenAI matched '{model_id}' to dataset: '{response}'")
                return response
            
            # Try fuzzy matching for cases like "bookcorpus" -> "bookcorpus/bookcorpus"
            for registry_key in available_datasets:
                if '/' in registry_key:
                    dataset_part = registry_key.split('/')[-1]
                    if response.lower() == dataset_part.lower():
                        logger.debug(f"  Purdue GenAI fuzzy matched '{response}' to '{registry_key}'")
                        return registry_key
            
            logger.debug(f"  Purdue GenAI response '{response}' not in registry")
            
        except Exception as e:
            logger.error(f"  Error calling Purdue GenAI: {e}")

    return ""


def extract_training_data_section(model_card: str) -> str:
    """
    Extract the 'Training data' section from a model card.

    Args:
        model_card (str): The full text of the model card.

    Returns:
        str: The content of the 'Training data' section, or an empty string if not found.
    """
    import re 

    # Locate the "Training data" section
    training_data_match = re.search(r"## Training data\n\n(.+?)(\n##|\Z)", model_card, re.DOTALL | re.IGNORECASE)
    if training_data_match:
        return training_data_match.group(1).strip()

    return ""


def print_model_summary(models: List[Model], dataset_registry: Dict[str, Dataset]) -> None:
    """
    Print a summary of parsed models and dataset registry for debugging
    
    Args:
        models (List[Model]): List of Model objects
        dataset_registry (Dict[str, Dataset]): Registry of all datasets
    """
    logger.debug(f"\nParsed {len(models)} models:")
    
    for i, model in enumerate(models, 1):
        logger.debug(f"Model {i}: {model.name}")
        logger.debug(f"  URL: {model.url}")
        logger.debug(f"  Code: {model.code._name if model.code else 'None (void)'}")
        logger.debug(f"  Dataset: {model.dataset._name if model.dataset else 'None (void)'}\n")
    
    logger.debug(f"\nDataset Registry ({len(dataset_registry)} datasets):")
    for name, dataset in dataset_registry.items():
        logger.debug(f"  {name}: {dataset._url}")


# if __name__ == "__main__":
#     # Test the URL parser with a model that has an empty dataset link
#     logger.debug("Testing URL parser with dataset population logic...")

#     # Create a dataset registry with a known dataset
#     dataset_registry = {
#         "bookcorpus/bookcorpus": Dataset("https://huggingface.co/datasets/bookcorpus/bookcorpus")
#     }
#     populate_dataset_info(dataset_registry["bookcorpus/bookcorpus"])

#     # Create a model with an empty dataset link
#     model = Model("https://huggingface.co/google-bert/bert-base-uncased")
#     populate_model_info(model)

#     # Simulate a model card with training data mentioning "BookCorpus"
#     model_card_text = """
#     ## Training data

#     The model was trained on the BookCorpus dataset and other datasets.
#     """

#     # Use the find_empty_dataset logic to populate the dataset link
#     chosen_dataset = find_empty_dataset(model.id, model_card_text, dataset_registry)
#     if chosen_dataset:
#         logger.debug(f"Dataset chosen for model {model.id}: {chosen_dataset}")
#         model.linkDataset(dataset_registry[chosen_dataset])
#     else:
#         logger.debug(f"No dataset found for model {model.id}")

#     # Print the results
#     print_model_summary([model], dataset_registry)