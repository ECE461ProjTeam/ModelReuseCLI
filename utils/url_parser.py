"""
URL Parser for ModelReuseCLI
Handles parsing URL files and creating Model, Code, Dataset objects
"""

import re
from typing import List, Tuple, Dict
from model import Model, Code, Dataset
from apis.hf_client import HFClient


def classify_url(url: str) -> str:
    """
    Classify a URL by its type
    
    Args:
        url (str): The URL to classify
        
    Returns:
        str: 'code', 'dataset', 'model', or 'unknown'
    """
    if not url or not url.strip():
        return 'unknown'
    
    url = url.strip()
    
    # GitHub patterns
    if re.search(r'github\.com', url, re.IGNORECASE):
        return 'code'
    
    # HuggingFace dataset patterns
    if re.search(r'huggingface\.co/datasets/', url, re.IGNORECASE):
        return 'dataset'
    
    # HuggingFace model patterns
    if re.search(r'huggingface\.co/', url, re.IGNORECASE):
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
    if not url:
        return ""
    
    # GitHub pattern: extract repo name
    github_match = re.search(r'github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/.*)?$', url, re.IGNORECASE)
    if github_match:
        owner, repo = github_match.groups()
        return repo.replace('.git', '')
    
    # HuggingFace pattern: extract model/dataset name
    hf_match = re.search(r'huggingface\.co/(?:datasets/)?([^/]+)/([^/]+?)(?:/.*)?$', url, re.IGNORECASE)
    if hf_match:
        namespace, name = hf_match.groups()
        return name
    
    return ""


def extract_id_from_url(url: str) -> str:
    """
    Extract a full ID from a HuggingFace URL
    
    Args:
        url (str): The URL
        
    Returns:
        str: Extracted ID or empty string if extraction fails
    """
    if not url:
        return ""
    
    # HuggingFace pattern: extract model/dataset ID 
    # Handle both formats: namespace/model and standalone model names
    hf_match = re.search(r'huggingface\.co/(?:datasets/)?([^/?]+(?:/[^/?]+)?)', url, re.IGNORECASE)
    if hf_match:
        return hf_match.group(1)
    
    return ""


def populate_code_info(code: Code) -> None:
    """
    Populate Code object with additional information from GitHub API
    
    Args:
        code (Code): Code object to populate
    """
    # Extract name from URL
    code._name = extract_name_from_url(code._url)
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
    dataset._name = extract_name_from_url(dataset._url)
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
    # Extract name and ID from URL
    model.name = extract_name_from_url(model.url)
    model.id = extract_id_from_url(model.url)
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
    dataset_registry = {}  # Track all datasets by name
    models_to_check = []   # Models with blank dataset fields
    hf_client = HFClient() # Create a single client instance
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()

                # Parse the CSV line
                parts = [part.strip() for part in line.split(',')]
                
                # Ensure we have exactly 3 parts
                if len(parts) != 3:
                    print(f"Warning: Line {line_num} does not have exactly 3 columns: {line}")
                    continue
                
                code_link, dataset_link, model_link = parts
                
                # Create Code object only if URL exists
                code = None
                if code_link:
                    code_type = classify_url(code_link)
                    if code_type == 'code':
                        code = Code(code_link)
                        populate_code_info(code)
                    else:
                        print(f"Warning: Code link on line {line_num} is not a GitHub URL: {code_link}")
                
                # Create Dataset object only if URL exists
                dataset = None
                if dataset_link:
                    dataset_type = classify_url(dataset_link)
                    if dataset_type == 'dataset':
                        dataset = Dataset(dataset_link)
                        populate_dataset_info(dataset)
                        dataset_registry[dataset._name] = dataset  # Add to registry
                    else:
                        print(f"Warning: Dataset link on line {line_num} is not a HuggingFace dataset URL: {dataset_link}")
                
                # Create Model object (always required)
                if not model_link:
                    print(f"Warning: Model link is missing on line {line_num}")
                    continue
                
                model_type = classify_url(model_link)
                if model_type != 'model':
                    print(f"Warning: Model link on line {line_num} is not a HuggingFace model URL: {model_link}")
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

        # After parsing, process models with missing datasets
        if models_to_check and dataset_registry:
            print("\nChecking models for dataset links...")
            for model in models_to_check:
                if not model.id:
                    continue
                
                # Collect all datasets mentioned in model metadata
                all_datasets_found = []
                model_info = hf_client.model_info(model.id)
                
                # Check cardData.datasets field
                if model_info.get('cardData') and hasattr(model_info['cardData'], 'datasets'):
                    card_datasets = model_info['cardData'].datasets or []
                    all_datasets_found.extend(card_datasets)
                
                # Check tags for dataset entries
                if model_info.get('tags'):
                    for tag in model_info['tags']:
                        if tag.startswith('dataset:'):
                            all_datasets_found.append(tag[8:])
                
                # Check model card text for dataset mentions
                model_card = hf_client.model_card_text(model.id)
                if model_card:
                    for name in dataset_registry.keys():
                        if re.search(r'\b' + re.escape(name) + r'\b', model_card, re.IGNORECASE):
                            all_datasets_found.append(name)
                
                # Remove duplicates while preserving order
                unique_datasets = []
                for dataset in all_datasets_found:
                    if dataset not in unique_datasets:
                        unique_datasets.append(dataset)
                
                # Find the first dataset that exists in our registry
                valid_datasets = [ds for ds in unique_datasets if ds in dataset_registry]
                
                if valid_datasets:
                    chosen_dataset = valid_datasets[0]
                    print(f"  Model '{model.id}' mentions datasets: {unique_datasets}")
                    if len(valid_datasets) > 1:
                        print(f"  Multiple valid datasets found: {valid_datasets}")
                        print(f"  Linking to first valid dataset: '{chosen_dataset}'")
                    else:
                        print(f"  Linking to dataset: '{chosen_dataset}'")
                    model.linkDataset(dataset_registry[chosen_dataset])
                else:
                    if unique_datasets:
                        print(f"  Model '{model.id}' mentions datasets: {unique_datasets}")
                        print(f"  No mentioned datasets found in registry")
                    else:
                        print(f"  No datasets found in metadata for model '{model.id}'")
                
                
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return [], {}
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return [], {}
    
    return models, dataset_registry


def print_model_summary(models: List[Model], dataset_registry: Dict[str, Dataset]) -> None:
    """
    Print a summary of parsed models and dataset registry for debugging
    
    Args:
        models (List[Model]): List of Model objects
        dataset_registry (Dict[str, Dataset]): Registry of all datasets
    """
    print(f"\nParsed {len(models)} models:")
    
    for i, model in enumerate(models, 1):
        print(f"Model {i}: {model.name}")
        print(f"  URL: {model.url}")
        print(f"  Code: {model.code._name if model.code else 'None (void)'}")
        print(f"  Dataset: {model.dataset._name if model.dataset else 'None (void)'}\n")
    
    print(f"\nDataset Registry ({len(dataset_registry)} datasets):")
    for name, dataset in dataset_registry.items():
        print(f"  {name}: {dataset._url}")
    print()


if __name__ == "__main__":
    # Test the URL parser
    # Note: Run this from the project root directory: python3 -m utils.url_parser
    test_content = """https://github.com/google-research/bert,https://huggingface.co/datasets/bookcorpus/bookcorpus,https://huggingface.co/google-bert/bert-base-uncased
,,https://huggingface.co/parvk11/audience_classifier_model
,,https://huggingface.co/openai/whisper-tiny"""
    
    with open("test_input.txt", "w") as f:
        f.write(test_content)
    
    print("Testing URL parser standalone...")
    try:
        models, dataset_registry = parse_URL_file("test_input.txt")
        print_model_summary(models, dataset_registry)
    except Exception as e:
        print(f"Test failed: {e}")
        print("Note: Run with 'python3 -m utils.url_parser' from project root")
    finally:
        # Clean up test file
        import os
        if os.path.exists("test_input.txt"):
            os.remove("test_input.txt")