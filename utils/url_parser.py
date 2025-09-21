import re
from typing import List, Tuple, Dict
from model import Model, Code, Dataset
from apis.hf_client import HFClient
from apis.gemini import get_gemini_key, prompt_gemini


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
        return "", ""
    
    # GitHub pattern: extract repo name
    github_match = re.search(r'github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/.*)?$', url, re.IGNORECASE)
    if github_match:
        owner, repo = github_match.groups()
        return owner, repo.replace('.git', '')
    
    # HuggingFace pattern: extract model/dataset name
    hf_match = re.search(r'huggingface\.co/(?:datasets/)?([^/]+)/([^/]+?)(?:/.*)?$', url, re.IGNORECASE)
    if hf_match:
        namespace, name = hf_match.groups()
        return namespace, name

    return "", ""


def find_best_dataset_with_gemini(model_id: str, model_card: str, dataset_registry: Dict[str, Dataset]) -> str:
    """
    Use Gemini to intelligently find the most relevant dataset for a model from the entire registry
    
    Args:
        model_id (str): The model ID
        model_card (str): The model card text
        dataset_registry (Dict[str, Dataset]): Registry of all available datasets
    
    Returns:
        str: The name of the most relevant dataset, or empty string if none found
    """
    if not dataset_registry:
        return ""
    
    # Get Gemini API key
    api_key = get_gemini_key()
    if not api_key:
        print(f"  Warning: No Gemini API key found, cannot analyze datasets")
        return ""
    
    # Create dataset descriptions for Gemini
    dataset_descriptions = []
    for dataset_name, dataset_obj in dataset_registry.items():
        dataset_url = dataset_obj._url
        dataset_descriptions.append(f"- {dataset_name}: {dataset_url}")
    
    # Create prompt for Gemini with confidence scoring
    prompt = f"""You are helping to match machine learning models with their most relevant training datasets.

Model ID: {model_id}

Model Description (from hugging face model card):
{model_card[:10000] if model_card else "No model card available"}

Available datasets in registry:
{chr(10).join(dataset_descriptions)}

Task: Analyze the model description and determine which dataset from the registry would be most likely used for training or fine-tuning this model. Consider:
1. The model's purpose and intended use case
2. The type of task (text classification, language modeling, etc.)
3. Domain relevance (sentiment analysis, question answering, etc.)
4. Which dataset would make the most sense for this specific model

If you find a relevant dataset from the list that you are reasonably confident about (6+ out of 10 confidence), respond with ONLY the dataset name (e.g., "glue" or "bookcorpus").
If no dataset from the list seems relevant or you are not confident enough, respond with exactly: "NONE"

Response:"""

    try:
        response = prompt_gemini(prompt, api_key)
        if response:
            # Clean the response
            cleaned_response = response.strip().lower().replace('"', '').replace("'", "")
            
            # Check if Gemini said no relevant dataset
            if "none" in cleaned_response:
                # print(f"  Gemini found no relevant dataset in registry")
                return ""
            
            # Check for exact match first
            for dataset_name in dataset_registry.keys():
                if dataset_name.lower() == cleaned_response:
                    # print(f"  Gemini selected: '{dataset_name}' (exact match)")
                    return dataset_name
            
            # Check for partial matches (dataset name found in Gemini's response)
            for dataset_name in dataset_registry.keys():
                if dataset_name.lower() in cleaned_response:
                    # print(f"  Gemini selected: '{dataset_name}' (partial match)")
                    return dataset_name
            
            # print(f"  Gemini response '{cleaned_response}' not recognized in registry")
        else:
            # print(f"  Gemini request failed")
            pass
    except Exception as e:
        print(f"  Error with Gemini analysis: {e}")
        pass
    
    return ""


def populate_code_info(code: Code) -> None:
    """
    Populate Code object with additional information from GitHub API
    
    Args:
        code (Code): Code object to populate
    """
    # Extract name from URL
    owner, name = extract_name_from_url(code._url)
    code._name = name
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
    dataset._name = name
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

        # After parsing, process models with missing datasets using Gemini
        if models_to_check and dataset_registry:
            # print(f"\nUsing Gemini to find relevant datasets for {len(models_to_check)} models...")
            for model in models_to_check:
                if not model.id:
                    # print(f"  Skipping model with no ID")
                    continue
                
                # print(f"  Analyzing model '{model.id}'...")
                
                # Get model card for analysis
                model_card = hf_client.model_card_text(model.id)
                
                # Use Gemini to find the best dataset from our registry
                chosen_dataset = find_best_dataset_with_gemini(model.id, model_card, dataset_registry)
                
                if chosen_dataset:
                    # print(f"  Linking '{model.id}' to dataset: '{chosen_dataset}'")
                    model.linkDataset(dataset_registry[chosen_dataset])
                else:
                    # print(f"  No relevant dataset found for '{model.id}'")
                    pass
                
                
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