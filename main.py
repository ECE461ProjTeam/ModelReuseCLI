#!/usr/bin/env python3
import argparse
import sys
import os
from typing import Dict

from apis.gemini import prompt_gemini
from apis.hf_client import HFClient, resolve_hf_token
from utils.url_parser import parse_URL_file, print_model_summary


def get_api_keys() -> Dict[str, str]:
    """
    Compile all available API keys into a dictionary.

    Returns:
        keys (dict): Dictionary of API keys with service names as keys
    """
    keys = {}

    try:
        with open('gemini_key.txt', 'r') as file:
            gemini_api_key = file.readline().strip()
        keys.update({'gemini': gemini_api_key})
    except FileNotFoundError:
        print("API key file not found. Please create 'gemini_key.txt' with your Gemini API key.")
        sys.exit(-1)

    # Hugging Face
    hf_token = resolve_hf_token()
    if hf_token:
        keys.update({'huggingface': hf_token})

    return keys


def main():
    parser = argparse.ArgumentParser(description="ModelReuseCLI main entry point")
    parser.add_argument('option', type=str, help="'install', 'test', or URL_FILE")
    args = parser.parse_args()

    if args.option == "test":
        print("Running tests...")
        pass
    else:
        # Treat as URL_FILE path
        url_file = args.option
        
        # Check if the file exists
        if not os.path.exists(url_file):
            print(f"Error: File '{url_file}' not found.")
            sys.exit(2)  # More specific error code for file not found
        
        # Parse the URL file and create Model objects
        models, dataset_registry = parse_URL_file(url_file)
        
        if not models:
            print("No models found in the file.")
            sys.exit(3)  # Specific error code for no models found
        
        # Print summary of parsed models
        print_model_summary(models, dataset_registry)
        
        print("\nURL parsing complete! Created:")
        print(f"  - {len(models)} Model objects")
        print(f"  - {len(dataset_registry)} unique datasets")
        print("Objects ready for metric calculation teams.")


if __name__ == "__main__":
    main()