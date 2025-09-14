#!/usr/bin/env python3
import argparse
import sys
import os
from typing import Dict

from apis.gemini import prompt_gemini
from apis.hf_client import HFClient, resolve_hf_token


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
        url_file = args.option
        # add processing here
        pass


if __name__ == "__main__":
    main()

