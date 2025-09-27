#!/usr/bin/env python3
import argparse
import sys
import os
from typing import Dict
import json
import logging

from utils.url_parser import parse_URL_file, print_model_summary
from utils.logger import setup_logger
from utils.env_check import check_environment
from typing import Dict
from apis.gemini import *
from apis.purdue_genai import *
from apis.hf_client import resolve_hf_token

# For Testing: Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # reads .env file into environment variables
# REMOVE ABOVE LINES IN PRODUCTION

setup_logger()  # configure logging once
logger = logging.getLogger('cli_logger')


def main():
    if not check_environment():
        sys.exit(1)
    
    logger.info("Starting ModelReuseCLI...")
    parser = argparse.ArgumentParser(description="ModelReuseCLI main entry point")
    parser.add_argument('option', type=str, help="URL_FILE")
    args = parser.parse_args()

    if args.option == "test":
        logger.info("Running tests...")
        pass
    else:
        # Treat as URL_FILE path
        url_file = args.option
        
        # Check if the file exists
        if not os.path.exists(url_file):
            logger.error(f"Error: File '{url_file}' not found.")
            sys.exit(2)  # More specific error code for file not found
        
        # Parse the URL file and create Model objects
        models, dataset_registry = parse_URL_file(url_file)
        
        if not models:
            logger.error("No models found in the file.")
            sys.exit(3)  # Specific error code for no models found
        
        # Print summary of parsed models
        print_model_summary(models, dataset_registry)
        
        logger.debug("\nURL parsing complete! Created:")
        logger.debug(f"  - {len(models)} Model objects")
        logger.debug(f"  - {len(dataset_registry)} unique datasets")
        logger.info("Objects ready for metric calculation teams.")
        for model in models:
            print(json.dumps(model.evaluate()))
            logger.info(f"Successfully evaludated model {model.name}!")


if __name__ == "__main__":
    main()