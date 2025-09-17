#!/usr/bin/env python3
import argparse
import sys
import os

from utils.url_parser import parse_URL_file, print_model_summary


def main():
    parser = argparse.ArgumentParser(description="ModelReuseCLI main entry point")
    parser.add_argument('option', type=str, help="'install', 'test', or URL_FILE")
    args = parser.parse_args()

    if args.option == "test":
        print("Running tests...")
        pass
    elif args.option == "install":
        print("Installing dependencies...")
        pass
    else:
        # Treat as URL_FILE path
        url_file = args.option
        
        # Check if the file exists
        if not os.path.exists(url_file):
            print(f"Error: File '{url_file}' not found.")
            sys.exit(2)  # More specific error code for file not found
        
        # Parse the URL file and create Model objects
        print(f"Processing URL file: {url_file}")
        models, dataset_registry = parse_URL_file(url_file)
        
        if not models:
            print("No models found in the file.")
            sys.exit(3)  # Specific error code for no models found
        
        # Print summary of parsed models
        print_model_summary(models, dataset_registry)
        
        print("\nURL parsing complete! Created:")
        print(f"  - {len(models)} Model objects")
        print(f"  - {len(dataset_registry)} unique datasets")
        print("Objects are ready for metric calculation")


if __name__ == "__main__":
    main()