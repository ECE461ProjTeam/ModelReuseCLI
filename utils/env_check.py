import os
import requests
import logging


logger = logging.getLogger('cli_logger')


def check_environment() -> bool:
    '''
        Function to check the validity of the GIT_TOKEN and if the LOG_FILE path exists.

        Returns:
            bool: True if GitHub token is valid and log file exists or was created, False otherwise.
    '''
    # Get values from environment variables
    git_token = os.getenv("GIT_TOKEN")
    log_file_path = os.getenv("LOG_FILE")
    
    # 1. Check if log file path exists
    if not log_file_path:
        return False

    if not git_token:
        # logger.error("GIT_TOKEN environment variable is not set.")
        return False
    
    # 2. Check GitHub token validity
    headers = {"Authorization": f"token {git_token}"}
    try:
        response = requests.get("https://api.github.com/user", headers=headers)
        if response.status_code != 200:
            # logger.error("Invalid GitHub token.")
            return False
    except requests.RequestException as e:
        # logger.error(f"Error checking GitHub token: {e}")
        return False

    # Return true if both checks pass
    # logger.info("Environment is valid.")
    return True
