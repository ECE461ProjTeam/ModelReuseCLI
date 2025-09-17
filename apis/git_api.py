import os
import requests
import time
from typing import List, Dict, Any, Optional


def check_git_token() -> Optional[str]:
    '''
    Check for a GitHub token in the environment variables and return it if found.
    If not found, check for a txt file named 'git_token.txt' in the current directory and read the token from there.
    If still not found, return None.
    '''
    token = os.getenv("GIT_TOKEN")
    if not token:
        try:
            with open("git_token.txt", "r") as f:
                token = f.read().strip()
        except FileNotFoundError:
            return None
    return token


def make_request(url: str, headers: Dict[str, str], max_time: int = 60) -> requests.Response:
    '''
    Make a GET request to the specified URL with the provided headers, handling rate limiting and retries
    using exponential backoff.
    Default wait time starts at 1 second and doubles with each retry up to a maximum of 60 seconds.

    Args:
        url (str): The URL to send the GET request to.
        headers (Dict[str, str]): Headers to include in the request.
        max_time (int): Maximum time to wait for retries in seconds.
    Returns:
        requests.Response: The response object from the GET request.'''
    wait_time = 1

    while wait_time <= 60:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response
        elif response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers and response.headers['X-RateLimit-Remaining'] == '0':
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + wait_time))
            sleep_time = max(reset_time - time.time(), wait_time)
            print(f"Rate limit exceeded. Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)
        else:
            print(f"Request failed with status code {response.status_code}. Retrying in {wait_time} seconds.")
            time.sleep(wait_time)
            wait_time *= 2

    raise Exception(f"Failed to fetch data from {url} after multiple attempts.")


def set_git_headers() -> Dict[str, str]:
    '''Set headers for GitHub API requests, specifically authorization if a token is set in the environment variables.'''
    token = check_git_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_contributors(owner: str, repo: str) -> List[Dict[str, Any]]:
    """
    Retrieve contributors from a GitHub repository.

    Args:
        owner (str): Repository owner (e.g., "octocat")
        repo (str): Repository name (e.g., "hello-world")

    Returns:
        list: A list of contributor objects (dicts) from the GitHub API
    """
    headers = set_git_headers()
    url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
    response = make_request(url, headers)
    return response.json()


def get_commit_history(owner: str, repo: str) -> List[Dict[str, Any]]:
    """
    Retrieve commits from a GitHub repository.
    
    Args:
        owner (str): Repository owner (e.g., "octocat")
        repo (str): Repository name (e.g., "hello-world")
    
    Returns:
        list: A list of commit objects (dicts) from the GitHub API
    """
    headers = set_git_headers()
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    response = make_request(url, headers)
    return response.json()


if __name__ == "__main__":
    # Sample Output
    owner = "ECE461ProjTeam"
    repo = "ModelReuseCLI"
    contributers = get_contributors(owner, repo)
    print("Contributors:")
    for contributor in contributers:
        print(f"Contributor: {contributor['login']} - Contributions: {contributor['contributions']}")
    commits = get_commit_history(owner, repo)
    print("\nCommits:")
    for commit in commits:
        print(f"Commit: {commit['sha']} - {commit['commit']['message']}")
        print(f"Author: {commit['commit']['author']['name']} <{commit['commit']['author']['email']}>")
