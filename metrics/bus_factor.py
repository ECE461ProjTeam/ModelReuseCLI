from apis import git_api


def bus_factor(owner: str, repo: str) -> float:
    """
    Calculate the bus factor of a repository.
    The bus factor is defined as the minimum number of developers that need to be incapacitated
    (e.g., hit by a bus) before the project is in serious trouble.

    Sort contributors by commits, add them up until you cover at least 50% of total commits.
    The bus factor is then the number of contributors needed to reach that 50% threshold divided by
    the total number of contributors.

    Args:
        owner (str): The owner of the repository.
        repo (str): The repository identifier.

    Returns:
        float: The bus factor of the repository. [0-1]
    """
    contributors = git_api.get_contributors(owner, repo)
    # Handle edge cases
    if not contributors:
        return 0
    elif len(contributors) == 1:
        return 1

    bus_factor = 0
    total_commits = sum(contributor['contributions'] for contributor in contributors)
    sorted_contributors = sorted(contributors, key=lambda x: x['contributions'], reverse=True)
    cumulative_commits = 0
    for contributor in sorted_contributors:
        cumulative_commits += contributor['contributions']
        bus_factor += 1
        if cumulative_commits >= total_commits / 2:
            break
    bus_factor /= len(contributors)

    return bus_factor


# Testing only - remove upon integration
if __name__ == "__main__":
    owner = "freeCodeCamp"
    repo = "freeCodeCamp"

    print(f"Bus Factor for {owner}/{repo}: {bus_factor(owner, repo)}")
