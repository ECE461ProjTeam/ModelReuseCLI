import subprocess
import os

def clone_with_isogit(repo_url: str, local_dir: str = "./models") -> None:
    os.makedirs(local_dir, exist_ok=True)
    result = subprocess.run(
        ["node", "clone.js", repo_url, local_dir],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Clone failed:\n{result.stderr}")
    print(result.stdout) # remove upon integration


# For testing purposes only - delete upon integration
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python clone_bridge.py <repo_url> [local_dir]")
        sys.exit(1)
    
    repo_url = sys.argv[1]
    local_dir = sys.argv[2] if len(sys.argv) > 2 else "./models"
    
    try:
        clone_with_isogit(repo_url, local_dir)
    except Exception as e:
        print(e)
        sys.exit(1)
    print("Clone completed successfully.")
