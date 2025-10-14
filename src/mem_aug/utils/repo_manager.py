"""
Repository management utility for downloading and tracking Git repositories.
"""
import os
import subprocess
import yaml
from pathlib import Path
from typing import List, Dict, Optional


def get_project_root() -> Path:
    """Get the project root directory by looking for config/config.yaml."""
    current = Path.cwd()
    # Look for config/config.yaml starting from current directory
    while current != current.parent:
        config_file = current / "config" / "config.yaml"
        if config_file.exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (config/config.yaml not found)")


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_repos_dir(config: Optional[dict] = None) -> str:
    """Get the repositories directory from config."""
    if config is None:
        config = load_config()
    return config['paths']['repos']


def list_downloaded_repos(repos_dir: Optional[str] = None) -> List[str]:
    """
    List all repositories currently downloaded in the repos directory.

    Returns:
        List of repository names (directory names in repos folder)
    """
    if repos_dir is None:
        config = load_config()
        repos_dir = get_repos_dir(config)

    if not os.path.exists(repos_dir):
        return []

    repos = []
    for item in os.listdir(repos_dir):
        item_path = os.path.join(repos_dir, item)
        # Check if it's a directory and contains a .git folder
        if os.path.isdir(item_path):
            git_path = os.path.join(item_path, '.git')
            git_disabled_path = os.path.join(item_path, '.git_disabled')
            if os.path.exists(git_path) or os.path.exists(git_disabled_path):
                repos.append(item)

    return sorted(repos)


def download_repo(name: str, url: str, repos_dir: str) -> bool:
    """
    Download (clone) a repository if it doesn't already exist.

    Args:
        name: Repository name (will be used as directory name)
        url: Git URL to clone from
        repos_dir: Base directory where repos are stored

    Returns:
        True if repo was downloaded, False if it already exists
    """
    repo_path = os.path.join(repos_dir, name)

    if os.path.exists(repo_path):
        print(f"✓ Repository '{name}' already exists at {repo_path}")
        return False

    print(f"Downloading '{name}' from {url}...")
    os.makedirs(repos_dir, exist_ok=True)

    try:
        subprocess.run(
            ['git', 'clone', url, repo_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Successfully downloaded '{name}'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download '{name}': {e.stderr}")
        return False


def sync_repos(config_path: str = "config/config.yaml") -> Dict[str, bool]:
    """
    Sync all repositories from config - download any that are missing.

    Returns:
        Dictionary mapping repo names to whether they were downloaded (True) or existed (False)
    """
    config = load_config(config_path)
    repos_dir = get_repos_dir(config)
    repositories = config.get('repositories', [])

    results = {}

    print(f"\n{'='*60}")
    print(f"Syncing repositories to {repos_dir}")
    print(f"{'='*60}\n")

    for repo in repositories:
        name = repo['name']
        url = repo['url']
        results[name] = download_repo(name, url, repos_dir)

    return results


def get_repo_status(config_path: str = "config/config.yaml") -> Dict[str, str]:
    """
    Get status of all repositories (configured vs downloaded).

    Returns:
        Dictionary mapping repo names to status:
        - 'downloaded': Repo is in config and downloaded
        - 'missing': Repo is in config but not downloaded
        - 'untracked': Repo is downloaded but not in config
    """
    config = load_config(config_path)
    repos_dir = get_repos_dir(config)
    configured_repos = {repo['name']: repo['url'] for repo in config.get('repositories', [])}
    downloaded_repos = set(list_downloaded_repos(repos_dir))

    status = {}

    # Check configured repos
    for name in configured_repos:
        if name in downloaded_repos:
            status[name] = 'downloaded'
        else:
            status[name] = 'missing'

    # Check for untracked repos
    for name in downloaded_repos:
        if name not in configured_repos:
            status[name] = 'untracked'

    return status


def print_status(config_path: str = "config/config.yaml"):
    """Print a formatted status report of all repositories."""
    status = get_repo_status(config_path)

    print(f"\n{'='*60}")
    print("Repository Status")
    print(f"{'='*60}\n")

    downloaded = [name for name, s in status.items() if s == 'downloaded']
    missing = [name for name, s in status.items() if s == 'missing']
    untracked = [name for name, s in status.items() if s == 'untracked']

    if downloaded:
        print(f"✓ Downloaded ({len(downloaded)}):")
        for name in downloaded:
            print(f"  - {name}")
        print()

    if missing:
        print(f"✗ Missing ({len(missing)}):")
        for name in missing:
            print(f"  - {name}")
        print()

    if untracked:
        print(f"? Untracked ({len(untracked)}):")
        for name in untracked:
            print(f"  - {name} (not in config)")
        print()

    print(f"{'='*60}\n")


def cli():
    """Command-line interface for repo manager."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  manage-repos sync    # Download missing repos")
        print("  manage-repos status  # Show repo status")
        print("  manage-repos list    # List downloaded repos")
        sys.exit(1)

    command = sys.argv[1]

    if command == "sync":
        sync_repos()
    elif command == "status":
        print_status()
    elif command == "list":
        repos = list_downloaded_repos()
        print(f"\nDownloaded repositories ({len(repos)}):")
        for repo in repos:
            print(f"  - {repo}")
        print()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
