"""
Utility to check AST coverage across all commits in all repositories.
Identifies empty or missing AST files.
"""
import os
import re
from typing import List, Tuple, Dict


def find_repo_dirs(base_dir: str) -> List[str]:
    """Finds repository directories within the base dataset directory."""
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return []
    return sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])


def find_commit_dirs(repo_dir: str) -> List[str]:
    """Finds and sorts commit directories within a repository directory."""
    commit_pattern = re.compile(r'^commit_(\d+)$')
    commit_dirs = []
    for d in os.listdir(repo_dir):
        if os.path.isdir(os.path.join(repo_dir, d)) and commit_pattern.match(d):
            commit_dirs.append(d)

    # Sort based on the numeric part
    commit_dirs.sort(key=lambda x: int(commit_pattern.match(x).group(1)))
    return commit_dirs


def check_ast_file(ast_path: str) -> Tuple[str, int]:
    """
    Checks an AST file and returns its status and line count.

    Returns:
        Tuple of (status, line_count) where status is:
        - 'missing': File doesn't exist
        - 'empty': File exists but has 0 lines
        - 'ok': File exists and has content
    """
    if not os.path.exists(ast_path):
        return ('missing', 0)

    try:
        with open(ast_path, 'r') as f:
            lines = sum(1 for _ in f)
            if lines == 0:
                return ('empty', 0)
            return ('ok', lines)
    except Exception as e:
        return ('error', 0)


def check_repository_coverage(repo_name: str, base_dir: str = 'data/ast_dataset') -> Dict:
    """
    Checks AST coverage for a single repository.

    Returns:
        Dictionary with statistics and list of issues
    """
    repo_path = os.path.join(base_dir, repo_name)
    commit_dirs = find_commit_dirs(repo_path)

    stats = {
        'total_commits': len(commit_dirs),
        'missing': 0,
        'empty': 0,
        'ok': 0,
        'issues': []
    }

    for commit_dir in commit_dirs:
        ast_path = os.path.join(repo_path, commit_dir, 'ast.jsonl')
        status, line_count = check_ast_file(ast_path)

        if status == 'missing':
            stats['missing'] += 1
            stats['issues'].append({
                'commit': commit_dir,
                'issue': 'missing',
                'path': f"{repo_name}/{commit_dir}"
            })
        elif status == 'empty':
            stats['empty'] += 1
            stats['issues'].append({
                'commit': commit_dir,
                'issue': 'empty',
                'path': f"{repo_name}/{commit_dir}"
            })
        else:
            stats['ok'] += 1

    return stats


def run_coverage_check(base_dir: str = 'data/ast_dataset', output_file: str = 'logs/ast_coverage_report.txt'):
    """
    Runs coverage check for all repositories and generates a report.

    Args:
        base_dir: Base directory containing AST dataset
        output_file: Path to output report file
    """
    repo_names = find_repo_dirs(base_dir)

    if not repo_names:
        print(f"No repository data found in {base_dir}.")
        return

    print(f"\n{'='*60}")
    print(f"AST Coverage Check")
    print(f"{'='*60}\n")

    all_issues = []
    total_stats = {
        'total_repos': len(repo_names),
        'total_commits': 0,
        'missing': 0,
        'empty': 0,
        'ok': 0
    }

    # Check each repository
    for repo_name in repo_names:
        stats = check_repository_coverage(repo_name, base_dir)

        total_stats['total_commits'] += stats['total_commits']
        total_stats['missing'] += stats['missing']
        total_stats['empty'] += stats['empty']
        total_stats['ok'] += stats['ok']

        # Print repository summary
        if stats['missing'] > 0 or stats['empty'] > 0:
            print(f"⚠️  {repo_name}: {stats['missing']} missing, {stats['empty']} empty (out of {stats['total_commits']} commits)")

            # Log individual issues
            for issue in stats['issues']:
                print(f"    [{issue['path']}] - {issue['issue']}")
                all_issues.append(issue)
        else:
            print(f"✓  {repo_name}: All {stats['total_commits']} commits have AST data")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total repositories: {total_stats['total_repos']}")
    print(f"Total commits:      {total_stats['total_commits']}")
    print(f"OK:                 {total_stats['ok']} ({100*total_stats['ok']//total_stats['total_commits'] if total_stats['total_commits'] > 0 else 0}%)")
    print(f"Missing AST:        {total_stats['missing']}")
    print(f"Empty AST:          {total_stats['empty']}")
    print(f"{'='*60}\n")

    # Write report to file
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("AST Coverage Report\n")
        f.write("="*60 + "\n\n")

        f.write(f"Total repositories: {total_stats['total_repos']}\n")
        f.write(f"Total commits:      {total_stats['total_commits']}\n")
        f.write(f"OK:                 {total_stats['ok']} ({100*total_stats['ok']//total_stats['total_commits'] if total_stats['total_commits'] > 0 else 0}%)\n")
        f.write(f"Missing AST:        {total_stats['missing']}\n")
        f.write(f"Empty AST:          {total_stats['empty']}\n\n")

        if all_issues:
            f.write("="*60 + "\n")
            f.write("Issues Found\n")
            f.write("="*60 + "\n\n")

            for issue in all_issues:
                f.write(f"[{issue['path']}] - {issue['issue']}\n")
        else:
            f.write("No issues found - all commits have AST data!\n")

    print(f"Report written to: {output_file}\n")


def cli():
    """Command-line interface for AST coverage check."""
    import sys

    base_dir = 'data/ast_dataset'
    output_file = 'logs/ast_coverage_report.txt'

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    # Allow custom base dir and output file
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    run_coverage_check(base_dir, output_file)


if __name__ == "__main__":
    cli()
