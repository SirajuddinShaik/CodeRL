import json
import sys
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple

def load_ast_data(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Loads AST data from a JSONL file into a dictionary keyed by function id."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            try:
                node = json.loads(line)
                if 'id' in node:
                    data[node['id']] = node
            except json.JSONDecodeError:
                print(f"Warning: Could not decode line in {filepath}: {line.strip()}", file=sys.stderr)
    return data

def compute_diff_indices(before_code: str, after_code: str) -> Tuple[List[int], List[int]]:
    """
    Computes the indices of changed lines between two code snippets.
    Returns a tuple of (before_indices, after_indices).
    """
    before_lines = before_code.splitlines()
    after_lines = after_code.splitlines()
    matcher = SequenceMatcher(None, before_lines, after_lines, autojunk=False)
    
    before_indices = []
    after_indices = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            before_indices.extend(range(i1, i2))
            after_indices.extend(range(j1, j2))

    # Fallback: If strings are different but no diffs were found (e.g. whitespace changes),
    # mark the whole function as changed.
    if not before_indices and not after_indices and before_code != after_code:
        return list(range(len(before_lines))), list(range(len(after_lines)))
            
    return before_indices, after_indices

def extract_context_lines(code: str, diff_indices: List[int], context: int = 2) -> str:
    """
    Extracts context lines around the given diff indices.
    """
    if not diff_indices:
        return ""
        
    lines = code.splitlines()
    min_index = max(0, min(diff_indices) - context)
    max_index = min(len(lines), max(diff_indices) + context + 1)
    
    return "\n".join(lines[min_index:max_index])

def main(file1: str, file2: str, output_file: str, commit_data_file: str = None):
    """
    Compares two AST JSONL files and writes the diffs to an output file.
    Optionally includes commit metadata if a commit_data file is provided.
    """
    commit_metadata = None
    if commit_data_file:
        try:
            with open(commit_data_file, 'r') as f:
                data = json.load(f)
                commit_metadata = data.get('metadata', {})
                commit_metadata['commit_hash'] = data.get('commit_hash')
        except (IOError, json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Warning: Could not load or parse commit data from {commit_data_file}: {e}", file=sys.stderr)

    ast1 = load_ast_data(file1)
    ast2 = load_ast_data(file2)
    
    ids1 = set(ast1.keys())
    ids2 = set(ast2.keys())
    
    common_ids = ids1.intersection(ids2)
    added_ids = ids2 - ids1
    removed_ids = ids1 - ids2
    
    diffs = []
    
    # Modified functions
    for func_id in common_ids:
        node1 = ast1[func_id]
        node2 = ast2[func_id]
        
        code_changed = node1.get('code') != node2.get('code')
        imports_changed = node1.get('imports') != node2.get('imports')

        if code_changed or imports_changed:
            before_code = node1.get('code', '')
            after_code = node2.get('code', '')
            
            before_indices, after_indices = compute_diff_indices(before_code, after_code)
            
            before_context = extract_context_lines(before_code, before_indices)
            after_context = extract_context_lines(after_code, after_indices)

            diff_entry = {
                "id": func_id,
                "file": node2.get('file'),
                "kind": node2.get('kind'),
                "status": "modified",
                "code_changed": code_changed,
                "imports_changed": imports_changed,
                "before_code": before_code,
                "after_code": after_code,
                "diff_span": {
                    "before": before_context,
                    "after": after_context
                }
            }

            if imports_changed:
                diff_entry["before_imports"] = node1.get('imports', [])
                diff_entry["after_imports"] = node2.get('imports', [])
            
            if commit_metadata:
                diff_entry['commit_metadata'] = commit_metadata

            diffs.append(diff_entry)
            
    # Added functions
    for func_id in added_ids:
        node = ast2[func_id]
        entry = {
            "id": func_id,
            "file": node.get('file'),
            "kind": node.get('kind'),
            "status": "added",
            "before_code": None,
            "after_code": node.get('code'),
            "diff_span": None
        }
        if commit_metadata:
            entry['commit_metadata'] = commit_metadata
        diffs.append(entry)
        
    # Removed functions
    for func_id in removed_ids:
        node = ast1[func_id]
        entry = {
            "id": func_id,
            "file": node.get('file'),
            "kind": node.get('kind'),
            "status": "removed",
            "before_code": node.get('code'),
            "after_code": None,
            "diff_span": None
        }
        if commit_metadata:
            entry['commit_metadata'] = commit_metadata
        diffs.append(entry)
        
    if not diffs:
        # User-requested sanity check
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            lines1 = sum(1 for _ in f1)
            lines2 = sum(1 for _ in f2)
            if lines1 != lines2:
                print(f"---------------------Warning: No functional diffs found, but line counts differ. File1: {lines1}, File2: {lines2}")
            else:
                print("No differences found between the two commits. File line counts match.")
        return

    with open(output_file, 'w') as f:
        for diff in diffs:
            f.write(json.dumps(diff) + '\n')
            
    print(f"Diffs written to {output_file}")

def cli():
    """Command-line interface for building commit diffs."""
    if len(sys.argv) not in [4, 5]:
        print("Usage: build-commit-diff <commit1.jsonl> <commit2.jsonl> <output.jsonl> [commit_data.json]", file=sys.stderr)
        sys.exit(1)

    commit_data_arg = sys.argv[4] if len(sys.argv) == 5 else None
    main(sys.argv[1], sys.argv[2], sys.argv[3], commit_data_file=commit_data_arg)


if __name__ == "__main__":
    cli()
