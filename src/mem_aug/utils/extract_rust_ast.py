import os
import json
from tree_sitter_languages import get_parser

# Get the Rust parser using tree_sitter_languages
parser = get_parser("rust")

def get_node_text(node, source_code):
    return source_code[node.start_byte:node.end_byte].decode("utf8", errors='ignore')

def process_rust_file(file_path, project_root):
    with open(file_path, "rb") as f:
        source_code = f.read()

    tree = parser.parse(source_code)
    root_node = tree.root_node

    imports = []
    extern_crates = []
    attributes = []

    def find_top_level_items(node):
        if node.type == 'use_declaration':
            imports.append(get_node_text(node, source_code))
        elif node.type == 'extern_crate_declaration':
            extern_crates.append(get_node_text(node, source_code))
        elif node.type == 'attribute_item':
            attributes.append(get_node_text(node, source_code))
        for child in node.children:
            find_top_level_items(child)

    find_top_level_items(root_node)

    items = []
    rel_file_path = os.path.relpath(file_path, project_root)

    # Add a module-level entry capturing all top-level declarations
    module_declarations = []
    module_declarations.extend(extern_crates)
    module_declarations.extend(attributes)
    module_declarations.extend(imports)

    if module_declarations:
        items.append({
            "id": f"{rel_file_path}::module",
            "kind": "module_declarations",
            "parent": None,
            "code": "\n".join(module_declarations),
            "file": rel_file_path,
            "extern_crates": extern_crates,
            "attributes": attributes,
            "imports": imports
        })

    def get_used_imports(code_block, all_imports):
        used = []
        for imp in all_imports:
            # This is a simplification. It checks if the imported name is in the code.
            # It doesn't handle complex cases like `use std::collections::*` or aliases.
            parts = imp.replace(';', '').split('::')
            last_part = parts[-1].strip()
            if '{' in last_part: # Handle `use std::collections::{HashMap, HashSet}`
                sub_imports = last_part.replace('{', '').replace('}', '').split(',')
                for sub_import in sub_imports:
                    if sub_import.strip() in code_block:
                        used.append(imp)
                        break
            elif last_part in code_block:
                used.append(imp)
        return list(set(used))

    def traverse(node, parent=None):
        if node.type in ["function_item", "struct_item", "enum_item", "impl_item",
                         "trait_item", "mod_item", "const_item", "static_item",
                         "type_item", "macro_definition", "union_item", "foreign_mod_item",
                         "function_signature_item"]:
            # For impl blocks, extract the type being implemented
            if node.type == "impl_item":
                type_node = node.child_by_field_name("type")
                item_name = get_node_text(type_node, source_code) if type_node else None
            # For foreign_mod_item (extern blocks), use "extern" + ABI as name
            elif node.type == "foreign_mod_item":
                # Try to extract the ABI string like "C"
                abi_node = None
                for child in node.children:
                    if child.type == "string_literal":
                        abi_node = child
                        break
                if abi_node:
                    abi_text = get_node_text(abi_node, source_code).strip('"')
                    item_name = f"extern_{abi_text}"
                else:
                    item_name = "extern"
            else:
                name_node = node.child_by_field_name("name")
                item_name = get_node_text(name_node, source_code) if name_node else None

            code_block = get_node_text(node, source_code)
            used_imports = get_used_imports(code_block, imports)

            # Extract the kind name for the id (e.g., "function", "struct", "impl")
            kind_name = node.type.replace("_item", "").replace("_definition", "")

            items.append({
                "id": f"{rel_file_path}::{parent+'::' if parent else ''}{kind_name}::{item_name or 'unknown'}",
                "kind": node.type,
                "parent": parent,
                "code": code_block,
                "file": rel_file_path,
                "imports": used_imports
            })

            # For impl blocks, extract methods as separate items
            if node.type == "impl_item":
                for child in node.children:
                    if child.type == "function_item":
                        method_name = get_node_text(child.child_by_field_name("name"), source_code)
                        method_code = get_node_text(child, source_code)
                        used_imports_method = get_used_imports(method_code, imports)
                        items.append({
                            "id": f"{rel_file_path}::{item_name}::method::{method_name}",
                            "kind": "method",
                            "parent": item_name,
                            "code": method_code,
                            "file": rel_file_path,
                            "imports": used_imports_method
                        })

            # For trait blocks, extract trait methods as separate items
            elif node.type == "trait_item":
                for child in node.children:
                    if child.type == "function_item":
                        method_name = get_node_text(child.child_by_field_name("name"), source_code)
                        method_code = get_node_text(child, source_code)
                        used_imports_method = get_used_imports(method_code, imports)
                        items.append({
                            "id": f"{rel_file_path}::{item_name}::trait_method::{method_name}",
                            "kind": "trait_method",
                            "parent": item_name,
                            "code": method_code,
                            "file": rel_file_path,
                            "imports": used_imports_method
                        })

        for child in node.children:
            traverse(child, parent=item_name if node.type in ["struct_item", "impl_item", "trait_item", "mod_item"] else parent)

    traverse(root_node)
    return items

def main(project_path, output_jsonl="rust_ast.jsonl"):
    all_items = []
    for root, _, files in os.walk(project_path):
        for fname in files:
            if fname.endswith(".rs"):
                fpath = os.path.join(root, fname)
                # print(f"[*] Processing {fpath}")
                try:
                    data = process_rust_file(fpath, project_path)
                    all_items.extend(data)
                except Exception as e:
                    print(f"[err] {fpath}: {e}")

    with open(output_jsonl, "w") as f:
        for item in all_items:
            f.write(json.dumps(item) + "\n")

    print(f"[ok] Extracted {len(all_items)} items → {output_jsonl}")

if __name__ == "__main__":
    # Example: process a local Rust project
    main("data/codebase/bat", "bat_ast.jsonl")
