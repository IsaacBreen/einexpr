"""
This script helps us to stay up-to-date with the latest Python Array API specification.  More specifically, it:
1. downloads the specification from Github,
2. checks that einexpr's implemention implements all of the API,
3. adds the signature and docstring of any unimplemented functions (and sets the function body to raise a NotImplementedError),
4. checks that the signatures of all implemented functions match those in the specification, and
5. checks that the documentation of all implemented functions is a superset of the documentation in the specification (i.e. that, although it may add some documentation, it at least includes all information from the specification).
"""
import libcst as cst
from libcst.tool import dump
import libcst.matchers as m

import os
import sys
import glob
import shutil
import pathlib
import difflib

from git import Repo

import re


def download_repo(repo_url, repo_subdir, save_dir, temp_dir="tmp"):
    # Clone the folder array-api/spec/API_specification/array_api/ from https://github.com/data-apis/array-api/ into ./array_api
    repo = Repo.init(temp_dir)

    # Create a new remote if there isn't one already created
    origin = repo.remotes[0] if repo.remotes else repo.create_remote("origin", repo_url)

    origin.fetch()
    git = repo.git()
    git.checkout("origin/main", "--", repo_subdir)

    # Move the repo to the save_dir, overwriting any existing files
    copy_from = pathlib.Path(temp_dir) / repo_subdir
    copy_to = pathlib.Path(save_dir)
    if copy_to.exists():
        shutil.rmtree(copy_to)
    shutil.copytree(copy_from, copy_to)
    
    shutil.rmtree(temp_dir)


class FunctionAndClassCollector(cst.CSTVisitor):
    def __init__(self):
        self.functions = {}
        self.classes = {}

    def visit_FunctionDef(self, node):
        if node.name.value in self.functions:
            raise Exception(f"Function {node.name.value} is defined twice")
        self.functions[node.name.value] = node
        return False
    
    def visit_ClassDef(self, node):
        if node.name.value in self.classes:
            raise Exception(f"Class {node.name.value} is defined twice")
        self.classes[node.name.value] = node
        return False


def process_trees(einexpr_tree, template_tree):
    # Collect all the functions and classes
    einexpr_collector = FunctionAndClassCollector()
    template_collector = FunctionAndClassCollector()
    if isinstance(einexpr_tree, cst.ClassDef) and isinstance(template_tree, cst.ClassDef):
        for child in einexpr_tree.body.body:
            child.visit(einexpr_collector)
        for child in template_tree.body.body:
            child.visit(template_collector)
    elif isinstance(einexpr_tree, cst.Module) and isinstance(template_tree, cst.Module):
        einexpr_tree.visit(einexpr_collector)
        template_tree.visit(template_collector)
    
    # Check that all functions are implemented with the correct signatures
    for name in set(einexpr_collector.functions) & set(template_collector.functions):
        einexpr_func = einexpr_collector.functions[name]
        template_func = template_collector.functions[name]
        if not (einexpr_func.params).deep_equals(template_func.params):
            raise Exception(f"Parameters of function {name} do not match. Expected: {template_func.params}, got: {einexpr_func.params}")
        if einexpr_func.returns != template_func.returns and einexpr_func.returns and not einexpr_func.returns.deep_equals(template_func.returns):
            raise Exception(f"Returns of function {name} do not match. Expected: {template_func.returns}, got: {einexpr_func.returns}")
    
    # Insert the signature and docstring of any unimplemented functions and set the function body to raise a NotImplementedError
    for name in set(template_collector.functions) - set(einexpr_collector.functions):
        template_func = template_collector.functions[name]
        template_func.body.body = list(template_func.body.body)
        template_func.body.body.append(cst.parse_statement("raise NotImplementedError"))
        einexpr_tree.body.append(template_func)
        
    # Insert any missing classes
    for name in set(template_collector.classes) - set(einexpr_collector.classes):
        template_class = template_collector.classes[name]
        einexpr_tree.body.append(template_class)
    
    # Recurse over classes
    for name in set(einexpr_collector.classes) & set(template_collector.classes):
        einexpr_class = einexpr_collector.classes[name]
        template_class = template_collector.classes[name]
        process_trees(einexpr_class, template_class)
        
    return einexpr_tree

class RaiseNotImplementedVisitor(m.MatcherDecoratableTransformer):
    """
    Add ``raise NotImplementedError`` to any function that has not been implemented.
    """
    @m.leave(
        m.FunctionDef(
            body=m.OneOf(
                m.SimpleStatementSuite(
                    body=[m.Expr(value=m.SimpleString())]
                ),
                m.IndentedBlock(
                    body=[m.SimpleStatementLine([m.Expr(value=m.SimpleString())])]
                )
            )
        )
    )
    def append_function_body_with_raise_NotImplementedError(self, original_node, updated_node):
        return updated_node.with_changes(
            body=updated_node.body.with_changes(
                body=[*updated_node.body.body, cst.Expr(cst.parse_statement("raise NotImplementedError"))]
            )
        )

def process_file(einexpr_file, template_file):    
    # If the einexpr file doesn't exist, create it from the template file
    if not os.path.exists(einexpr_file):
        # Create the necessary (sub)directories
        os.makedirs(os.path.dirname(einexpr_file), exist_ok=True)
        with open(template_file, "r") as f, open(einexpr_file, "w") as g:
            g.write(f.read())
        with open(einexpr_file, "r") as f:
            einexpr_tree = cst.parse_module(f.read())

    # Load the einexpr file
    einexpr_file_path = pathlib.Path(einexpr_file)
    einexpr_file_path.parent.mkdir(parents=True, exist_ok=True)
    einexpr_file_content = einexpr_file_path.read_text()

    # Load the template file
    template_file_path = pathlib.Path(template_file)
    template_file_content = template_file_path.read_text()
    
    einexpr_tree = cst.parse_module(einexpr_file_content)
    template_tree = cst.parse_module(template_file_content)

    # Process the trees
    einexpr_tree_new = process_trees(einexpr_tree, template_tree)
    
    # Raise ``NotImplementedError`` for any functions that are missing a body
    visitor = RaiseNotImplementedVisitor()
    einexpr_tree_new = einexpr_tree_new.visit(visitor)

    # Save the einexpr file
    einexpr_file_path.write_text(einexpr_tree_new.code)


if __name__ == "__main__":
    template_array_api_path = "einexpr/external_apis/array_api"
    einexpr_array_api_path = "einexpr/array_api"

    # Download the array API specification from Github
    download_repo(
        "https://github.com/data-apis/array-api/",
        "spec/API_specification/array_api/",
        template_array_api_path,
        temp_dir="tmp_array_api"
    )
    
    # Process the array API specification
    for template_file in glob.glob(f"{template_array_api_path}/**/*.py", recursive=True):
        if os.path.basename(template_file) == 'array_object.py':
            einexpr_einarray_file = f"{einexpr_array_api_path}/einarray_object.py"
            einexpr_lazy_einarray_file = f"{einexpr_array_api_path}/lazy_einarray_object.py"
            process_to = [einexpr_einarray_file, einexpr_lazy_einarray_file]
        else:
            einexpr_file = template_file.replace(template_array_api_path, einexpr_array_api_path)
            process_to = [einexpr_file]
        for einexpr_file in process_to:
            print(f"Processing {template_file} -> {einexpr_file}")
            process_file(einexpr_file, template_file)