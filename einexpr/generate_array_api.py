"""
This script helps us to stay up-to-date with the latest Python Array API specification. It downloads the specification from Github,
checks that einexpr's implemention implements all of the API, adds the signature and docstring of any unimplemented functions (and
sets the function body to raise a NotImplementedError), checks that the signatures of all implemented functions match those in the
specification, and checks that the documentation of all implemented functions is a superset of the documentation in the specification
(i.e. that, although it may add some documentation, it at least includes all information from the specification).
"""
import libcst as cst
from libcst.tool import dump

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


def process_file(einexpr_file, template_file):
    # Load the einexpr file
    einexpr_file_path = pathlib.Path(einexpr_file)
    einexpr_file_path.parent.mkdir(parents=True, exist_ok=True)
    einexpr_file_content = einexpr_file_path.read_text()

    # Load the template file
    template_file_path = pathlib.Path(template_file)
    template_file_content = template_file_path.read_text()
    
    einexpr_tree = cst.parse_module(einexpr_file_content)
    template_tree = cst.parse_module(template_file_content)

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
    
    # Process the trees and save the result
    einexpr_tree_new = process_trees(einexpr_tree, template_tree)
    einexpr_file_path.write_text(einexpr_tree_new.code)
    print(f"Processed {einexpr_file_path}")


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
        einexpr_file = template_file.replace(template_array_api_path, einexpr_array_api_path)
        process_file(einexpr_file, template_file)