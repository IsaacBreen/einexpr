"""
Provides some useful information about einexpr.
"""

import libcst as cst
from libcst.tool import dump
import libcst.matchers as m
from libcst.metadata import FullRepoManager, FullyQualifiedNameProvider

import os
import sys
import glob
import shutil
import pathlib
from pathlib import Path
import colorful as cf

import re

from common import unimplemented_function_matcher


def function_is_implemented(function_def: cst.FunctionDef):
    """
    Returns True if the function is implemented.
    """
    return not m.matches(function_def, unimplemented_function_matcher)


# Tools for analysing the Array API
def analyse_array_api():
    ARRAY_API_FOLDER = Path('einexpr/array_api')
    FUNCTION_NAME_PADDING = 32

    # Get the list of files in the array_api folder
    files = glob.glob(str(ARRAY_API_FOLDER / '**/*.py'), recursive=True)

    # Setup libcst
    repo_manager = FullRepoManager('.', files, {FullyQualifiedNameProvider})

    # Process each file
    for file in files:
        # Get the fully qualified names of all symbols in the file
        wrapper = repo_manager.get_metadata_wrapper_for_path(file)
        fqnames_by_node = wrapper.resolve(FullyQualifiedNameProvider)
        fqnames_by_node_by_type = dict()
        for k, v in fqnames_by_node.items():
            fqnames_by_node_by_type.setdefault(type(k), dict())[k] = v
        # Print all functions
        print(f'{cf.bold("Functions implemented in")} {file}')
        for function_def, fqnames_by_node in fqnames_by_node_by_type.get(cst.FunctionDef, dict()).items():
            for fqname in fqnames_by_node:
                function_name_padded = fqname.name + ' ' * max(FUNCTION_NAME_PADDING - len(function_def.name.value), 1)
                if function_is_implemented(function_def):
                    print(f'    ✅  {function_name_padded}')
                else:
                    print(f'    ❌  {function_name_padded}')
                
        
if __name__ == "__main__":
    analyse_array_api()