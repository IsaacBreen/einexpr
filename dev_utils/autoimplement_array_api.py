"""
Implements the Array API in a semi-automated way.
"""

import libcst as cst
from libcst.tool import dump
import libcst.matchers as m
from libcst.codemod.visitors import AddImportsVisitor
from libcst.codemod import CodemodContext

import os
import sys
import glob
import shutil
import pathlib
import difflib

from git import Repo

import re


single_arg_elementwise_function_names = "sign sqrt inv where clip argsort ones_like zeros_like full_like softmax log_softmax cumsum abs absolute sin cos tan exp log exp2 log2 log10 sqrt pow reciprocal floor ceil round floor_divide".split()
multiple_arg_elementwise_function_names = []
single_dimension_reduction_function_names = "sum product mean std max min maximum minimum all any".split()
multiple_dimension_reduction_function_names = "argmax argmin".split()
concatenation_function_names = "concat concatenate".split()


class ImplementFunctions(m.MatcherDecoratableTransformer):
    def __init__(self, *args, **kwargs):
        self.implementation_helper_names_used = set()
        super().__init__(*args, **kwargs)

    @m.leave(
        m.FunctionDef(
            body=m.OneOf(
                # TODO: This bit should be simplified and generalised to a function definition formatted any way, not just these two ways.
                m.SimpleStatementSuite(
                    body=[
                        m.ZeroOrOne(m.Expr(value=m.SimpleString())),
                        m.Raise(exc=m.Call(func=m.Name(value="NotImplementedError")))
                    ]
                ),
                m.IndentedBlock(
                    body=[
                        m.ZeroOrOne(m.SimpleStatementLine(body=[m.Expr(value=m.SimpleString())])),
                        m.SimpleStatementLine(
                            body=[
                                m.Raise(
                                    exc=m.OneOf(
                                        m.Name(value="NotImplementedError"),
                                        m.Call(func=m.Name(value="NotImplementedError"),
                                        )
                                    )
                                )
                            ]
                        )
                    ]
                )
            )
        )
    )
    def implement_function(self, original_node, updated_node):
        # TODO: this assert will fail if the function is on a single line. Replace the Raise instead.
        assert isinstance(raise_node := updated_node.body.body[-1].body[0], cst.Raise) and (raise_node.exc.value == "NotImplementedError" or raise_node.exc.func.value == "NotImplementedError"), f"Expected raise NotImplementedError, got {raise_node}"

        func_name = original_node.name.value
        if func_name in single_arg_elementwise_function_names:
            implementation_helper_name = 'SingleArgumentElementwise'
        elif func_name in multiple_arg_elementwise_function_names:
            implementation_helper_name = 'MultiArgumentElementwise'
        elif func_name in single_dimension_reduction_function_names:
            implementation_helper_name = 'SingleDimensionReduction'
        elif func_name in multiple_dimension_reduction_function_names:
            implementation_helper_name = 'MultiDimensionReduction'
        elif func_name in concatenation_function_names:
            implementation_helper_name = 'Concatenation'
        else:
            return updated_node
        
        self.implementation_helper_names_used.add(implementation_helper_name)
        
        implementation_lines = [
            f'out_dims = {implementation_helper_name}.calculate_output_dims(args, kwargs)',
            f'ambiguous_dims = {implementation_helper_name}.calculate_output_ambiguous_dims(args, kwargs)',
            f'processed_args, processed_kwargs = {implementation_helper_name}.process_args(args, kwargs)',
            f'result = einexpr.einarray({func_name}(*processed_args, **processed_kwargs), dims=out_dims, ambiguous_dims=ambiguous_dims)',
            f'return result',
        ]
        
        implementation_expressions = [cst.parse_statement(line) for line in implementation_lines]

        print(f"    Implementing {func_name} with {implementation_helper_name}")

        return updated_node.with_changes(
            body=updated_node.body.with_changes(
                body=cst.FlattenSentinel([
                    *updated_node.body.body[:-1],
                    *implementation_expressions
                ])
            )
        )


def process_file(file_path):
    """
    For each function definition in the file, set the function body to the corresponding 
    """
    # Load the file as a CST tree
    with open(file_path, 'r') as f:
        code = f.read()
    tree = cst.parse_module(code)
    # Implement the functions
    implement_functions_transformer = ImplementFunctions()
    tree = tree.visit(implement_functions_transformer)
    # Import the necessary helpers
    codemod_context = CodemodContext()
    AddImportsVisitor.add_needed_import(codemod_context, 'einexpr', 'einarray')
    for helper_name in implement_functions_transformer.implementation_helper_names_used:
        AddImportsVisitor.add_needed_import(codemod_context, 'einexpr', helper_name)
    tree = AddImportsVisitor(codemod_context).transform_module(tree)
    # Write the code back to the file
    with open(file_path, 'w') as f:
        f.write(tree.code)


if __name__ == '__main__':
    einexpr_array_api_path = "einexpr/array_api"
    
    # Get the list of files to process
    for file_path in glob.glob(f"{einexpr_array_api_path}/**/*.py", recursive=True):
        print(f"Processing {file_path}")
        process_file(file_path)