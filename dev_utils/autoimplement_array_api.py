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

from common import unimplemented_function_matcher


single_arg_elementwise_function_names = """
sign sqrt inv where clip argsort ones_like zeros_like full_like softmax log_softmax cumsum abs absolute flip sort astype sin cos tan exp 
log exp2 log2 log10 sqrt pow reciprocal floor ceil round floor_divide nonzero acos acosh add asin abs acos acosh add asin asinh atan atan2 
atanh bitwise_and bitwise_left_shift bitwise_invert bitwise_or bitwise_right_shift bitwise_xor ceil cos cosh divide equal exp expm1 floor 
floor_divide greater greater_equal isfinite isinf isnan less less_equal log log1p log2 log10 logaddexp logical_and logical_not logical_or 
logical_xor multiply negative not_equal positive pow remainder round sign sin sinh square sqrt subtract tan tanh trunc
__abs__ __bool__ __float__ __int__ __invert__ __neg__ __pos__
""".split()
multiple_arg_elementwise_function_names = """
__add__ __and__ __eq__ __floordiv__ __ge__ __gt__ __le__ __lshift__ __lt__ __mod__ __mul__ __ne__ __or__ __pow__ __rshift__ __sub__ 
__truediv__ __xor__
""".split()
single_dimension_reduction_function_names = "".split()
multiple_dimension_reduction_function_names = "argmax argmin sum prod mean std max min maximum minimum all any var".split()
concatenation_function_names = "concat concatenate".split()


class ImplementFunctions(m.MatcherDecoratableTransformer):
    def __init__(self, *args, **kwargs):
        # self.implementation_helper_names_used = set()
        super().__init__(*args, **kwargs)

    @m.leave(unimplemented_function_matcher)
    def implement_function(self, original_node, updated_node):
        # TODO: this assert will fail if the function is on a single line. Replace the Raise instead.
        assert isinstance(raise_node := updated_node.body.body[-1].body[0], cst.Raise) and (raise_node.exc.value == "NotImplementedError" or raise_node.exc.func.value == "NotImplementedError"), f"Expected raise NotImplementedError, got {raise_node}"
        func_name = original_node.name.value
        
        if func_name in single_arg_elementwise_function_names:
            implementation_helper_name = 'SingleArgumentElementwise'
        elif func_name in multiple_arg_elementwise_function_names:
            implementation_helper_name = 'MultipleArgumentElementwise'
        elif func_name in single_dimension_reduction_function_names:
            implementation_helper_name = 'SingleArgumentSingleDimensionReduction'
        elif func_name in multiple_dimension_reduction_function_names:
            implementation_helper_name = 'SingleArgumentMultipleDimensionReduction'
        elif func_name in concatenation_function_names:
            implementation_helper_name = 'Concatenation'
        else:
            return updated_node
        
        # If the return type is not `array`, don't implement it - warn the user.
        return_annotation = updated_node.returns.annotation.value
        if return_annotation != "array":
            print(f"    ⚠️  WARNING: Expected a return type of 'array' for function {func_name!r}; got {return_annotation}")
            return updated_node
        
        # self.implementation_helper_names_used.add(implementation_helper_name)
        
        params = updated_node.params
        posonly_args = [param.name.value for param in params.posonly_params]
        kwonly_args = [param.name.value for param in params.kwonly_params]
        
        posonly_args_str = '(' + ' '.join(arg + ',' for arg in posonly_args) + ')'
        kwonly_args_str = '{' + ', '.join(f"'{name}': {name}" for name in kwonly_args) + '}'
        
        array_param_name = None
        for param in params.posonly_params:
            if isinstance(param.annotation, cst.Tuple):
                for i, element in enumerate(param.annotation.elements):
                    if element.value == "array":
                        array_param_name = f"{param.name.value}[{i}]"
                        break
            elif param.annotation.annotation.value == "array":
                array_param_name = param.name.value
        if array_param_name is None:
            print(f"    ⚠️  WARNING: Expected a parameter of type 'array' in posonly_args for function {func_name!r}; got annotations {[param.annotation.annotation.value for param in params.posonly_params]}")
            return updated_node
        

        implementation_lines = [
            f'args = {posonly_args_str}',
            f'kwargs = {kwonly_args_str}',
            f'out_dims = einexpr.dimension_utils.{implementation_helper_name}.calculate_output_dims(args, kwargs)',
            f'ambiguous_dims = einexpr.dimension_utils.{implementation_helper_name}.calculate_output_ambiguous_dims(args, kwargs)',
            f'processed_args, processed_kwargs = einexpr.dimension_utils.{implementation_helper_name}.process_args(args, kwargs)',
            f'result = einarray(\n    einexpr.backends.get_array_api_backend(array={array_param_name}).{func_name}(*processed_args, **processed_kwargs), \n    dims=out_dims, \n    ambiguous_dims=ambiguous_dims)',
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


def add_reflected_operators(code):
    operators_to_reflect = 'add sub mul truediv floordiv pow mod matmul and or xor lshift rshift'.split()
    tree = cst.parse_module(code)
    wrapper = cst.MetadataWrapper(tree)
    qualified_names_by_node = wrapper.resolve(cst.metadata.QualifiedNameProvider)
    all_qualified_names = {qualified_name.name for qualified_names in qualified_names_by_node.values() for qualified_name in qualified_names}

    class OperatorReflector(m.MatcherDecoratableTransformer):
        @m.leave(m.FunctionDef(name=m.OneOf(*(m.Name(value=f'__{op}__') for op in operators_to_reflect))))
        def reflect_operator(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef):
            op_name = original_node.name.value
            reflected_op_name = f'__r{op_name[2:]}'
            op_qname = list(qualified_names_by_node[original_node])[0].name
            reflected_op_qname = '.'.join(op_qname.split('.')[:-1] + [reflected_op_name])
            # If the reflected operator is already defined, do nothing.
            if reflected_op_qname in all_qualified_names:
                return updated_node
            original_comment_node = updated_node.body.body[0].body[0].value
            original_comment = original_comment_node.value
            assert original_comment.split()[0] in ['"""', "'''"], f"Expected comment to start with ''' or '\"\"', got {original_comment}"
            comment_start = original_comment.split('\n')[0]
            comment_body = '\n'.join(original_comment.split('\n')[1:])
            indent = re.search(r'\s*', comment_body).group(0)
            new_comment = f'{comment_start}\n{indent}Reflection of {op_name}. Original comments:\n\n{comment_body}'
            reflected_node = updated_node
            # Replace the comment string
            reflected_node = reflected_node.deep_replace(original_comment_node, original_comment_node.with_changes(value=new_comment))
            # Replace all instances of the symbol of the operator with the reflected operator
            class ReplaceOperator(m.MatcherDecoratableTransformer):
                @m.leave(m.Name(value=op_name))
                def replace_operator(self, original_node: cst.Name, updated_node: cst.Name):
                    return updated_node.with_changes(value=reflected_op_name)
            
            reflected_node = reflected_node.visit(ReplaceOperator())
            return cst.FlattenSentinel([original_node, cst.EmptyLine(), reflected_node])

    tree = wrapper.module.visit(OperatorReflector())
    return tree.code


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
    # AddImportsVisitor.add_needed_import(codemod_context, 'einexpr', 'einarray')
    AddImportsVisitor.add_needed_import(codemod_context, 'einexpr')
    tree = AddImportsVisitor(codemod_context).transform_module(tree)
    # Write the code back to the file
    with open(file_path, 'w') as f:
        f.write(tree.code)


if __name__ == '__main__':
    einexpr_array_api_path = pathlib.Path("einexpr/array_api")
    
    # Get the list of files to process
    for file_path in glob.glob(str(einexpr_array_api_path / "**/*.py"), recursive=True):
        print(f"Processing {file_path}")
        process_file(file_path)
    
    # Special modifications
    # __init__.py
    # Convert ``import array_api.data_types as dtype`` into a relative import
    print("Processing __init__.py")
    print("    Converting `import array_api.data_types as dtype` into a relative import")
    with open(einexpr_array_api_path / "__init__.py", 'r') as f:
        code = f.read()
    code = code.replace("import array_api.data_types as dtype", "from . import data_types as dtype")
    with open(einexpr_array_api_path / "__init__.py", 'w') as f:
        f.write(code)
        
    # einarray_object.py
    print("Processing einarray_object.py")
    with open(einexpr_array_api_path / "einarray_object.py", 'r') as f:
        code = f.read()
    # Remove the line 
    print("    Removing lines `array = _array` and `__all__ = ['array']`")
    # Replace ``array = _array`` with ``array = einarray``
    code = code.replace("\narray = _array\n", "\narray = einarray\n")
    # Change the name of the _array class to einarray
    print("    Changing the name of the _array class to einarray")
    code = re.sub(r'class _array(?P<args>.*?):', r'class einarray\g<args>:', code)
    # Warn the user if the symbol _array is used anywhere else. Won't trigger if it's used within a symbol (e.g. _array2)
    if re.search(r'[^\w_]_array[^\w\d_]', code):
        print("    WARNING: _array is used somewhere else. This will not trigger if it's used within a symbol (e.g. _array2)")
    # Add reflected operators
    code = add_reflected_operators(code)
    # Write the code back to the file
    with open(einexpr_array_api_path / "einarray_object.py", 'w') as f:
        f.write(code)