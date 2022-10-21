import ast
from typing import Dict, overload
import libcst as cst
from dataclasses import fields, _MISSING_TYPE, replace

class ExportedFunctionGather(cst.CSTVisitor):
    def __init__(self):
        self.exported_functions = {}
        super().__init__()

    def visit_FunctionDef(self, node):
        if not node.name.value.startswith('_'):
            self.exported_functions[node.name.value] = node
        return False

    def visitClassDef(self, node):
        return False


class StandardiseAst(cst.CSTTransformer):
    """
    Standardise the AST by removing all stylistic information. Useful if you want to check that two pieces of code are equivalent up to but not including differences in formatting.
    """
    
    def on_leave(self, original_node: cst.CSTNode, updated_node: cst.CSTNode) -> cst.CSTNode:
        # Remove all optional nodes
        OPTIONAL_NODES = (
            cst.Comment,
            cst.EmptyLine,
            # cst.Newline,
            # cst.ParenthesizedWhitespace,
            # cst.TrailingWhitespace,
            # cst.BaseParenthesizableWhitespace,
            cst.MaybeSentinel,
        )
        if isinstance(updated_node, OPTIONAL_NODES):
            return cst.RemoveFromParent()
        # Delete metadata
        if hasattr(updated_node, 'metadata'):
            updated_node = replace(updated_node, metadata=[])
        return original_node


def standardised_parse_module(code: str) -> cst.Module:
    code = ast.unparse(ast.parse(code))
    tree = cst.parse_module(code)
    tree = tree.visit(StandardiseAst())
    return tree


BACKENDS = ["numpy", "torch", "jax"]
IVY_PATH_TEMPLATE = "ivy/ivy/functional/ivy/{filename}"
BACKEND_PATH_TEMPLATE = "ivy/ivy/functional/backends/{backend}/{filename}"


@overload
def get_exported_functions(tree: cst.CSTNode) -> Dict[str, cst.FunctionDef]:
    ...


@overload
def get_exported_functions(path: str) -> Dict[str, cst.FunctionDef]:
    ...


def get_exported_functions(x):
    if isinstance(x, str):
        path = x
        code = open(path).read()
        tree = standardised_parse_module(code)
        return get_exported_functions(tree)
    elif isinstance(x, cst.CSTNode):
        tree = x
        # Gather all exported functions
        g = ExportedFunctionGather()
        tree.visit(g)
        return g.exported_functions


def make_array_api_compliance_report(backend: str) -> str:
    """
    Generate a report of array API compliance for a backend.
    """
    # Gather all exported functions for Ivy and the given backend
    ivy_exported_functions = get_exported_functions(IVY_PATH_TEMPLATE.format(filename="manipulation.py"))
    backend_exported_functions = get_exported_functions(BACKEND_PATH_TEMPLATE.format(backend=backend, filename="manipulation.py"))
    # Build the report
    report = f"| Function | Ivy | {backend} | Parameter names | Parameter types | Return type |\n"
    report += "| --- | --- | --- | --- | --- |\n"
    for name in sorted(set(ivy_exported_functions.keys() | set(backend_exported_functions.keys()))):
        if name in ivy_exported_functions and name not in backend_exported_functions:
            report += rf"| {name} | ✅ | ❌ | ❌ | ❌ |"
            continue
        if name not in ivy_exported_functions and name in backend_exported_functions:
            report += rf"| {name} | ❌ | ✅ | ❌ | ❌ |"
            continue
        else:
            ivy_function = ivy_exported_functions[name]
            backend_function = backend_exported_functions[name]
            # Check parameter names
            ivy_parameter_names = [param.name.value for param in ivy_function.params.params]
            backend_parameter_names = [param.name.value for param in backend_function.params.params]
            parameter_name_pass = ivy_parameter_names != backend_parameter_names
            # Check parameter types
            ivy_parameter_types = [param.annotation for param in ivy_function.params.params]
            backend_parameter_types = [param.annotation for param in backend_function.params.params]
            parameter_type_pass = all(ivy_param.deep_equals(backend_param) for ivy_param, backend_param in zip(ivy_parameter_types, backend_parameter_types))
            # Check return type
            # ivy_return_type = ivy_function.
            # backend_return_type = backend_function.return_type.annotation
            # return_type_pass = ivy_return_type == backend_return_type
            return_type_pass = True
            # Build the report row
            def make_ticks(*args):
                return ["✅" if arg else "❌" for arg in args]
            report += f"| {name} | {' | '.join(make_ticks(True, True, parameter_name_pass, parameter_type_pass, return_type_pass))} |\n"
    return report


def make_detailed_array_api_compliance_report(backend: str) -> str:
    """
    Generate a report of array API compliance for a backend where each row is a single parameter.
    """
    tree = standardised_parse_module(open(IVY_PATH_TEMPLATE.format(filename="manipulation.py")).read())
    # Gather all exported functions for Ivy and the given backend
    ivy_exported_functions = get_exported_functions(IVY_PATH_TEMPLATE.format(filename="manipulation.py"))
    backend_exported_functions = get_exported_functions(BACKEND_PATH_TEMPLATE.format(backend=backend, filename="manipulation.py"))
    # Build the report
    report = f"| Function | Match | Param | Ivy | {backend} |\n"
    report += "| --- | --- | --- | --- | --- |\n"
    for name in sorted(set(ivy_exported_functions.keys() | set(backend_exported_functions.keys()))):
        if name in ivy_exported_functions and name not in backend_exported_functions:
            report += rf"| {name} | ✅ | ❌ | ❌ | ❌ |"
            continue
        if name not in ivy_exported_functions and name in backend_exported_functions:
            report += rf"| {name} | ❌ | ✅ | ❌ | ❌ |"
            continue
        else:
            ivy_function = ivy_exported_functions[name]
            backend_function = backend_exported_functions[name]
            # Get dictionaries of parameters
            ivy_parameters = {param.name.value: tree.code_for_node(param) for param in ivy_function.params.params + ivy_function.params.kwonly_params}
            backend_parameters = {param.name.value: tree.code_for_node(param) for param in backend_function.params.params}
            # Build the row
            for parameter_name, ivy_parameter_type in ivy_parameters.items():
                if parameter_name not in backend_parameters:
                    report += f"| {name} | ❌ | {parameter_name} | ✅ | ❌ |\n"
                    continue
                backend_parameter_type = backend_parameters[parameter_name]
                report += f"| {name} | ✅ | {parameter_name} | ✅ | ✅ |\n"
                name = ""
            for parameter_name in backend_parameters:
                if parameter_name not in ivy_parameters:
                    report += f"| {name} | ❌ | {parameter_name} | ❌ | ✅ |\n"
                name = ""
    return report


if __name__ == "__main__":
    for backend in BACKENDS:
        print(make_array_api_compliance_report(backend))
    for backend in BACKENDS:
        print(make_detailed_array_api_compliance_report(backend))