import re
from dataclasses import dataclass, field
from typing import List, Set


def make_empty_signature_str(n_inputs):
    return ','.join(['()'] * n_inputs) + '->()'

# @dataclass
class UfuncSignatureDimension:
    name: str
    optional: bool = field(default=True)
    
    def __init__(self, name, optional=True):
        assert isinstance(name, str)
        assert isinstance(optional, bool)
        self.name = name
        self.optional = optional
    
    def __repr__(self):
        if self.optional:
            return f'{self.name}?'
        else:
            return f'{self.name}'

    def __str__(self):
        if self.optional:
            return f'{self.name}?'
        else:
            return f'{self.name}'
    
            
# @dataclass
class UfuncSignatureDimensions:
    dims: List[UfuncSignatureDimension]
    
    def __init__(self, dims):
        assert isinstance(dims, list)
        self.dims = dims
    
    def get_dims(self):
        return [dim.name for dim in self.dims]
    
    def __iter__(self):
        return iter(self.dims)
    
    def __len__(self):
        return len(self.dims)
    
    def __repr__(self):
        return '(' + ','.join(str(dim) for dim in self.dims) + ')'

    def __str__(self):
        return '(' + ','.join(str(dim) for dim in self.dims) + ')'

    def concretise_optionals(self, optional_set: Set[str]):
        """
        Remove inactive optionals and replace active optionals with non-optionals.
        """
        new_dims = []
        for dim in self.dims:
            if not dim.optional or dim.name in optional_set:
                new_dims.append(UfuncSignatureDimension(dim.name, False))
        return UfuncSignatureDimensions(new_dims)
    

# @dataclass
class UfuncSignature:
    input_dims: List[UfuncSignatureDimensions]
    output_dims: UfuncSignatureDimensions
    
    def __init__(self, input_dims, output_dims):
        assert isinstance(input_dims, list)
        assert isinstance(output_dims, UfuncSignatureDimensions)
        self.input_dims = input_dims
        self.output_dims = output_dims
    
    def get_dims(self):
        names = []
        for dims in self.input_dims:
            for dim in dims.dims:
                if dim.name not in names:
                    names.append(dim.name)
        return names
    
    def to_tuple(self):
        return (tuple(inp.get_dims() for inp in self.input_dims), self.output_dims.get_dims())
    
    def __repr__(self):
        return ','.join(str(dim) for dim in self.input_dims) + '->' + str(self.output_dims)

    def __str__(self):
        return ','.join(str(dim) for dim in self.input_dims) + '->' + str(self.output_dims)

    def concretise_optionals(self, optional_set):
        """
        Remove inactive optionals and replace active optionals with non-optionals.
        """
        return UfuncSignature(
            [dims.concretise_optionals(optional_set) for dims in self.input_dims],
            self.output_dims.concretise_optionals(optional_set)
        )


def parse_ufunc_signature(signature):
    """
    Parse a signature string into a tree.
    
    Parameters
    ----------
    signature : str
        The signature string to parse.
    
    Returns
    -------
    tree : tuple
        A tuple representing the parsed signature.
    
    Examples
    --------
    >>> parse_ufunc_signature('(m?,n),(n,p?)->(m?,p?)')
    {
        UfuncSignature(
            input_dims=[
                UfuncSignatureDimensions([
                    UfuncSignatureDimension(name='m', optional=True),
                    UfuncSignatureDimension(name='n', optional=False)
                ]),
                UfuncSignatureDimensions([
                    UfuncSignatureDimension(name='n', optional=False),
                    UfuncSignatureDimension(name='p', optional=True)
                ])
            ]),
            output_dims=UfuncSignatureDimensions([
                UfuncSignatureDimension(name='m', optional=True),
                UfuncSignatureDimension(name='n', optional=False)
            ])
        )
    """
    # Verify that the user has passed a valid signature and not a None value.
    if signature is None:
        raise ValueError('The signature must not be None. Do you want to parse the signature of an element-wise ufunc (e.g. np.add)? If so, you cannot pass the signature directly as UfuncSignature(np.add.signature), since the value of np.add.signature is None. The reason is that it is not possible for me to infer how many arguments you want to call such a ufunc. You are responsible for constructing an element-wise ufunc with your desired number of arguments. For example, with for a two-argument signature you should pass \'(),()->()\'.')
    # Split the signature into its parts (e.g. '(m?,n),(n,p?)->(m?,p?)' to ['(m?,n)', '(n,p?)', '(m?,p?)'])
    parts = re.split(r'->', signature)
    if len(parts) == 1:
        parts.append('')
    if len(parts) != 2:
        raise ValueError(f'Invalid signature: {signature}')
    
    # Parse the input and output parts
    inputs = [parse_ufunc_signature_part(part) for part in re.findall(r"\([^)]*\)", parts[0])]
    outputs = parse_ufunc_signature_part(parts[1])
    
    return UfuncSignature(inputs, outputs)


def parse_ufunc_signature_part(part):
    """
    Parse a signature part string into a tree.
    
    Parameters
    ----------
    part : str
        The signature part string to parse.
    
    Returns
    -------
    tree : tuple
        A tuple representing the parsed signature part.
    
    Examples
    --------
    >>> parse_ufunc_signature_part('(m?,n)')
    [
        UfuncSignatureDimension(name='m', optional=True),
        UfuncSignatureDimension(name='n', optional=False)
    ]
    """
    dims = []
    for match in re.finditer(r'[a-zA-Z]\??', part):
        dims.append(UfuncSignatureDimension(name=match.group(0)[0], optional=match.group(0).endswith('?')))
    return UfuncSignatureDimensions(dims)

