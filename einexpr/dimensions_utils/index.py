import uuid

class Index:
    pass


class NamedIndex(Index):
    def __init__(self, name, version=None):
        self.name = name
        self.version = version
        
    def get_base(self):
        return self.name
    
    def get_version(self):
        return self.version
    
    def __repr__(self):
        if self.version is not None:
            return f'{type(self).__name__}({self.name!r}, {self.version!r})'
        else:
            return f'{type(self).__name__}({self.name!r})'
        
    def __str__(self):
        if self.version is not None:
            return f'{self.name}[{self.version}]'
        else:
            return f'{self.name}'
    
    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name and self.version == other.version
    
    def __hash__(self):
        return hash((self.name, self.version))


class AnonymousIndex(Index):
    def __init__(self, version=None):
        self.id = uuid.uuid4()
        
    def get_base(self):
        return self.id
    
    def get_version(self):
        return 0
    
    def __repr__(self):
        return f'{type(self).__name__}({self.id!r})'
    
    def equals(self, other):
        return type(self) == type(other) and self.id == other.id
    
    def __hash__(self):
        return hash(self.id)