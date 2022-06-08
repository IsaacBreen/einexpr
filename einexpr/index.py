class Index:
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