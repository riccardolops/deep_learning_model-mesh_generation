class DatasetFactory:
    def __init__(self, dataset_name, root, splits, transform=None, filename=None):
        self.dataset_name = dataset_name
        self.root = root
        self.splits = splits
        self.transform = transform
        self.filename = filename
    def get(self):
        if self.dataset_name == 'AVT':
            from .AVT import AVT
            return AVT(self.root, self.splits, self.transform)
        elif self.dataset_name == 'Heart':
            from .Heart import Heart
            return Heart(self.root, self.splits, self.transform)
        elif self.dataset_name == 'Maxillo':
            from .Maxillo import Maxillo
            return Maxillo(self.root, self.filename, self.splits, self.transform)
        else:
            raise ValueError(f'Missing dataset: {self.dataset_name} ')
