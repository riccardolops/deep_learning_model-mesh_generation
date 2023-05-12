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
            return AVT()
