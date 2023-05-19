from experiments.experiment import Experiment


class Regression(Experiment):
    def __init__(self, config, debug=False):
        self.debug = debug
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        super().__init__(config, self.debug)
