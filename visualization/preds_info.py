import torch

class PredsInfo():
    """ Data structure for holding model and preprocessed data for prediction """

    def __init__(self):
        self.ready = 0
        self.model = []
        self.data = []
        self.model_loaded = 0
        self.data_loaded = 0

    def set_data(self, data_fn):
        self.data = torch.load(data_fn)
        self.data_loaded = 1
        self.update_ready()

    def set_model(self, model_fn):
        self.model = torch.load(model_fn)
        self.model_loaded = 1
        self.update_ready()

    def update_ready(self):
        if self.model_loaded and self.data_loaded:
            self.ready = 1
