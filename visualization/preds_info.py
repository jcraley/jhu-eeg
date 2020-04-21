import torch
import numpy as np

class PredsInfo():
    """ Data structure for holding model and preprocessed data for prediction """

    def __init__(self):
        self.ready = 0
        self.model = []
        self.data = []
        self.preds = []
        self.model_fn = ""
        self.data_fn = ""
        self.preds_fn = ""
        self.model_loaded = 0
        self.data_loaded = 0
        self.preds_loaded = 0
        self.plot_model_preds = 0
        self.plot_preds_preds = 0

    def set_data(self, data_fn):
        self.data = torch.load(data_fn)
        self.data_loaded = 1
        self.update_ready()

    def set_model(self, model_fn):
        self.model = torch.load(model_fn)
        self.model_loaded = 1
        self.update_ready()

    def set_preds(self, preds_fn, max_time):
        """
        Loads predictions.

        returns:
            0 for sucess, -1 if predictions are not the right length
        """
        self.preds = torch.load(preds_fn)
        self.preds = self.preds.detach()
        self.preds = np.array(self.preds)
        dim = len(self.preds.shape)
        if dim == 1:
            if max_time == self.preds.shape[0]:
                self.preds_loaded = 1
                return 0
        elif dim == 2:
            if np.argmax(self.preds.shape) == 1:
                self.preds = self.preds.T
            if max_time == self.preds.shape[0]:
                if self.preds.shape[1] == 2:
                    temp = np.zeros((self.preds.shape[0],1))
                    temp = self.preds[:,0] <= 0.5
                    self.preds_loaded = 1
                    self.preds = temp
                    return 0
        return -1

    def update_ready(self):
        if self.model_loaded and self.data_loaded:
            self.ready = 1
