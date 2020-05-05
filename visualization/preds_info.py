import torch
import numpy as np

class PredsInfo():
    """ Data structure for holding model and preprocessed data for prediction """

    def __init__(self):
        self.ready = 0
        self.model = []
        self.data = []
        self.model_preds = []
        self.preds = []
        self.preds_to_plot = []
        self.model_fn = ""
        self.data_fn = ""
        self.preds_fn = ""
        self.model_loaded = 0
        self.data_loaded = 0
        self.preds_loaded = 0
        self.plot_model_preds = 0
        self.plot_preds_preds = 0
        self.pred_width = 0 # width in samples of each prediction, must be an int
        self.pred_by_chn = 0

    def set_data(self, data_fn):
        self.data = torch.load(data_fn)
        self.data_loaded = 1
        self.update_ready()

    def set_model(self, model_fn):
        self.model = torch.load(model_fn)
        self.model_loaded = 1
        self.update_ready()

    def set_preds(self, preds_fn, max_time, fs, nchns):
        """
        Loads predictions.

        returns:
            0 for sucess, -1 if predictions are not the right length
            predictions must be for an integer number of samples in the file
        """
        preds = torch.load(preds_fn)
        try:
            preds = preds.detach()
        except:
            pass
        preds = np.array(preds)
        ret = self._check_preds_shape(preds,0, max_time, fs, nchns)
        return ret

    def predict(self, max_time, fs, nchns):
        """
        Loads model, passes data through the model to get binary seizure predictions

        inputs:
            data - the pytorch tensor, fully preprocessed
            model_fn - filename of the model to load

        returns:
            0 for sucess, -2 for failure to pass through the predict function,
            -1 for incorrect size
        """
        try:
            preds = self.model.predict(self.data)
            preds = np.array(preds)
        except:
            return -2

        ret = self._check_preds_shape(preds,1, max_time, fs, nchns)
        return ret

    def _check_preds_shape(self, preds, model_or_preds, max_time, fs, nchns):
        """
        Checks whether the predictions are the proper size.
        Samples in the file must be an integer multiple of length.
        The other dimension must be either 1, 2 (which will be collapsed) or nchns

        Input:
            preds - the predictions (np array)
            model_or_preds - 1 for model loaded, 0 for loaded predictions
            max_time - amount of seconds in the .edf file
            nchns - number of channels
        Returns:
            0 for sucess, -1 for incorrect length
        """
        # check size
        dim = len(preds.shape)
        ret = -1
        self.pred_by_chn = 0 # reset
        if dim == 1:
            if (fs * max_time) % preds.shape[0] == 0:
                ret = 0
        elif dim == 2:
            if preds.shape[0] == nchns:
                preds = preds.T
            if (fs * max_time) % preds.shape[0] == 0:
                if preds.shape[1] == 1:
                    ret = 0
                elif preds.shape[1] == 2:
                    preds = preds[:,1]
                    ret = 0
                elif preds.shape[1] == nchns:
                    self.pred_by_chn = 1
                    ret = 0
        if ret == 0:
            self.pred_width = fs * max_time / preds.shape[0]
            if model_or_preds: # model
                self.model_preds = preds
            else: # loaded_preds
                self.preds_loaded = 1
                self.preds = preds
        return ret

    def update_ready(self):
        if self.model_loaded and self.data_loaded:
            self.ready = 1

    def compute_starts_ends_chns(self, thresh, count, ws, fs, nchns):
        """
        Computes start / end / chn values of predictions in given window in samples

        Input:
            thresh - the threshold
            count - the current time in seconds
            ws - the current window_size in seconds
            fs - the frequency
        Output:
            starts - the start times
            ends - the corresponding end times
            chns - the channel to plot the given prediction
        """
        start_t = count * fs
        end_t = start_t + ws * fs
        starts = []
        ends = []
        chns = []
        start_pred_idx = 0
        end_pred_idx = 0
        pw = self.pred_width
        if len(self.preds_to_plot.shape) > 1 and self.preds_to_plot.shape[1] != nchns:
            return starts, ends, chns
        i = 0
        while i * pw < start_t: # find starting value
            i += 1
        i -= 1
        if i * pw < start_t and (i + 1) * pw > start_t:
            if np.max(self.preds_to_plot[i]) > thresh:
                starts.append(start_t)
                ends.append((i + 1) * pw)
                if self.pred_by_chn:
                    chn_i = self.preds_to_plot[i] > thresh
                    chns.append(chn_i)
            i += 1
        while i * pw < start_t: # find starting value
            i += 1
        while i * pw < end_t:
            if (i + 1) * pw > end_t:
                if np.max(self.preds_to_plot[i]) > thresh:
                    starts.append(i * pw)
                    ends.append(end_t)
                    if self.pred_by_chn:
                        chn_i = self.preds_to_plot[i] > thresh
                        chns.append(chn_i)
                    self.preds_to_plot = temp
                    return starts, ends, chns
            if np.max(self.preds_to_plot[i]) > thresh:
                starts.append(i * pw)
                ends.append((i + 1) * pw)
                if self.pred_by_chn:
                    chn_i = self.preds_to_plot[i] > thresh
                    chns.append(chn_i)
            i += 1
        return starts, ends, chns
