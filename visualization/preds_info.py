import torch
import numpy as np

class PredsInfo():
    """ Data structure for holding model and preprocessed data for prediction """
    def __init__(self):
        self.ready = 0 # whether both model and data have been loaded
        self.model = [] # loaded model
        self.data = [] # loaded data for model
        self.model_preds = [] # predictions from model/data
        self.preds = [] # loaded predictions
        self.preds_to_plot = [] # predictions that are to be plotted
        self.model_fn = "" # name of the model file
        self.data_fn = "" # name of the data file
        self.preds_fn = "" # name of the loaded predictions file
        self.model_loaded = 0 # if the model has been loaded
        self.data_loaded = 0 # if the data has been loaded
        self.preds_loaded = 0 # if the predictions have been loaded
        self.plot_model_preds = 0 # whether or not to load model_preds into preds_to_plot
        self.plot_loaded_preds = 0 # whether or not to load preds into preds_to_plot
        self.pred_width = 0 # width in samples of each prediction, must be an int
        self.pred_by_chn = 0 # whether or not we are predicting by channel

    def write_data(self, pi2):
        """
        Writes data from pi2 into self.
        """
        self.ready = pi2.ready
        self.model = pi2.model
        self.data = pi2.data
        self.model_preds = pi2.model_preds
        self.preds = pi2.preds
        self.preds_to_plot = pi2.preds_to_plot
        self.model_fn = pi2.model_fn
        self.data_fn = pi2.data_fn
        self.preds_fn = pi2.preds_fn
        self.model_loaded = pi2.model_loaded
        self.data_loaded = pi2.data_loaded
        self.preds_loaded = pi2.preds_loaded
        self.plot_model_preds = pi2.plot_model_preds
        self.plot_loaded_preds = pi2.plot_loaded_preds
        self.pred_width = pi2.pred_width
        self.pred_by_chn = pi2.pred_by_chn

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
        try:
            preds = torch.load(preds_fn)
        except:
            raise Exception("The predictions file could not be loaded.")
        try:
            preds = preds.detach()
        except:
            pass
        preds = np.array(preds)
        ret = self.check_preds_shape(preds, 0, max_time, fs, nchns)
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

        ret = self.check_preds_shape(preds, 1, max_time, fs, nchns)
        return ret

    def check_preds_shape(self, preds, model_or_preds, max_time, fs, nchns):
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
        ret = -1
        self.pred_by_chn = 0 # reset
        if len(preds.shape) == 3:
            if preds.shape[0] == 2:
                preds = preds[1,:,:]
            elif preds.shape[1] == 2:
                preds = preds[:,1,:]
            elif preds.shape[2] == 2:
                preds = preds[:,:,1]
            if len(preds.shape) == 3 and (preds.shape[0] == 1 or preds.shape[1] == 1 or preds.shape[2] == 1):
                preds = np.squeeze(preds)
        dim = len(preds.shape)
        if dim == 1:
            if (fs * max_time) % preds.shape[0] == 0:
                ret = 0
        elif dim == 2:
            if preds.shape[0] < preds.shape[1]:
                preds = preds.T
            if (fs * max_time) % preds.shape[0] == 0:
                if preds.shape[1] == 1:
                    ret = 0
                elif preds.shape[1] == 2:
                    preds = preds[:,1]
                    ret = 0
                else:
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

        if self.pred_by_chn:
            preds_flipped = np.zeros(self.preds_to_plot.shape)
            for i in range(preds_flipped.shape[1]):
                preds_flipped[:,i] += self.preds_to_plot[:,self.preds_to_plot.shape[1] - i - 1]

            preds_mutli_chn = np.zeros((self.preds_to_plot.shape[0], nchns))
            if self.preds_to_plot.shape[1] >= nchns:
                preds_mutli_chn += preds_flipped[:,self.preds_to_plot.shape[1] - nchns:]
            else:
                preds_mutli_chn[:,nchns - self.preds_to_plot.shape[1]:] += preds_flipped

        i = 0
        while i * pw < start_t: # find starting value
            i += 1
        i -= 1
        if i * pw < start_t and (i + 1) * pw > start_t:
            if np.max(self.preds_to_plot[i]) > thresh:
                starts.append(start_t)
                ends.append((i + 1) * pw)
                if self.pred_by_chn:
                    chn_i = preds_mutli_chn[i] > thresh
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
                        chn_i = preds_mutli_chn[i] > thresh
                        chns.append(chn_i)
                    self.preds_to_plot = temp
                    return starts, ends, chns
            if np.max(self.preds_to_plot[i]) > thresh:
                starts.append(i * pw)
                ends.append((i + 1) * pw)
                if self.pred_by_chn:
                    chn_i = preds_mutli_chn[i] > thresh
                    chns.append(chn_i)
            i += 1
        return starts, ends, chns

    def updatePredicted(self, data, max_time, predicted):
        """
        Function used to update self.predicted from plot whenever new data
        is loaded.

        inputs:
            data - the data used for plotting
            predicted - the current value of predicted

        returns:
            1 for predicted = 1, 0 for predicted = 0
        """
        """fs = data.shape[1] / max_time
        nchns = data.shape[0]

        if predicted:
            # check if valid and set plot_model/loaded_preds to 0 if not
            if self.plot_model_preds:
                if ready:
                    ret = self.predict(max_time, fs, nchns)
                    if ret == 0:
                        return 1
                    else:
                        self.plot_model_preds = 0
                else:
                    self.plot_model_preds = 0
            elif self.plot_loaded_preds:
                ret = self.check_preds_shape(self.preds,0,max_time,fs,nchns)
                if ret == 0:
                    return 1
                else:
                    self.plot_loaded_preds = 0"""
        if self.plot_model_preds or self.plot_loaded_preds:
            return 1
        return 0
