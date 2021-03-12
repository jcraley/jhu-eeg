from copy import deepcopy
import preprocessing.dsp as dsp
import numpy as np

class SignalStatsInfo():
    """ Class to hold relevant information for computing statistics
    """
    def __init__(self):
        self.chn = 0
        self.chn_items = []
        self.fs = -1
        self.fs_bands = {'delta':(0,4),'theta':(4,8),'alpha':(8,14), 'beta':(14,30), 'gamma': (30,self.fs)}

    def _get_power_for_band(self, sig, s, f, band):
        """ Returns the power in the given fs band.

        Args:
            sig: the signal to use
            s: where to start in samples
            f: where to end in samples
            band: which type of band ('delta','theta','alpha','beta','gamma')
        Returns:
            The power of the signal
        """
        lp = self.fs_bands[band][0]
        hp = self.fs_bands[band][1]
        filt_bufs = deepcopy(sig[s:f])
        # LPF
        if lp > 0:
            filt_bufs = dsp.applyLowPass(filt_bufs, self.fs, lp)

        # HPF
        if hp > 0:
            filt_bufs = dsp.applyHighPass(filt_bufs, self.fs, hp)

        if lp == 0 and hp == 0:
            return 0
        
        return np.sum( np.abs( filt_bufs ) ** 2 )
    
    def get_power(self, sig, s, f):
        """ Returns power for all bands.

        Args:
            sig: the signal
            s: where to start in samples
            f: where to end in samples
        Returns:
            alpha, beta, theta, gamma, delta
        """
        bands = ['alpha', 'beta', 'theta', 'gamma', 'delta']
        ret = []
        for b in bands:
            ret.append(self._get_power_for_band(sig, s, f, b))
        
        return ret[0], ret[1], ret[2], ret[3], ret[4]

