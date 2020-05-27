class SpecInfo():
    """ Data structure for holding information for spectrograms """

    def __init__(self):
        self.data = [] # data for plotting
        self.plotSpec = 0 # whether or not to plot spectrograms
        self.chnPlotted = -1
