class SaveImgInfo():
    """ Data structure for holding information for print preview
        and saving images """

    def __init__(self):
        # Parameters to change
        self.plotAnn = 1 # whether to plot annotations
        self.linethick = 0.5 # line thickness
        self.fontSize = 12 # default text size
        self.plotTitle = 0 # whether to plot a title
        self.title = "" # title
        # Parameters for plotting from parent
        self.data = [] # data for plotting
        self.pi = [] # the plotInfo object
        self.ci = [] # the channelInfo object
        self.predicted = 0 # self.predicted from parent
        self.fs = 0 # the frequency
        self.count = 0 # the count
        self.window_size = 10 # the window_size
        self.y_lim = 150 # y_lim for plotting
        self.thresh = 0.5 # threshold for predictions
