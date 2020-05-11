class FilterInfo():
    """ Data structure for holding information for filtering """

    def __init__(self):
        self.fs = 0
        self.hp = 2
        self.lp = 30
        self.notch = 60
        self.do_lp = 1
        self.do_hp = 1
        self.do_notch = 0
        self.filter_canceled = 0
