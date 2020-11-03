import datetime

class SaveEdfInfo():
    """ Data structure for holding information for saving to edf namely
        the anonymized header """

    def __init__(self):
        # Parameters to change
        self.fn = "" # Full file path to load in header
        self.ptId = "X X X X" + " " * 73
        self.recInfo = "Startdate X X X X" + " " * 63
        self.startDate = "01.01.01"
        self.startTime = "01.01.01"
        self.py_h = 2
        self.pyedf_header = {'technician': '002', 'recording_additional': '', 'patientname': '',
                        'patient_additional': '', 'patientcode': '', 'equipment': '',
                        'admincode': '', 'gender': '',
                        'startdate': datetime.datetime(2001, 1, 1, 1, 1, 1),
                        'birthdate': ''}

    def convertToHeader(self):
        """
        Converts from native EDF format:
        self.data.ptId = file[8:88].decode("utf-8")
        self.data.recInfo = file[88:168].decode("utf-8")
        self.data.startDate = file[168:176].decode("utf-8")
        self.data.startTime = file[176:184].decode("utf-8")
        To a header dict for pyedflib
        """
        ptId_fields = self.ptId.split(" ")
        recInfo_fields = self.recInfo.split(" ")

        self.pyedf_header['patientcode'] = ptId_fields[0]
        self.pyedf_header['gender'] = ptId_fields[1]
        if ptId_fields[2] != "X":
            MONTHS = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP",
                            "OCT","NOV","DEC"]
            month = MONTHS.index(ptId_fields[2].split("-")[1]) + 1
            self.pyedf_header['birthdate'] = datetime.datetime(int(ptId_fields[2].split("-")[2]), month, int(ptId_fields[2].split("-")[0]))
        else:
            self.pyedf_header['birthdate'] = ""
        self.pyedf_header['patientname'] = ptId_fields[3]
        if len(ptId_fields) > 4:
            self.pyedf_header['patient_additional'] = "".join(ptId_fields[4:])
        else:
            self.pyedf_header['patient_additional'] = ""

        self.pyedf_header['admincode'] = recInfo_fields[2]
        self.pyedf_header['technician'] = recInfo_fields[3]
        self.pyedf_header['equipment'] = recInfo_fields[4]
        if len(recInfo_fields) > 5:
            self.pyedf_header['recording_additional'] = "".join(recInfo_fields[5:])
        else:
            self.pyedf_header['recording_additional'] = ""

        yr = int(self.startDate.split(".")[2])
        if yr > 20:
            yr += 1900
        else:
            yr += 2000
        self.pyedf_header['startdate'] = datetime.datetime(yr,
                                            int(self.startDate.split(".")[1]),
                                            int(self.startDate.split(".")[0]),
                                            int(self.startTime.split(".")[0]),
                                            int(self.startTime.split(".")[1]),
                                            int(self.startTime.split(".")[2]))
