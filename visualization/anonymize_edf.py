from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QProgressDialog

def main(edf_file, output_file, ptId = "X X X X" + " " * 73, recInfo = "Startdate X X X X" + " " * 63,
                                            startDate = "01.01.01", startTime = "01.01.01"):
    anonymizeFile(edf_file, output_file, ptId, recInfo, startDate, startTime)

def anonymizeFile(edf_file, output_file, ptId = "X X X X" + " " * 73,
                    recInfo = "Startdate X X X X" + " " * 63,
                            startDate = "01.01.01", startTime = "01.01.01"):
    """Modify the header of a given edf file

    Allows EDFbrowser to make derivations by setting the max and
    min pysical and digital values to the same things for each EEG channel.

    Get the number of signals in the file, find the start of header sections

    from edf specs:
    HEADER RECORD (we suggest to also adopt the 12 simple additional EDF+ specs)
    8 ascii : version of this data format (0)
    80 ascii : local patient identification (mind item 3 of the additional EDF+ specs)
    80 ascii : local recording identification (mind item 4 of the additional EDF+ specs)
    8 ascii : startdate of recording (dd.mm.yy) (mind item 2 of the additional EDF+ specs)
    8 ascii : starttime of recording (hh.mm.ss)
    8 ascii : number of bytes in header record
    44 ascii : reserved
    8 ascii : number of data records (-1 if unknown, obey item 10 of the additional EDF+ specs)
    8 ascii : duration of a data record, in seconds
    4 ascii : number of signals (ns) in data record
    ns * 16 ascii : ns * label (e.g. EEG Fpz-Cz or Body temp) (mind item 9 of the additional EDF+ specs)
    ns * 80 ascii : ns * transducer type (e.g. AgAgCl electrode)
    ns * 8 ascii : ns * physical dimension (e.g. uV or degreeC)
    ns * 8 ascii : ns * physical minimum (e.g. -500 or 34)
    ns * 8 ascii : ns * physical maximum (e.g. 500 or 40)
    ns * 8 ascii : ns * digital minimum (e.g. -2048)
    ns * 8 ascii : ns * digital maximum (e.g. 2047)
    ns * 80 ascii : ns * prefiltering (e.g. HP:0.1Hz LP:75Hz)
    ns * 8 ascii : ns * nr of samples in each data record
    ns * 32 ascii : ns * reserved
    """

    progress = QProgressDialog("Filtering...", "Cancel", 0, 6)
    progress.setWindowModality(Qt.WindowModal)
    i = 0
    if progress.wasCanceled():
        return
    # Open the file
    # print('Loading file %s' % edf_file)
    progress.setLabelText('Loading file %s' % edf_file)
    with open(edf_file, 'rb') as f:
        file = f.read()
        f.close()
    # edf_file.close()
    file = bytearray(file)
    i += 1
    progress.setValue(i)
    if progress.wasCanceled():
        return

    progress.setLabelText('Anonymizing patient ID')
    # print('Anonymizing patient ID')
    # print(file[8:88])
    file[8:88] = bytes(ptId, 'utf-8')
    # print(ptId)
    # print(file[8:88])
    i += 1
    progress.setValue(i)
    if progress.wasCanceled():
        return

    progress.setLabelText('Anonymizing recording information')
    # print('Anonymizing recording information')
    file[88:168] = bytes(recInfo, 'utf-8')
    i += 1
    progress.setValue(i)
    if progress.wasCanceled():
        return

    progress.setLabelText('Anonymizing startdate')
    # print('Anonymizing startdate')
    file[168:176] = bytes(startDate, 'utf-8')
    i += 1
    progress.setValue(i)
    if progress.wasCanceled():
        return

    progress.setLabelText('Anonymizing starttime')
    # print('Anonymizing starttime')
    file[176:184] = bytes(startTime, 'utf-8')
    i += 1
    progress.setValue(i)
    if progress.wasCanceled():
        return

    progress.setLabelText('Writing file %s' % output_file)
    # print('Writing file %s' % output_file)
    with open(output_file, 'wb') as f:
        f.write(file)
    i += 1
    progress.setValue(i)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('edf_file',
                        help='EDF file to be modified')
    parser.add_argument('output_file',
                        help='Output file for modified EDF file')
    args = parser.parse_args()
    main(args.edf_file, args.output_file)
