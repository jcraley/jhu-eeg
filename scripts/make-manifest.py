import glob
import os
import argparse
from preprocessing.edf_loader import EdfLoader


def main():
    # Get the folder
    parser = argparse.ArgumentParser(
        description='Get metadata for a set of EDFs')
    parser.add_argument('EDFfolder', help='Folder containing EDF files')
    args = vars(parser.parse_args())
    EDFfolder = args['EDFfolder']

    EDFfiles = glob.glob(os.path.join(EDFfolder, '*.edf'))
    EDFfiles.sort()

    loader = EdfLoader()
    header = 'fn;fs;duration;nsamples;nchns;nsz;sz_starts;sz_ends;pt_num;onset_zone'
    print(header)
    for edffile in EDFfiles:
        fn = edffile.split('/')[-1]
        eeg_info = loader.load_metadata(edffile)
        str = '{};{};{};{};{};0;[];[];-1;-1'.format(
            fn,
            eeg_info.fs,
            eeg_info.file_duration,
            eeg_info.nsamples[0],
            eeg_info.nchns
        )
        print(str)


if __name__ == '__main__':
    main()
