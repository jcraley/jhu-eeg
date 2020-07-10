import argparse
import pyedflib
from scipy.signal import resample_poly
import numpy as np


def read_edf(fn):
    f = pyedflib.EdfReader(fn)
    annotations = f.readAnnotations()
    header = f.getHeader()

    ch_nrs = range(f.signals_in_file)
    signal_headers = [f.getSignalHeaders()[c] for c in ch_nrs]
    # Read signals
    signals = []
    for c in ch_nrs:
        signal = f.readSignal(c, digital=False)
        signals.append(signal)
    return signals, signal_headers, header, annotations


def write_edf(edf_file, signals, signal_headers, header, annotations):
    n_channels = len(signal_headers)
    with pyedflib.EdfWriter(edf_file, n_channels=n_channels) as f:
        f.setSignalHeaders(signal_headers)
        f.setHeader(header)
        f.writeSamples(signals)
        for ii in range(len(annotations[0])):
            f.writeAnnotation(annotations[0][ii],
                              annotations[1][ii],
                              annotations[2][ii])
    del f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fn_in')
    parser.add_argument('start', type=int)
    parser.add_argument('end', type=int)
    parser.add_argument('fn_out')
    args = vars(parser.parse_args())
    fn_in = args['fn_in']
    fn_out = args['fn_out']
    start = args['start']
    end = args['end']

    signals, signal_headers, header, annotations = read_edf(fn_in)

    # Get the new annotations
    aidx = np.where((annotations[0] >= start) * (annotations[0] <= end))
    new_annotations = [
        annotations[0][aidx] - start,
        annotations[1][aidx],
        annotations[2][aidx],
    ]

    # Trim the signals
    if isinstance(signals, np.ndarray):
        fs = signal_headers[0]['sample_rate']
        new_signals = signals[:, start*fs:end*fs]
    if isinstance(signals, list):
        new_signals = []
        for signal, signal_header in zip(signals, signal_headers):
            fs = signal_header['sample_rate']
            new_signals.append(signal[start*fs:end*fs])

    # Write
    write_edf(fn_out, new_signals, signal_headers, header, new_annotations)


if __name__ == '__main__':
    main()
