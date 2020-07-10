import argparse
import pyedflib
from scipy.signal import resample_poly


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
    parser.add_argument('fn_out')
    args = vars(parser.parse_args())
    fn_in = args['fn_in']
    fn_out = args['fn_out']

    signals, signal_headers, header, annotations = read_edf(fn_in)

    # Check for the same sample rate
    SampleRateCheck = True
    for sheader in signal_headers:
        if sheader['sample_rate'] != 256:
            SampleRateCheck = False
        sheader['sample_rate'] = 200

    assert SampleRateCheck, "Channels with different sample rate"

    new_signals = resample_poly(signals, up=25, down=32, axis=1)
    write_edf(fn_out, new_signals, signal_headers, header, annotations)


if __name__ == '__main__':
    main()
