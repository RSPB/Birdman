import os, sys
import numpy as np
import glob
import configobj
import multiprocessing
from validate import Validator
from scipy.io import wavfile
from aubio import onset
from scipy.signal import butter, lfilter
from read_labels import read_labels
from functools import partial


def butter_highpass(highcut, fs, order=12):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='highpass')
    return b, a


def highpass_filter(signal, highcut, sr, order=12):
    b, a = butter_highpass(highcut, sr, order)
    return lfilter(b, a, signal).astype('float32')


def onset_in_call(onset, calls_list, buffer=0):
    for index, call in calls_list.iterrows():
        if call['Time Start'] - buffer >= onset >= call['Time End'] + buffer:
            return call['Species']
    else:
        return None


def get_onsets(signal, sr, params):
    onsets = []

    onset_detector_type = params['OnsetDetector']['type']
    onset_threshold = params['OnsetDetector']['threshold']
    onset_silence_threshold = params['OnsetDetector']['silence_threshold']

    win = params['Signal']['win']
    hop = win // params['Signal']['hop_factor']

    min_duration_s = params['Signal']['min_duration_s']

    onset_detector = onset(onset_detector_type, win, hop, sr)
    onset_detector.set_threshold(onset_threshold)
    onset_detector.set_silence(onset_silence_threshold)
    onset_detector.set_minioi_s(min_duration_s)

    signal_windowed = np.array_split(signal, np.arange(hop, len(signal), hop))

    for frame in signal_windowed[:-1]:
        if onset_detector(frame):
            onsets.append(onset_detector.get_last_s())
    return onsets


def get_chunks(onsets, max_duration_s):
    chunks_s = []

    for onset, next_onset in zip(onsets, onsets[1:]):
        interval = next_onset - onset
        cut = next_onset if interval < max_duration_s else onset + max_duration_s
        chunks_s.append((onset, cut))

    return chunks_s


def chop_wave(config, labels, wave_path):
    print(wave_path)
    sr, signal = wavfile.read(wave_path)
    signal_norm = signal.astype('float32') / config['Signal']['scaling_factor']
    signal_filtered = highpass_filter(signal_norm, config['Signal']['highpass_cut'], sr)
    onsets = get_onsets(signal_filtered, sr, config)
    chunks_s = get_chunks(onsets, config['Signal']['max_duration_s'])
    filename_noext = os.path.splitext(os.path.basename(wave_path))[0]
    for start, end in chunks_s[1:]:

        dirname = os.path.join(config['Data']['rootdir'], config['Data']['outdir'])

        call = onset_in_call(start, labels[filename_noext], buffer=0) if labels else None
        if call:
            chunk_name = '{}_{:07.3f}_{:07.3f}_{}.wav'.format(filename_noext, start, end, call)
        else:
            chunk_name = '{}_{:07.3f}_{:07.3f}.wav'.format(filename_noext, start, end)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk_path_out = os.path.join(dirname, chunk_name)
        wavfile.write(chunk_path_out, sr, signal[start_sample: end_sample])


def main():
    config = configobj.ConfigObj('config.ini', configspec='configspec.ini')
    validation_successful = config.validate(Validator())
    if not validation_successful:
        print('Params incorrect')
        sys.exit(1)

    datadir = os.path.join(config['Data']['rootdir'], config['Data']['waves'])
    labels = read_labels(os.path.join(config['Data']['rootdir'], config['Data']['labels_xls'])) if 'labels_xls' in config['Data'] else None

    chop_wrapper = partial(chop_wave, config, labels)

    pool = multiprocessing.Pool()
    pool.map(chop_wrapper, glob.glob(datadir + '*.wav'))
    pool.close()
    pool.join()

    # for wave_path in glob.glob(datadir + '*.wav'):
    #     chop_wave(config, labels, wave_path)




if __name__ == '__main__':
    main()





