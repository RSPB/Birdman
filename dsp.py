import os
import numpy as np
import glob
import configobj
import aubio
from validate import Validator
from scipy.io import wavfile
from aubio import onset
from scipy.signal import butter, lfilter
from read_labels import read_labels
from functools import partial

def _find_pitches(win, pitch_o, tolerance):
    pitches = []
    for frame_idx, frame in enumerate(win[:-1]):
        pitch = pitch_o(frame)[0]
        confidence = pitch_o.get_confidence()
        if confidence > tolerance:
            pitches.append((frame_idx, pitch))
    return pitches


def get_pitch(signal, sr, block_size, hop, tolerance = 0.7, unit = 'seconds'):
    pitch_o = aubio.pitch("yin", block_size, hop, sr)
    pitch_o.set_unit('Hz')
    pitch_o.set_tolerance(tolerance)
    signal = signal.astype('float32')
    signal_win = np.array_split(signal, np.arange(hop, len(signal), hop))

    pitches = _find_pitches(signal_win, pitch_o, tolerance)

    if unit == 'seconds':
        pitches = [(frame_idx * hop / sr, value) for frame_idx, value in pitches]
    elif unit == 'samples':
        pitches = [(frame_idx * hop, value) for frame_idx, value in pitches]
    else:
        raise NotImplemented('Unit %s is not implemented', unit)

    if not pitches:
        raise ValueError('No pitches!')

    return pitches


def _butter_highpass(highcut, fs, order=6):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='highpass')
    return b, a

def _butter_lowpass(lowcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='lowpass')
    return b, a


def lowpass_filter(signal, sr, lowcut, order=12):
    b, a = _butter_lowpass(lowcut, sr, order)
    return lfilter(b, a, signal).astype('float32')


def highpass_filter(signal, sr, highcut, order=12):
    b, a = _butter_highpass(highcut, sr, order)
    return lfilter(b, a, signal).astype('float32')



def onset_in_call(onset, calls_list, buffer=0):
    for index, call in calls_list.iterrows():
        if call['Time Start'] - buffer <= onset <= call['Time End'] + buffer:
            return call['Species']
    else:
        return None


def get_onsets_config(signal, sr, params):
   return get_onsets(signal=signal,
               sr=sr,
               nfft=params['Signal']['win'],
               hop=params['Signal']['win'] // params['Signal']['hop_factor'],
               onset_detector_type=params['OnsetDetector']['type'],
               onset_threshold=params['OnsetDetector']['threshold'],
               onset_silence_threshold=params['OnsetDetector']['silence_threshold'],
               min_duration_s=params['Signal']['min_duration_s'])


def get_onsets(signal, sr, nfft, hop, onset_detector_type, onset_threshold=None,
               onset_silence_threshold=None, min_duration_s=None, unit='s'):
    onsets = []

    onset_detector = onset(onset_detector_type, nfft, hop, sr)
    if onset_threshold:
        onset_detector.set_threshold(onset_threshold)
    if onset_silence_threshold:
        onset_detector.set_silence(onset_silence_threshold)
    if min_duration_s:
        onset_detector.set_minioi_s(min_duration_s)

    signal_windowed = np.array_split(signal, np.arange(hop, len(signal), hop))

    for frame in signal_windowed[:-1]:
        if onset_detector(frame):
            if unit in {'seconds', 'second', 's'}:
                onsets.append(onset_detector.get_last_s())
            elif unit in {'sample', 'samples'}:
                raise NotImplemented('Yeah, I was planning to add that')
                # onsets.append(onset_detector.get_last())
            else:
                raise ValueError('Unknown unit provided to the onset detector')

    return onsets[1:]

def get_slices_config(signal, sr, params):
    return get_slices(signal=signal,
               sr=sr,
               nfft=params['Signal']['win'],
               hop=params['Signal']['win'] // params['Signal']['hop_factor'],
               onset_detector_type=params['OnsetDetector']['type'],
               onset_threshold=params['OnsetDetector']['threshold'],
               onset_silence_threshold=params['OnsetDetector']['silence_threshold'],
               min_duration_s=params['Signal']['min_duration_s'],
               unit='s')

def get_slices(signal, sr, nfft, hop, onset_detector_type, onset_threshold=None, onset_silence_threshold=None,
               min_duration_s=None, max_duration_s=None, method='fixed', unit='s'):
    slices = []

    onsets_fw = np.array(get_onsets(signal, sr, nfft, hop, onset_detector_type, onset_threshold,
                                    onset_silence_threshold, min_duration_s, unit))

    if method == 'dynamic':
        onsets_bw = np.array(get_onsets(signal[::-1], sr, nfft, hop, onset_detector_type, onset_threshold,
                                             onset_silence_threshold, min_duration_s, unit))
        onsets_bw_reversed = (len(signal) - onsets_bw)[::-1]


        for onset_fw in onsets_fw:
            nearest_onset_backward_condition = onset_fw < onsets_bw_reversed - min_duration_s
            if nearest_onset_backward_condition[-1] == False: # no backward onset detected
                break
            else:
                idx_onset_backward = np.argmax(nearest_onset_backward_condition)
                offset = onsets_bw_reversed[idx_onset_backward]
                if max_duration_s:
                    if onset_fw + max_duration_s < offset:
                        offset = onset_fw + max_duration_s
                slices.append((onset_fw, offset))

    if method == 'fixed':
        for onset, next_onset in zip(onsets_fw, onsets_fw[1:]):
            interval = next_onset - onset
            cut = next_onset if interval < max_duration_s else onset + max_duration_s
            slices.append((onset, cut))

    return slices

def get_chunks(onsets, max_duration_s):
    chunks_s = []

    for onset, next_onset in zip(onsets, onsets[1:]):
        interval = next_onset - onset
        cut = next_onset if interval < max_duration_s else onset + max_duration_s
        chunks_s.append((onset, cut))

    return chunks_s


def chop_wave(config, labels, wave_path):
    sr, signal = wavfile.read(wave_path)
    signal_norm = signal.astype('float32') / config['Signal']['scaling_factor']
    signal_filtered = highpass_filter(signal_norm, sr=sr, highcut=config['Signal']['highpass_cut'])
    onsets = get_onsets_config(signal_filtered, sr, config)
    # get_slices_config(signal_filtered, sr=sr, params=config)
    chunks_s = get_chunks(onsets, config['Signal']['max_duration_s'])
    filename_noext = os.path.splitext(os.path.basename(wave_path))[0]
    dirname = os.path.join(config['Data']['rootdir'], config['Data']['outdir'])

    for start, end in chunks_s[1:]:
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
        raise ValueError('Params incorrect')

    datadir = os.path.join(config['Data']['rootdir'], config['Data']['waves'])
    labels = read_labels(os.path.join(config['Data']['rootdir'], config['Data']['labels_xls'])) if 'labels_xls' in config['Data'] else None

    # chop_wrapper = partial(chop_wave, config, labels)
    #
    # pool = multiprocessing.Pool()
    # pool.map(chop_wrapper, glob.glob(datadir + '*.wav'))
    # pool.close()
    # pool.join()

    for wave_path in glob.glob(datadir + '*.wav'):
        chop_wave(config, labels, wave_path)




if __name__ == '__main__':
    main()





