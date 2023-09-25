import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def detrend(time, signal, w=31, preserve_length=False):
    offset = int(w // 2)
    trend = moving_average(signal, w)
    signal_cropped = signal[offset:-offset] - trend
    if preserve_length:
        signal_same_len = np.concatenate(
            (signal[:offset] - trend[0], signal_cropped,
             signal[-offset:] - trend[-1]))
        return time, signal_same_len
    return time[offset:-offset], signal_cropped


def plot_signals2(sigs, title, subtitles, ylabel=''):
    num_plt = len(sigs)
    fig, axes = plt.subplots(nrows=num_plt, ncols=1, figsize=(12, 6), dpi=100)
    fig.subplots_adjust(hspace=1)
    fig.suptitle(title, fontsize=12)

    sig_num = 0
    sig_all = len(sigs)
    for i in range(num_plt):
        if sig_num >= sig_all:
            break
        n_samples = sigs[sig_num].shape[0]
        time = np.linspace(0, n_samples / 30, n_samples)
        if len(sigs[sig_num].shape) > 1:
            axes[i].plot(time, sigs[sig_num])
        else:
            axes[i].plot(time, sigs[sig_num], color='black')
        axes[i].set_xlabel('Time (sec)')
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(subtitles[i])
        sig_num += 1


def plot_signals(X, title):
    fig = plt.figure()
    channels = X.shape[1]
    n_samples = X.shape[0]
    time = np.linspace(0, n_samples / 30, n_samples)
    fig.suptitle(title, fontsize=12)
    colors = sns.color_palette('husl', n_colors=channels)
    for i, (sig, color) in enumerate(zip(X.T, colors)):
        plt.subplot(channels, 1, i + 1)
        plt.plot(time, sig, color=color)
    plt.show()


def find_matching_time(sig1, sig2, max_shift=60):
    n1, n2 = sig1.shape[0], sig2.shape[0]
    if min(n1, n2) < max_shift:
        raise Exception("Error : min length violation")
    cors = []
    shifts = range(-max_shift, max_shift)
    for d in shifts:
        # shift sig1 d units to the left
        if d >= 0:
            valid_len = min(n1 - d, n2)
            cors.append(
                np.corrcoef(sig1[d:d + valid_len], sig2[:valid_len])[0, 1])
        else:
            valid_len = min(n1, n2 + d)
            cors.append(
                np.corrcoef(sig1[:valid_len], sig2[-d:-d + valid_len])[0, 1])
    d = shifts[np.argmax(cors)]  # matching shift
    if d >= 0:
        valid_len = min(n1 - d, n2)
        sig1, sig2 = sig1[d:d + valid_len], sig2[:valid_len]
    else:
        valid_len = min(n1, n2 + d)
        sig1, sig2 = sig1[:valid_len], sig2[-d:-d + valid_len]
    return sig1, sig2


def filter_zero(x):
    return x[x > 0]


def calc_rmse(err):
    return np.sqrt(np.mean(err * err))


def calc_sd(err):
    return np.sqrt(np.mean(np.power(err - np.mean(err), 2)))


def combine_channels(r, g, b):
    return r + g + b


def plot_spectrogram(spectrogram, spec_time, spec_freq , title=''):
    n_frq = spectrogram.shape[0]
    spec_freq_grid = np.linspace(spec_freq[0], spec_freq[1], n_frq)
    plt.figure(figsize=(10, 4), dpi=80)
    plt.pcolormesh(spec_time[:,0]*60, spec_freq_grid, spectrogram, cmap='jet', shading='auto')
    plt.ylabel('Frequency (bpm)')
    plt.xlabel('Time [sec]')
    plt.title(title)


def plot_time_vs_freq(f_time, freq):
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (bpm)')
    ax.plot(f_time, freq, label='AMTC estimated frequency')
    ax.legend()
    ax.grid()
