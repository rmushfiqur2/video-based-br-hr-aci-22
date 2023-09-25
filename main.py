from amtc.usage import amtc_freq_from_signal, amtc_freq_from_spectrogram
from utils import *
import json
import cv2
import os


def read_file(vid_id, collection_id, data_folder, prefix):

    # returns the signal (npy) and corresponding signal recording information (json)
    data_path = data_folder + prefix + vid_id + '_' + str(
        collection_id) + '.npy'
    box_info_path = data_folder + prefix + vid_id + '.json'

    with open(box_info_path, 'r') as fp:
        info = json.load(fp)[str(collection_id)]

    sig = np.load(data_path)

    return sig, info


def quick_peep(info):
    cap = cv2.VideoCapture(info["vid_path"])
    ret, img = cap.read()
    if info["crop"] is not None:
        img = img[info["crop"][0]:info["crop"][1],
                  info["crop"][2]:info["crop"][3]]
    img = cv2.resize(img, (0, 0), fx=info["fx"], fy=info["fy"])

    img = cv2.rectangle(img, (20, 0),(200, 200), (0, 0, 0),-1)  # hide face for 'face002'
    #img = cv2.rectangle(img, (500, 0),(700, 200), (0, 0, 0),-1)  # hide face for 'dog006'
    #img = cv2.rectangle(img, (380, 0),(550, 150), (0, 0, 0),-1)  # hide face for 'dog003'
    img = cv2.rectangle(img, (info["boxes"][0][0], info["boxes"][0][2]),
                        (info["boxes"][0][1], info["boxes"][0][3]), (255, 0, 0),
                        4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.title("quick look of crop region")
    plt.imshow(img)


def get_br(vid_id, collection_id, data_folder, verbose=True, GT_folder=None):

    # inputs
    # vid_id: (string) a title of the video
    # collection_id: (integer) defines a specific signal collection event
    # you can collect signal multiple times from same video, and hence it will have multiple collection ids
    # i.e., you collect signal by selecting a random crop from the video, and repeat such collection 3 times
    # data_folder: storage location of the collected signals
    # verbose: if False spectrogram and other graphs will note be plotted (silent mode)
    # GT folder: folder of the ground truth breathing rate

    # reading the signal and corresponding signal recording information
    sig, info = read_file(vid_id, collection_id, data_folder, 'displacement_')

    # quickly peep the first frame and cropping region if video is stored locally.
    if verbose and os.path.exists(info["vid_path"]):
        quick_peep(info)

    sig_u = sig[:, 0]  # optical flow through x direction (not used)
    sig_v = sig[:, 1]  # optical flow through y direction

    sig = sig_v
    n_samples = sig.shape[0]
    time = np.linspace(0, n_samples / info["fps"], n_samples, endpoint=False)
    t, sig_detrended = detrend(time, sig, w=61, preserve_length=True)
    sig_clipped = np.clip(sig_detrended, -1, 1)

    if verbose:
        plot_signals2([sig, sig_detrended, sig_clipped],
                      '', [
                          'original displacement signal', 'detrended signal',
                          'large motion compensated signal'
                      ],
                      ylabel='Displacement (pixel)')
        plt.show()

    specgrm_png, _, est_br, _, _, _ = amtc_freq_from_signal(
        sig_clipped,
        info["fps"], [15, 80],
        window_length_in_sec=10,
        nfft=2048,
        smoothness=1.0,
        preprocess_switch=True,
        preprocess_k=4,
        preprocess_accum=True,
        label="BR")  # spectrogram, time, estimated frequency

    if verbose:
        plt.figure(figsize=(8, 6), dpi=100)
        plt.imshow(
            specgrm_png
        )  # camera estimated spectrogram with overlaying breath rate trace

    if GT_folder is not None:
        human_recorded_gt = np.load(GT_folder + vid_id + '_result.npy')
        # GT breath rate signal and camera estimated br signal may not start at exactly same point
        # find_matching_time function below uses maximum cross-correlation to sync the starting
        est_br, ref_br = find_matching_time(filter_zero(est_br),
                                            filter_zero(human_recorded_gt[:,
                                                                          1]),
                                            max_shift=60)
        if verbose:
            time = np.linspace(0,
                               est_br.shape[0] / info["fps"],
                               est_br.shape[0],
                               endpoint=False)  # common timing
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.title('BR Estimated Camera and Gold Standard'
                     )  # - dog video '+str(i+1)+' set '+str(j+1))
            plt.xlabel('Time (s)')
            plt.ylabel('BR of dog (bpm)')
            ax.plot(time, est_br, color='r', label='BR Estimated Camera')
            ax.plot(time, ref_br, label='BR Gold Standard')
            ax.legend()
            ax.grid()
            plt.gca().set_ylim(bottom=0)
            plt.show()

        err = est_br - ref_br
        err_per = err / ref_br

        return err, err_per
    return [-1], [-1]  # no ground truth, so error unknown


def get_hr(vid_id, collection_id, data_folder, verbose=True, GT_folder=None):
    # inputs
    # vid_id: (string) a title of the video
    # collection_id: (integer) defines a specific signal collection event
    # you can collect signal multiple times from same video, and hence it will have multiple collection ids
    # i.e., you collect signal by selecting a random crop from the video, and repeat such collection 3 times
    # data_folder: storage location of the collected signals
    # verbose: if False spectrogram and other graphs will note be plotted (silent mode)
    # GT folder: folder of the ground truth heart rate

    # reading the signal and corresponding signal recording information
    sig_rgb, info = read_file(vid_id, collection_id, data_folder, 'rgb_')

    # quickly peep the first frame and cropping region if video is stored locally.
    if verbose and os.path.exists(info["vid_path"]):
        quick_peep(info)

    sig = combine_channels(sig_rgb[:, 0], sig_rgb[:, 1], sig_rgb[:, 2])

    n_samples = sig.shape[0]
    time = np.linspace(0, n_samples / info["fps"], n_samples, endpoint=False)
    t, sig_detrended = detrend(time, sig, w=61, preserve_length=True)
    sig_clipped = np.clip(sig_detrended, -100, 100)

    if verbose:
        ln = 1800  # signal display length
        plot_signals2([
            sig_rgb[:ln, [2, 0, 1]], sig[:ln], sig_detrended[:ln],
            sig_clipped[:ln]
        ], '', [
            'original RGB signal', '1D signal using linear weight (rPhys)',
            'detrended signal', 'motion compensated signal'
        ])
        plt.show()

    specgrm_png, est_time, est_hr, _, _, _ = amtc_freq_from_signal(
        sig_clipped,
        info["fps"],
        [70, 100],
        window_length_in_sec=10,
        nfft=2048,
        smoothness=1.0,
        preprocess_k=2,
        preprocess_switch=True,
        preprocess_accum=False,
    )  # spectrogram, time, estimated frequency

    if verbose:
        plt.figure(figsize=(8, 6), dpi=100)
        plt.imshow(
            specgrm_png
        )  # camera estimated spectrogram with overlaying breath rate trace

    if GT_folder is not None:
        hr_human_e4 = np.load(GT_folder + 'human_hr(e4)_' + vid_id + '.npy')
        if verbose:
            # plot GT hr and estimated hr on same figure
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.title('HR Estimated Camera and Gold Standard')
            plt.xlabel('Time (s)')
            plt.ylabel('HR of human (bpm)')
            ax.plot(est_time, est_hr, label='HR Estimated Camera')
            ax.plot(hr_human_e4[:, 0], label='HR Gold Standard')
            ax.legend()
            ax.grid()
            plt.show()

        # GT heart rate signal and camera estimated hr signal may not start at exactly same point
        # find_matching_time function below uses maximum cross-correlation to sync the starting
        # e4 device has 1 fps sample per second output rate, we match out estimated output to that output rate
        ref_hr, est_hr = find_matching_time(
            hr_human_e4[:, 0],
            moving_average(filter_zero(est_hr),
                           int(info["fps"]))[::int(info["fps"])])

        err = est_hr - ref_hr
        err_per = err / ref_hr

        return err, err_per
    return [-1], [-1]  # no ground truth, so error unknown


def get_and_print_results(video_set, collection_set, br_or_hr, verbose=True):
    # verbose: if set true, it will show step by step processing result. It will pop up as figures, you will need to
    # press 'q' to close the figure and proceed to the next step.
    errs = []  # error
    err_pers = []  # error percentage

    for vid_id in video_set:
        for collection_id in collection_set:
            if br_or_hr=='br':
                err, err_per = get_br(vid_id,
                                      collection_id,
                                      data_folder=signal_collection_folder,
                                      verbose=verbose,
                                      GT_folder='GT/')
            elif br_or_hr=='hr':
                err, err_per = get_hr(vid_id,
                                      collection_id,
                                      data_folder=signal_collection_folder,
                                      verbose=verbose,
                                      GT_folder='GT/')
            else:
                raise Exception("Wrong argument")
            err_pers = np.concatenate((err_pers, err_per))
            errs = np.concatenate((errs, err))

    print('RMSE error: {:.2f} Standard Deviation: {:.2f}'.format(calc_rmse(np.array(errs)), calc_sd(np.abs(errs))))
    print('Relative error MeRate: {:.2f}%'.format(np.mean(np.abs(err_pers)) * 100))
    print('(Percentage) RMSE error: {:.2f}% Standard Deviation: {:.2f}%'.format(calc_rmse(np.array(err_pers)) * 100,
                                                                                calc_sd(np.array(err_pers)) * 100))


if __name__ == "__main__":

    # reproduces the results presented in the ACI 2022 paper (https://arxiv.org/abs/2211.03636)
    signal_collection_folder = "outputs/"
    # necessary signals to compute BR/ HR has already been collected and saved in the outputs/ folder
    # to get the full details of how the signals were collected and saved please see the example.ipynb notebook

    print("BR estimation results")
    video_set = ['dog006', 'dog008', 'dog012']  # distinct videos
    collection_set = [0, 1, 2]  # from each video signal has been collected 3 times independently
    get_and_print_results(video_set, collection_set, 'br', verbose=False)
    # verbose: if set true, it will show step by step processing result. It will pop up as figures, you will need to
    # press 'q' to close the figure and proceed to the next step.

    print("\n\n")

    print("HR estimation results")
    video_set = ['face002', 'face003', 'face004', 'face005'] # distinct videos
    collection_set = [0, 1, 2]  # from each video signal has been collected 3 times independently
    get_and_print_results(video_set, collection_set, 'hr', verbose=False)


