import requests
import json
import numpy as np

def amtc_freq_from_signal(sig,
                          fps,
                          freq_limit,
                          window_length_in_sec=10,
                          overlap=0.98,
                          nfft=2048,
                          smoothness=1.0,
                          notch_switch=False,
                          notch_in_nbins=3,
                          preprocess_switch=True,
                          preprocess_k=2,
                          preprocess_accum=False,
                          label='Frequency (bpm)'):

    url = 'https://amtc-i3bncenngq-uc.a.run.app/amtc-signal'

    send_dic = {
        "signal": sig.tolist(),
        "window_length_in_sec": window_length_in_sec,
        "overlap":overlap,
        "sample_per_sec": fps,
        "preprocess_k": preprocess_k,
        "preprocess_switch": preprocess_switch,
        "preprocess_accum": preprocess_accum,
        "freq_limit": freq_limit,
        "label": label,
        "notch_switch": notch_switch,
        "notch_in_nbins": notch_in_nbins,
        "nfft": nfft,
        "smoothness": smoothness
    }

    # curl -X POST https://process---amtc-i3bncenngq-uc.a.run.app/amtc-processed -H 'Content-Type: application/json' -d '{"signal":[1,-1,1,-1,1],"fps":2,"freq_limit":[30,90]}'

    x = requests.post(url, json=send_dic)
    x_as_dic = json.loads(x.text)

    spec_res = np.array(x_as_dic["spec_rgb"],
                        dtype=np.uint8)  # uint8 image sent
    rec_time = np.array(x_as_dic["time"], dtype=np.float32)
    rec_est = np.array(x_as_dic["est"], dtype=np.float32)

    spec = np.array(x_as_dic["spec"], dtype=np.float32)  # uint8 image sent
    spec_time = np.array(x_as_dic["spec_time"], dtype=np.float32)
    spec_freq = np.array(x_as_dic["spec_freq"], dtype=np.float32)

    return spec_res, rec_time, rec_est, spec, spec_time, spec_freq


def amtc_freq_from_spectrogram(spec, spec_time, spec_freq, delay, notch_switch,
                               notch_in_nbins=4, smoothness=1.0):

    url = 'https://amtc-i3bncenngq-uc.a.run.app/amtc-spectrogram'

    send_dic = {
        "spec": spec.tolist(),
        "spec_time": spec_time.tolist(),
        "spec_freq": spec_freq.tolist(),
        "delay": delay,
        "smoothness": smoothness,
        "notch_switch": notch_switch,
        "notch_in_nbins": notch_in_nbins,
    }

    x = requests.post(url, json=send_dic)
    x_as_dic = json.loads(x.text)

    spec_rgb = np.array(x_as_dic["spec_rgb"],
                        dtype=np.uint8)  # uint8 image sent
    rec_time = np.array(x_as_dic["time"],
                        dtype=np.float32)  # amtc_process time (5 fps)
    rec_est = np.array(x_as_dic["est"], dtype=np.float32)
    notched = np.array(x_as_dic["notched"], dtype=np.float32)

    tm, hr_remote = rec_time, rec_est

    return spec_rgb, tm, hr_remote, notched
