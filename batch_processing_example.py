# This example shows how to process multiple video files in a batch
# applicable for both heart rate and breathing rate estimation by changing the variable "what_to_collect"

from motion_vector.usage import get_gma_mv_result
from collect_subtle_signal import collect_signals
from main import get_br, get_hr
import pandas as pd

video_set = ['dog006', 'dog008', 'dog012']
video_set = ['dog006', 'dog006 (copy)']
video_ext = '.mp4'

fx, fy = 0.5, 0.5  # your choice  fx=0.5 will reduce the width of the video by half (it fastens the OF calculation)
what_to_collect = 'displacement'  # 'displacement' for breathing rate. 'rgb' for heart rate

video_folder = '/home/mrahman7/Documents/'
of_folder = '/home/mrahman7/Documents/of_out/' # change this to a folder name of your choice (will be generated)
signal_collection_folder = "outputs/" # will be generated
collection_id = 0 # change it if you do not want to overwrite a previously drawn box for a video file
# collection_id is for your convenience, you can save multiple result (drawing different regions)
# for a video file with different collection_ids


for video_id in video_set:
    video_path = video_folder + video_id + video_ext  # change this to the video you want to analyze
    flow_path = of_folder + video_id  # change this to a folder name of your choice (will be generated)

    print("working on video id: " + video_id)

    collect_signals(video_id,
                    video_path,
                    flow_path,
                    what_to_collect,
                    fx=fx, fy=fy, crop=None, start_sec=0, end_sec=None, set_id=collection_id,
                    apply_mv=True, vis_on=True, graph_on=True, radius=2, output_folder=signal_collection_folder,
                    dry_run=True)

for video_id in video_set:
    video_path = video_folder + video_id + video_ext # change this to the video you want to analyze
    flow_path = of_folder + video_id  # change this to a folder name of your choice (will be generated)

    get_gma_mv_result(video_path, flow_path, fx=fx, fy=fy)

    collect_signals(video_id,
                    video_path,
                    flow_path,
                    what_to_collect,
                    fx=fx, fy=fy, crop=None, start_sec=0, end_sec=None, set_id=collection_id,
                    apply_mv=True, vis_on=True, graph_on=True, radius=2, output_folder=signal_collection_folder,
                    wet_run=True)

    if what_to_collect == 'displacement':
        est_time, est_rate, est_err, err_per = get_br(video_id, collection_id, data_folder=signal_collection_folder, verbose=False,
                                                GT_folder=None)  # GT_folder = '/GT' if you have gt file else None
    else:
        est_time, est_rate, est_err, err_per = get_hr(video_id, collection_id, data_folder=signal_collection_folder, verbose=False,
                                                GT_folder=None)  # GT_folder = '/GT' if you have gt file else None

    list_dict = {'time': est_time, 'est_rate(bpm)': est_rate}
    df = pd.DataFrame(list_dict)
    df.to_csv(signal_collection_folder + 'displacement_rate' + video_id + '_' + str(collection_id) + '.csv', index=False)