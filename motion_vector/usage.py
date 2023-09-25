import argparse
import os

from .GMA.core.network import RAFTGMA
from .GMA.core.utils import frame_utils
from .GMA.core.utils.utils import InputPadder, forward_interpolate

import cv2
import torch
import time

cap = None
vid_id = None


@torch.no_grad()
def create_sintel_submission_vis(model,
                                 output_path,
                                 warm_start=False,
                                 global_ref=False,
                                 fx=1.0,
                                 fy=1.0,
                                 crop=None):
    """ Create submission for the Sintel leaderboard """
    global cap, vid_id
    model.eval()

    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    image1, flow_prev, sequence_prev = None, None, None
    t_start = time.time()
    restored = 0

    while cap.isOpened():

        ret, img = cap.read()
        if ret == False:
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if crop is not None:
            img = img[crop[0]:crop[1], crop[2]:crop[3]]
        img = cv2.resize(img, (0, 0), fx=fx, fy=fy)

        img = torch.from_numpy(img).permute(2, 0, 1).float()  # HxWxC to CxHxW

        if image1 is None:
            image1 = img
            padder = InputPadder(image1.shape)
            image1 = padder.pad(
                image1[None].to(f'cuda:{model.device_ids[0]}'))[0]
            continue
        else:
            image2 = img

        image2 = padder.pad(image2[None].to(f'cuda:{model.device_ids[0]}'))[0]

        frame = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        output_file = os.path.join(output_path, 'frame%04d.flo' % (frame + 1))

        if os.path.exists(output_file):
            restored = restored + 1
        else:
            flow_low, flow_pr = model.module(image1,
                                             image2,
                                             iters=32,
                                             flow_init=flow_prev,
                                             test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            frame_utils.writeFlow(output_file, flow)

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

        if not global_ref:
            image1 = image2

        if frame % 300 == 0:
            print('Processing frame {} out of {}'.format(frame, int(dur)))
            remaining_time = (time.time() - t_start) * (dur / (frame + 1) - 1)
            print('Remaining time: %d mins %d secs' %
                  (remaining_time / 60, remaining_time % 60))
    if restored > 0:
        print(
            "Warning: {} flow files were already available in the target folder. \
        Choose a different folder if you want to create new flow files.".format(
                restored))


def get_gma_mv_result(vid_path, output_path, fx=1.0, fy=1.0, crop=None):

    # vid_path path of the input video
    # output_path folder path where you want to store the optical flow output (folder will be created if needed)

    global cap, vid_id

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default='GMA/checkpoints/gma-sintel.pth',
                        help="restore checkpoint")
    #parser.add_argument('--dataset', default='sintel', help="dataset for evaluation")
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--num_heads',
                        default=1,
                        type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only',
                        default=False,
                        action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content',
                        default=False,
                        action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision',
                        default=True,
                        help='use mixed precision')
    parser.add_argument('--model_name')

    # Ablations
    parser.add_argument(
        '--replace',
        default=False,
        action='store_true',
        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha',
                        default=False,
                        action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument(
        '--no_residual',
        default=False,
        action='store_true',
        help=
        'Remove residual connection. Do not add local features with the aggregated features.'
    )

    args = parser.parse_args("")  # "" as its jupyter notebook

    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(
        torch.load(os.path.join(os.path.dirname(__file__), args.model)))

    model.cuda()
    model.eval()

    assert os.path.exists(vid_path), "incorrect video path"
    vid_id = os.path.basename(vid_path).split('.')[0]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    create_sintel_submission_vis(model,
                                 output_path,
                                 warm_start=False,
                                 global_ref=True,
                                 fx=fx,
                                 fy=fy,
                                 crop=crop)


if __name__ == '__main__':
    # example usage
    get_gma_mv_result('/home/mrahman7/Documents/ratVideos/rat001.MOV',
                      '/home/mrahman7/Documents/of_out/rat001',
                      fx=0.5,
                      fy=0.5)
