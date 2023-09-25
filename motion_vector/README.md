# Application of Global Motion Aggregation

This repository is used for finding the motion vector between two frames of a video.

The original GitHub repository can be found here:

[Learning to Estimate Hidden Motions with Global Motion Aggregation](https://github.com/zacjiang/GMA)<br/>

For our project of remote physiological sensing (heart rate or breath rate estimation) we have used the "gma-sintel.pth" weight provided in the original repository.

Settings of GMA used in our application:

`warm start: False`, although it doesn't matter much.

## Environments
The code has been tested for python 3.7, 3.8 and pytorch 1.8.0, 1.13.1
The inner versions should also work.
```Shell
conda create --name gma python==3.7
conda activate gma
pip install -r requirements.txt
```
## Usage
The code has been adapted for simple application given the full path of input video file and the intended output folder.

For the below example if `rat001.MOV` has 1000 frames, after running the code successfully it will generate 999 .flo files inside `/home/mrahman7/Documents/mv_out/rat001` folder.
```Shell
from usage import get_gma_mv_result
get_gma_mv_result('/home/mrahman7/Documents/ratVideos/rat001.MOV',
                  '/home/mrahman7/Documents/mv_out/rat001')
```
`frame0002.flo` is the motion vector of the 2nd frame comparing with the first frame.
`frame0999.flo` is the motion vector of the 1000th frame comparing with the first frame.

If the video frames have the size of (height, width) = (480, 720) then each `.flo` file will be a numpy array of shape (2, 280, 720) containing motion vector in x and y direction for each pixel location.

flow files can be read using the `readFlow` file given inside the `GMA.core.utils.frame_utils.py` file.

## Optional arguments

Optionally we can crop the video frames or resize the video frames before motion vector calculation.

The below code block will crop the video frames as (y1, y2, x1, x2) = (30, 270, 120, 600) and then it will resize width and height as half. So, the final frames will be of sized height 120, width 240.
```Shell
from usage import get_gma_mv_result
get_gma_mv_result('/home/mrahman7/Documents/ratVideos/rat001.MOV',
                  '/home/mrahman7/Documents/mv_out/rat001', crop=[30, 270, 120, 600], fx=0.5, fy=0.5)
```
## Hardware
The code has been tested with an NVIDIA 1080 Ti GPU. The average dedicated memory usage was around 2 GB and GPU utilization was 80%. It takes a while to generate the output.

## Acknowledgement
The codebase is from [GMA](https://github.com/zacjiang/GMA). We
thank the authors for the contribution.
