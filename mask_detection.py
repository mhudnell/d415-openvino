import pyrealsense2 as rs
import numpy as np
import cv2
import os
import logging as log
import sys
from argparse import ArgumentParser, SUPPRESS
from fps import FPS

OS_CWD = os.getcwd()
sys.path.append("D:\\mhudnell\\repos\\d415_face_mask_ncs2\\FaceMaskDetection")
os.chdir("D:\\mhudnell\\repos\\d415_face_mask_ncs2\\FaceMaskDetection")
from openvino_infer import MaskNetwork
os.chdir(OS_CWD)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)

    return parser


if __name__ == '__main__':
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    # LIBREALSENSE INITIALIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Configure depth and color streams
    image_width = 640
    image_height = 480
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # OPENVINO INITIALIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mask_net = MaskNetwork(args)

    # FPS INITIALIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # initialize fps counting
    frame_count = 0  # counts every 15 frames
    fps_update_rate = 15  # 1 # the interval (of frames) at which the fps is updated
    fps_deque_size = 2  # 5
    fps = FPS(deque_size=fps_deque_size, update_rate=fps_update_rate)
    curr_fps = 0

    try:
        while True:

            # Wait for a coherent pair of frames, and align them
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not depth_frame or not color_frame:  # try until both images are ready
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # print("color_image.shape:", color_image.shape)

            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            # perform inference
            res = mask_net.openvino_infer(color_image_rgb)

            # compute fps
            if frame_count == fps_update_rate:
                frame_count = 0
                fps.update()
                curr_fps = fps.fps()

            # draw fps on image
            cv2.putText(
                color_image_rgb, str(int(curr_fps)),
                (0, 15), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2
            )
            frame_count += 1

            # Display frame, exit if key press
            color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image_bgr)
            k = cv2.waitKey(1)
            # if k != -1:  # exit if key pressed   ESC (k=27)
            if k == 27:  # exit if key pressed   ESC (k=27)
                cv2.destroyAllWindows()
                break

    finally:
        pipeline.stop()
