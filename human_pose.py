import argparse
import logging as log
import sys

import cv2
import numpy as np
from openvino.inference_engine import IECore
import pyrealsense2 as rs

from fps import FPS


# get passed in arguments
ap = argparse.ArgumentParser()
ap.add_argument('model-xml-path', type=str, help='path to model .xml')
ap.add_argument('model-bin-path', type=str, help='path to model .bin')
args = vars(ap.parse_args())

if __name__ == '__main__':
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    # LIBREALSENSE INITIALIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Configure depth and color streams
    image_width = 640
    image_height = 480
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)

    # OPENVINO NETWORK INITIALIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ie = IECore()
    net = ie.read_network(model=args['model-xml-path'], weights=args['model-bin-path'])
    exec_net = ie.load_network(network=net, device_name='CPU')
    input_layer_name = next(iter(net.inputs))

    NET_INPUT_HEIGHT_SIZE = 256
    scale = NET_INPUT_HEIGHT_SIZE / image_height

    # Print input info
    print('\nNETWORK INPUTS~~~~~~')
    for layer, data_ptr in net.inputs.items():
        print(
              f'layer name:{data_ptr.name}\n'
              f'  - precision: {data_ptr.precision}\n'
              f'  - shape: {data_ptr.shape}\n'
              f'  - layout: {data_ptr.layout}'
        )

    # Print output info
    print('\nNETWORK OUTPUTS~~~~~~')
    for layer, data_ptr in net.outputs.items():
        print(
              f'layer name:{data_ptr.name}\n'
              f'  - precision: {data_ptr.precision}\n'
              f'  - shape: {data_ptr.shape}\n'
              f'  - layout: {data_ptr.layout}'
        )

    # FPS INITIALIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # initialize fps counting
    frame_count = 0  # counts every 15 frames
    fps_update_rate = 15  # 1 # the interval (of frames) at which the fps is updated
    fps_deque_size = 2  # 5
    fps = FPS(deque_size=fps_deque_size, update_rate=fps_update_rate)
    curr_fps = 0

    # request_id = 0
    try:
        while True:

            # wait for a coherent pair of frames, and align them
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not depth_frame or not color_frame:  # try until both images are ready
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            # print(color_image_rgb.shape)
            color_image_resize = cv2.resize(color_image_rgb, (456, 256))
            # print(color_image_resize.shape)
            color_image_reshape = np.reshape(color_image_resize, (1, 3, 256, 456))
            # print(color_image_reshape.shape)

            # perform SYNC inference
            res = exec_net.infer(inputs={input_layer_name: color_image_reshape})
            pafs = res['Mconv7_stage2_L1']
            heatmaps = res['Mconv7_stage2_L2']

            # perform ASYNC inference
            # res = exec_net.start_async(request_id=request_id, inputs={input_layer_name: color_image_reshape})
            # request_id += 1

            # print(res)
            # print('--')
            # for k, v in res.items():
            #     print(k, v.shape)

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
