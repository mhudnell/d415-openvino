import pyrealsense2 as rs
import numpy as np
import cv2
import os
import math
import logging as log
import sys
import time
from argparse import ArgumentParser, SUPPRESS
from fps import FPS
from openvino.inference_engine import IECore

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

    # LIBREALSENSE INITIALIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

    # OPENVINO INITIALIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model_xml = "./FaceMaskDetection/model/face_mask_detection.xml"
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    # assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    # assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    # net.batch_size = len(args.input)

    # Read and pre-process input images
    n, c, h, w = net.inputs[input_blob].shape
    print(net.inputs[input_blob].shape)
    # images = np.ndarray(shape=(n, c, h, w))
    # for i in range(n):
    #     image = cv2.imread(args.input[i])
    #     if image.shape[:-1] != (h, w):
    #         log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
    #         image = cv2.resize(image, (w, h))
    #     image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    #     images[i] = image
    log.info("Batch size is {}".format(n))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    # Start sync inference
    


    # FPS INITIALIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # initialize fps counting
    frame_count = 0 # counts every 15 frames
    fps_update_rate = 15 # 1 # the interval (of frames) at which the fps is updated
    fps_deque_size = 2 # 5
    fps = FPS(deque_size=fps_deque_size, update_rate=fps_update_rate)
    curr_fps = 0

    try:
        while True:

            ###### Wait for a coherent pair of frames, and align them
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not depth_frame or not color_frame: # try until both images are ready
                continue
                
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # print("color_image.shape:", color_image.shape)

            # perform inference
            image_resized = cv2.resize(color_image, (260, 260))
            # print("image_resized.shape:", image_resized.shape)
            image_np = image_resized / 255.0  # 归一化到0~1
            image_exp = np.expand_dims(image_np, axis=0)

            image_transposed = image_exp.transpose((0, 3, 1, 2))
            # print("image_transposed.shape:", image_transposed.shape)

            # log.info("Starting inference in synchronous mode")
            start = time.time()
            res = exec_net.infer(inputs={input_blob: image_transposed})
            end = time.time()
            print(f"time elapsed: {(end-start) * 1000}")
            # print("res.shape:", res.shape)
            print(res)

            # compute fps
            if frame_count == fps_update_rate:
                frame_count = 0
                fps.update()
                curr_fps = fps.fps()

            # draw fps on image
            cv2.putText(
                color_image, str(int(curr_fps)),
                (0, 15), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2
            )
            frame_count += 1

            ###### Display frame, exit if key press
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            k = cv2.waitKey(1)  #1
            if k != -1:  # exit if key pressed   ESC (k=27)
                cv2.destroyAllWindows()
                break

    finally:
        pipeline.stop()
