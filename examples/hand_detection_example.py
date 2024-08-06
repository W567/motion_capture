#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import cv2
from motion_capture.detector import DetectionModelFactory
from motion_capture.mocap import MocapModelFactory

def main(args):
    image = cv2.imread(args.input)
    img_size = (image.shape[1], image.shape[0])

    detector = args.detector
    if detector == "hand_object_detector":
        detector_config = {
            "threshold": 0.9,
            "object_threshold": 0.9,
            "margin": 10,
            "device": args.device,
        }
    elif detector == "mediapipe_hand":
        detector_config = {
            "threshold": 0.9,
            "margin": 10,
            "device": args.device,
        }
    elif detector == "yolo":
        detector_config = {
            "margin": 10,
            "threshold": 0.9,
            "device": args.device,
        }
    else:
        raise ValueError(f"Invalid detector model: {detector}")

    mocap = args.mocap
    if mocap == "frankmocap_hand":
        mocap_config = {
            "render_type": "opengl",
            "img_size": img_size,
            "visualize": True,
            "device": args.device,
        }
    elif mocap == "hamer":
        mocap_config = {
            "focal_length": 525.0,
            "rescale_factor": 2.0,
            "img_size": img_size,
            "visualize": True,
            "device": args.device,
        }
    elif mocap == "4d-human":
        mocap_config = {
            "focal_length": 525.0,
            "rescale_factor": 2.0,
            "img_size": img_size,
            "visualize": True,
            "device": args.device,
        }
    else:
        raise ValueError(f"Invalid mocap model: {mocap}")

    detection_model = DetectionModelFactory.from_config(
        model=detector,
        model_config=detector_config,
    )
    mocap_model = MocapModelFactory.from_config(
        model=mocap,
        model_config=mocap_config,
    )

    detections, visualization = detection_model.predict(image)
    cv2.imwrite(args.output, visualization)

    mocap_result, vis_im = mocap_model.predict(detections, image, visualization)
    cv2.imwrite(args.output, vis_im)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand detection example")
    parser.add_argument("--input", type=str, default="input.jpg", help="Input image")
    parser.add_argument("--output", type=str, default="output.jpg", help="Output image")
    parser.add_argument("--detector", type=str, default="hand_object_detector", help="Detector model")
    parser.add_argument("--mocap", type=str, default="hamer", help="Mocap model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()

    main(args)
