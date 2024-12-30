#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import sys
import torch
import numpy as np
from typing import Tuple
import cv2
import supervision as sv
from pyvirtualdisplay import Display
from scipy.spatial.transform import Rotation as R

# utils and constants
from motion_capture.utils.utils import (
    rotation_matrix_to_quaternion,
    draw_axis,
    load_hamer,
    load_hmr2,
    recursive_to,
    cam_crop_to_full,
    MANO_JOINTS_CONNECTION,
    MANO_CONNECTION_NAMES,
    MANO_KEYPOINT_NAMES,
    SPIN_KEYPOINT_NAMES,
    FRANKMOCAP_PATH,
    FRANKMOCAP_CHECKPOINT,
    SMPL_DIR,
    HAMER_CHECKPOINT_PATH,
    HAMER_CONFIG_PATH,
    Renderer,
    draw_hand_keypoints,
)

# frankmocap hand
sys.path.insert(0, FRANKMOCAP_PATH)
import mocap_utils.demo_utils as demo_utils
from handmocap.hand_mocap_api import HandMocap as FrankMocapHand

# hamer
from hamer.datasets.vitdet_dataset import ViTDetDataset as HamerViTDetDataset

# 4DHuman
from hmr2.models import DEFAULT_CHECKPOINT
from hmr2.datasets.vitdet_dataset import ViTDetDataset as HMR2ViTDetDataset


BOX_ANNOTATOR = sv.BoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

from motion_capture.detector import DetectionResult

@dataclass
class MocapResult:
    detection: DetectionResult
    position: np.ndarray
    orientation: np.ndarray
    keypoint_names: List[str]
    keypoints: np.ndarray
    keypoints_2d: np.ndarray


class MocapModelFactory:
    @staticmethod
    def from_config(model: str, model_config: dict):
        if model == "frankmocap_hand":
            return FrankMocapHandModel(**model_config)
        elif model == "hamer":
            return HamerModel(**model_config)
        elif model == "4d-human":
            return HMR2Model(**model_config)
        else:
            raise ValueError(f"Invalid mocap model: {model_config['model']}")


class MocapModelBase(ABC):
    @abstractmethod
    def predict(self, detections, im, vis_im):
        pass


class FrankMocapHandModel(MocapModelBase):
    def __init__(
        self,
        img_size: Tuple[int, int] = (640, 480),
        render_type: str = "opengl",
        visualize: bool = True,
        device: str = "cuda:0",
    ):
        self.display = Display(visible=False, size=img_size)
        self.display.start()
        self.img_size = img_size
        self.visualize = visualize
        self.render_type = render_type
        self.device = device

        # init model
        self.mocap = FrankMocapHand(FRANKMOCAP_CHECKPOINT, SMPL_DIR, device=self.device)
        if self.visualize:
            if self.render_type in ["pytorch3d", "opendr"]:
                from renderer.screen_free_visualizer import Visualizer
            elif self.render_type == "opengl":
                from renderer.visualizer import Visualizer
            else:
                raise ValueError("Invalid render type")
            self.renderer = Visualizer(self.render_type)

    def predict(self, detections, im, vis_im):
        hand_bbox_list = []
        hand_bbox_dict = {"left_hand": None, "right_hand": None}
        mocap_results = []
        if detections:
            for detection in detections:
                hand_bbox_dict[detection.label] = detection.rect
            hand_bbox_list.append(hand_bbox_dict)
            # Hand Pose Regression
            pred_output_list = self.mocap.regress(im, hand_bbox_list, add_margin=True)
            pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

            if self.visualize:
                # visualize
                vis_im = self.renderer.visualize(
                    vis_im, pred_mesh_list=pred_mesh_list, hand_bbox_list=hand_bbox_list
                )

            for hand in pred_output_list[0]:  # TODO: handle multiple hands
                if pred_output_list[0][hand] is not None:
                    joint_coords = pred_output_list[0][hand]["pred_joints_img"]
                    hand_origin = np.sum(joint_coords[PALM_JOINTS] * WEIGHTS[:, None], axis=0)
                    hand_orientation = pred_output_list[0][hand]["pred_hand_pose"][0, :3].astype(
                        np.float32
                    )  # angle-axis representation

                    joint_3d_coords = pred_output_list[0][hand]["pred_joints_smpl"]  # (21, 3)


                    # for detection in detections:
                    #     if detection.label == hand:
                    #         detection.pose = hand_pose

                    rotation, _ = cv2.Rodrigues(hand_orientation)
                    quat = rotation_matrix_to_quaternion(rotation)  # [w, x, y, z]
                    if hand == "right_hand":
                        x_axis = np.array([0, 0, 1])
                        y_axis = np.array([0, -1, 0])
                        z_axis = np.array([-1, 0, 0])
                        rotated_result = R.from_rotvec(np.pi * np.array([1, 0, 0])) * R.from_quat(quat)
                        quat = rotated_result.as_quat()  # [w, x, y, z]
                    else:
                        x_axis = np.array([0, 0, 1])
                        y_axis = np.array([0, 1, 0])
                        z_axis = np.array([1, 0, 0])
                    x_axis_rotated = rotation @ x_axis
                    y_axis_rotated = rotation @ y_axis
                    z_axis_rotated = rotation @ z_axis

                    # visualize hand orientation
                    vis_im = draw_axis(vis_im, hand_origin, x_axis_rotated, (0, 0, 255))  # x: red
                    vis_im = draw_axis(vis_im, hand_origin, y_axis_rotated, (0, 255, 0))  # y: green
                    vis_im = draw_axis(vis_im, hand_origin, z_axis_rotated, (255, 0, 0))  # z: blue

        return mocap_results, vis_im

    def __del__(self):
        self.display.stop()


class HamerModel(MocapModelBase):
    def __init__(
        self,
        focal_length: float = 525.0,
        rescale_factor: float = 2.0,
        img_size: Tuple[int, int] = (640, 480),
        visualize: bool = True,
        device: str = "cuda:0",
    ):
        self.display = Display(visible=False, size=img_size)
        self.display.start()
        self.focal_length = focal_length
        self.rescale_factor = rescale_factor
        self.img_size = img_size
        self.visualize = visualize
        self.device = device

        # init model
        self.mocap, self.model_cfg = load_hamer(
            HAMER_CHECKPOINT_PATH,
            HAMER_CONFIG_PATH,
            img_size=self.img_size,
            focal_length=self.focal_length,
        )
        self.mocap.to(self.device)
        self.mocap.eval()
        if self.visualize:
            self.renderer = Renderer(
                faces=self.mocap.mano.faces,
                cfg=self.model_cfg,
                width=self.img_size[0],
                height=self.img_size[1],
            )

    def predict(self, detections, im, detection_im=None):
        # im : BGR image
        mocap_results = []
        if detections:
            boxes = np.array([detection.rect for detection in detections])  # x1, y1, x2, y2
            right = np.array([1 if detection.label == "right_hand" else 0 for detection in detections])

            # TODO clean it and fix this not to use datasetloader
            dataset = HamerViTDetDataset(self.model_cfg, im, boxes, right, rescale_factor=self.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
            batch = None
            for batch in dataloader:
                batch = recursive_to(batch, torch.device(self.device))  # to device
                with torch.no_grad():
                    out = self.mocap(batch)
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] *= 2 * batch["right"] - 1
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            )
            pred_cam_t_full = (
                cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
                .detach()
                .cpu()
                .numpy()
            )

            # 2D keypoints
            if self.visualize:
                box_center = batch["box_center"].detach().cpu().numpy()  # [N, 2]
                box_size = batch["box_size"].detach().cpu().numpy()  # [N,]
                pred_keypoints_2d = out["pred_keypoints_2d"].detach().cpu().numpy()  # [N, 21, 2]
                pred_keypoints_2d[:, :, 0] = (2 * right[:, None] - 1) * pred_keypoints_2d[
                    :, :, 0
                ]  # flip x-axis for left hand
                pred_keypoints_2d = pred_keypoints_2d * box_size[:, None, None] + box_center[:, None, :]

            # 3D keypoints
            pred_keypoints_3d = out["pred_keypoints_3d"].detach().cpu().numpy()  # [N, 21, 3]
            pred_keypoints_3d[:, :, 0] = (2 * right[:, None] - 1) * pred_keypoints_3d[:, :, 0]
            pred_keypoints_3d += pred_cam_t_full[:, None, :]

            # hand pose
            global_orient = (
                out["pred_mano_params"]["global_orient"].squeeze(1).detach().cpu().numpy()
            )  # [N, 3, 3]

            for i, hand_id in enumerate(right):  # for each hand
                assert (
                    detections[i].label == "right_hand" if hand_id == 1 else "left_hand"
                ), "Hand ID and hand detection mismatch"
                rotation = global_orient[i]
                if hand_id == 0:
                    rotation[1::3] *= -1
                    rotation[2::3] *= -1

                quat = rotation_matrix_to_quaternion(rotation)  # [w, x, y, z]
                assert len(MANO_KEYPOINT_NAMES) == len(pred_keypoints_3d[i]), "Keypoint mismatch"
                mocap_result = MocapResult(
                    detection=detections[i],
                    position=pred_keypoints_3d[i][0], # wrist position
                    orientation=quat,
                    keypoint_names=MANO_KEYPOINT_NAMES,
                    keypoints=pred_keypoints_3d[i],
                    keypoints_2d=pred_keypoints_2d[i],
                )
                mocap_results.append(mocap_result)

            if self.visualize:
                if detection_im is not None:
                    vis_im = detection_im.copy()
                else:
                    # Draw BBOX and LABEL with annotator
                    vis_im = im.copy()
                    vis_detections = sv.Detections(
                        xyxy=boxes,
                        class_id=right,
                    )
                    vis_im = BOX_ANNOTATOR.annotate(scene=vis_im, detections=vis_detections)
                    vis_im = LABEL_ANNOTATOR.annotate(
                        scene=vis_im,
                        detections=vis_detections,
                        labels=["right_hand" if class_id == 1 else "left_hand" for class_id in right],
                    )

                all_verts = []
                all_cam_t = []
                all_right = []

                # Render the result
                batch_size = batch["img"].shape[0]
                for n in range(batch_size):
                    # Add all verts and cams to list
                    verts = out["pred_vertices"][n].detach().cpu().numpy()
                    is_right = batch["right"][n].cpu().numpy()
                    verts[:, 0] = (2 * is_right - 1) * verts[:, 0]  # Flip x-axis
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)

                # Render front view
                if len(all_verts) > 0:
                    rgba, _ = self.renderer.render_rgba_multiple(
                        all_verts, cam_t=all_cam_t, is_right=all_right
                    )
                    rgb = rgba[..., :3].astype(np.float32)
                    alpha = rgba[..., 3].astype(np.float32) / 255.0
                    vis_im = vis_im[:, :, ::-1]
                    vis_im = (
                        alpha[..., None] * rgb
                        + (1 - alpha[..., None]) * vis_im
                    ).astype(np.uint8)
            else:
                vis_im = im.copy()

        else:  # no detections
            vis_im = im.copy()

        return mocap_results, vis_im

    def __del__(self):
        self.display.stop()


class HMR2Model(MocapModelBase):
    def __init__(
        self,
        focal_length: float = 525.0,
        rescale_factor: float = 2.0,
        img_size: Tuple[int, int] = (640, 480),
        visualize: bool = True,
        device: str = "cuda:0",
    ):
        self.display = Display(visible=False, size=img_size)
        self.display.start()
        self.focal_length = focal_length
        self.rescale_factor = rescale_factor
        self.img_size = img_size
        self.visualize = visualize
        self.device = device

        # init model
        self.mocap, self.model_cfg = load_hmr2(
            DEFAULT_CHECKPOINT,
            img_size=self.img_size,
            focal_length=self.focal_length,
        )
        self.mocap.to(self.device)
        self.mocap.eval()
        if self.visualize:
            self.renderer = Renderer(
                faces=self.mocap.smpl.faces,
                cfg=self.model_cfg,
                width=self.img_size[0],
                height=self.img_size[1],
            )

    def predict(self, detections, im, vis_im):
        # im : BGR image
        mocap_results = []
        if detections:
            boxes = np.array([detection.rect for detection in detections])  # x1, y1, x2, y2

            # TODO clean it and fix this not to use datasetloader
            dataset = HMR2ViTDetDataset(self.model_cfg, im, boxes)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
            batch = None
            for batch in dataloader:
                batch = recursive_to(batch, torch.device(self.device))  # to device
                with torch.no_grad():
                    out = self.mocap(batch)
            pred_cam = out["pred_cam"]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            )
            pred_cam_t_full = (
                cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
                .detach()
                .cpu()
                .numpy()
            )

            # this model uses 44 keypoints, but we use only 25 keypoints which corresponds to OpenPose keypoints
            # 2D keypoints
            box_center = batch["box_center"].detach().cpu().numpy()  # [N, 2]
            box_size = batch["box_size"].detach().cpu().numpy()  # [N,]
            pred_keypoints_2d = out["pred_keypoints_2d"].detach().cpu().numpy()  # [N, 44, 2]
            pred_keypoints_2d = pred_keypoints_2d * box_size[:, None, None] + box_center[:, None, :]
            pred_keypoints_2d = pred_keypoints_2d[:, :25, :]  # use only 25 keypoints

            # 3D keypoints
            pred_keypoints_3d = out["pred_keypoints_3d"].detach().cpu().numpy()  # [N, 44, 3]
            pred_keypoints_3d += pred_cam_t_full[:, None, :]
            pred_keypoints_3d = pred_keypoints_3d[:, :25, :]  # use only 25 keypoints

            # body pose
            body_origin = np.mean(pred_keypoints_2d, axis=1)  # [N, 2]
            body_origin = np.concatenate([body_origin, np.zeros((body_origin.shape[0], 1))], axis=1)  # [N, 3]
            global_orient = (
                out["pred_smpl_params"]["global_orient"].squeeze(1).detach().cpu().numpy()
            )  # [N, 3, 3]

            for i in range(len(detections)):  # for each body
                rotation = global_orient[i]

                quat = rotation_matrix_to_quaternion(rotation)  # [w, x, y, z]
                x_axis = np.array([0, 1, 0])
                y_axis = np.array([1, 0, 0])
                z_axis = np.array([0, 0, 1])
                rotated_result = R.from_rotvec(np.pi * np.array([1, 0, 0])) * R.from_quat(
                    quat
                )  # rotate 180 degree around x-axis
                quat = rotated_result.as_quat()  # [w, x, y, z]
                x_axis_rotated = rotation @ x_axis
                y_axis_rotated = rotation @ y_axis
                z_axis_rotated = rotation @ z_axis
                # visualize hand orientation
                vis_im = draw_axis(vis_im, body_origin[i], x_axis_rotated, (0, 0, 255))  # x: red
                vis_im = draw_axis(vis_im, body_origin[i], y_axis_rotated, (0, 255, 0))  # y: green
                vis_im = draw_axis(vis_im, body_origin[i], z_axis_rotated, (255, 0, 0))  # z: blue

                assert len(SPIN_KEYPOINT_NAMES) == len(pred_keypoints_3d[i]), "Keypoint mismatch"
                mocap_result = MocapResult(
                    detection=detections[i],
                    position=pred_keypoints_3d[i][0], #
                    orientation=quat,
                    keypoint_names=SPIN_KEYPOINT_NAMES,
                    keypoints=pred_keypoints_3d[i],
                )
                mocap_results.append(mocap_result)


            if self.visualize:
                all_verts = []
                all_cam_t = []

                # Render the result
                batch_size = batch["img"].shape[0]
                for n in range(batch_size):
                    # Add all verts and cams to list
                    verts = out["pred_vertices"][n].detach().cpu().numpy()
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)

                # Render front view
                if len(all_verts) > 0:
                    rgba, _ = self.renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t)
                    rgb = rgba[..., :3].astype(np.float32)
                    alpha = rgba[..., 3].astype(np.float32) / 255.0
                    vis_im = (
                        alpha[..., None] * rgb + (1 - alpha[..., None]) * cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    ).astype(np.uint8)

                # Draw 2D keypoints
                for i, keypoints in enumerate(pred_keypoints_2d):
                    for j, keypoint in enumerate(keypoints):
                        cv2.circle(vis_im, (int(keypoint[0]), int(keypoint[1])), 5, (0, 255, 0), -1)

        return mocap_results, vis_im

    def __del__(self):
        self.display.stop()
