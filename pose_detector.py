# encoding: utf-8
import cv2 as cv
import numpy as np
import os
import sys
import math
from PIL import Image, ImageDraw, ImageFont

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"MediaPipe 导入失败: {e}")

from standard_pose_library import (
    STANDARD_POSE_LIBRARY,
    RADAR_DIMENSIONS,
    detect_running_type,
    detect_gender,
    get_standard_pose,
    get_optimization_rules,
    generate_training_plan
)

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

class MockLandmark:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

class PoseDetector:
    def __init__(self, model_path='pose_landmarker.task', maxPoses=1):
        self.model_loaded = False
        self.landmarker = None
        self.mp_pose = None
        self.mp_drawing = None
        self.use_mock = False
        # 时序平滑：对 landmark 坐标做指数移动平均，减少帧间抖动
        self._smooth_alpha = 0.45        # 0~1, 越小越平滑但延迟越高
        self._prev_smoothed = None       # 上一帧平滑后的 33×2 坐标数组
        
        if not MEDIAPIPE_AVAILABLE:
            print("MediaPipe 库不可用，使用模拟姿态检测模式")
            self.use_mock = True
            self.model_loaded = True
            print("模拟模式已启用")
            return
        
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                num_poses=maxPoses,
                output_segmentation_masks=False)
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self.model_loaded = True
            print("PoseLandmarker 模型加载成功")
        except Exception as e:
            print(f"Task API 模型加载失败: {e}")
            print("尝试使用 MediaPipe Legacy API...")
            self._init_legacy_api()
    
    def _init_legacy_api(self):
        try:
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
            self.mp_drawing = mp.solutions.drawing_utils
            self.model_loaded = True
            print("Legacy API 初始化成功")
        except Exception as e:
            print(f"Legacy API 初始化失败: {e}")
            print("切换到模拟姿态检测模式")
            self.use_mock = True
            self.model_loaded = True
    
    def _generate_mock_landmarks(self, frame):
        import math
        h, w = frame.shape[:2]
        
        frame_hash = hash(frame.tobytes()[:100]) % 100
        
        if frame_hash < 34:
            base_torso_lean = 10
            base_knee_flexion = 55
            base_hip_extension = 162
            base_stride_ratio = 1.4
            vertical_amplitude = 8
        elif frame_hash < 67:
            base_torso_lean = 8
            base_knee_flexion = 45
            base_hip_extension = 168
            base_stride_ratio = 1.25
            vertical_amplitude = 5
        else:
            base_torso_lean = 28
            base_knee_flexion = 155
            base_hip_extension = 145
            base_stride_ratio = 2.5
            vertical_amplitude = 15
        
        self.frame_counter = getattr(self, 'frame_counter', 0)
        self.frame_counter += 1
        
        gait_cycle = 30
        cycle_phase = self.frame_counter % gait_cycle
        phase_ratio = cycle_phase / gait_cycle
        
        torso_len = h * 0.22
        thigh_len = h * 0.24
        calf_len = h * 0.24
        
        bounce_amplitude = h * (vertical_amplitude / 100)
        bounce_offset = int(bounce_amplitude * math.sin(phase_ratio * 2 * math.pi))
        
        torso_lean = base_torso_lean + math.sin(phase_ratio * 2 * math.pi) * 2
        
        if phase_ratio < 0.25:
            knee_flexion = base_knee_flexion + (165 - base_knee_flexion) * (phase_ratio / 0.25)
        elif phase_ratio < 0.5:
            knee_flexion = 165 - (165 - base_knee_flexion) * ((phase_ratio - 0.25) / 0.25)
        elif phase_ratio < 0.75:
            knee_flexion = base_knee_flexion
        else:
            knee_flexion = base_knee_flexion + (160 - base_knee_flexion) * ((phase_ratio - 0.75) / 0.25)
        
        hip_extension = base_hip_extension + math.sin(phase_ratio * 2 * math.pi) * 5
        
        stride_ratio = base_stride_ratio + math.sin(phase_ratio * 2 * math.pi) * 0.1
        
        lean_rad = math.radians(torso_lean)
        
        shoulder_center_x = w // 2
        shoulder_center_y = h // 3
        
        hip_center_x = int(shoulder_center_x + torso_len * math.sin(lean_rad))
        hip_center_y = int(shoulder_center_y + torso_len * math.cos(lean_rad) + bounce_offset)
        
        knee_flexion_rad = math.radians(knee_flexion)
        hip_extension_rad = math.radians(180 - hip_extension)
        
        forward_angle = hip_extension_rad
        left_knee_x = int(hip_center_x - thigh_len * math.sin(forward_angle - knee_flexion_rad))
        left_knee_y = int(hip_center_y + thigh_len * math.cos(forward_angle - knee_flexion_rad))
        
        right_knee_x = int(hip_center_x + thigh_len * math.sin(forward_angle + knee_flexion_rad))
        right_knee_y = int(hip_center_y + thigh_len * math.cos(forward_angle + knee_flexion_rad))
        
        ankle_flexion = 85 + math.sin(phase_ratio * 2 * math.pi) * 10
        ankle_flexion_rad = math.radians(ankle_flexion)
        
        left_ankle_x = int(left_knee_x - calf_len * math.sin(forward_angle - knee_flexion_rad - ankle_flexion_rad))
        left_ankle_y = int(left_knee_y + calf_len * math.cos(forward_angle - knee_flexion_rad - ankle_flexion_rad))
        
        right_ankle_x = int(right_knee_x + calf_len * math.sin(forward_angle + knee_flexion_rad + ankle_flexion_rad))
        right_ankle_y = int(right_knee_y + calf_len * math.cos(forward_angle + knee_flexion_rad + ankle_flexion_rad))
        
        hip_width = w // 8
        
        stride_offset = int(hip_width * stride_ratio)
        if base_torso_lean > 20:
            stride_offset = int(stride_offset * 1.5)
        
        if phase_ratio < 0.5:
            left_foot_x = left_ankle_x - stride_offset
            right_foot_x = right_ankle_x + stride_offset // 3
        else:
            left_foot_x = left_ankle_x - stride_offset // 3
            right_foot_x = right_ankle_x + stride_offset
        
        body_points = [
            (shoulder_center_x, shoulder_center_y - h//6),
            (shoulder_center_x, shoulder_center_y),
            (shoulder_center_x - w//6, shoulder_center_y),
            (shoulder_center_x + w//6, shoulder_center_y),
            (shoulder_center_x - w//6, shoulder_center_y + h//5),
            (shoulder_center_x + w//6, shoulder_center_y + h//5),
            (hip_center_x - hip_width, hip_center_y),
            (hip_center_x + hip_width, hip_center_y),
            (left_knee_x, left_knee_y),
            (right_knee_x, right_knee_y),
            (left_ankle_x, left_ankle_y),
            (right_ankle_x, right_ankle_y),
            (left_foot_x, left_ankle_y),
            (right_foot_x, right_ankle_y),
        ]
        
        landmarks = []
        for i in range(33):
            if i == 0:
                landmarks.append(MockLandmark(body_points[0][0]/w, body_points[0][1]/h))
            elif i == 1:
                landmarks.append(MockLandmark(body_points[1][0]/w, body_points[1][1]/h))
            elif i == 11:
                landmarks.append(MockLandmark(body_points[2][0]/w, body_points[2][1]/h))
            elif i == 12:
                landmarks.append(MockLandmark(body_points[3][0]/w, body_points[3][1]/h))
            elif i == 13:
                landmarks.append(MockLandmark(body_points[4][0]/w, body_points[4][1]/h))
            elif i == 14:
                landmarks.append(MockLandmark(body_points[5][0]/w, body_points[5][1]/h))
            elif i == 15:
                landmarks.append(MockLandmark(body_points[6][0]/w, body_points[6][1]/h))
            elif i == 16:
                landmarks.append(MockLandmark(body_points[7][0]/w, body_points[7][1]/h))
            elif i == 23:
                landmarks.append(MockLandmark(body_points[8][0]/w, body_points[8][1]/h))
            elif i == 24:
                landmarks.append(MockLandmark(body_points[9][0]/w, body_points[9][1]/h))
            elif i == 25:
                landmarks.append(MockLandmark(body_points[10][0]/w, body_points[10][1]/h))
            elif i == 26:
                landmarks.append(MockLandmark(body_points[11][0]/w, body_points[11][1]/h))
            elif i == 27:
                landmarks.append(MockLandmark(body_points[12][0]/w, body_points[12][1]/h))
            elif i == 28:
                landmarks.append(MockLandmark(body_points[13][0]/w, body_points[13][1]/h))
            else:
                landmarks.append(MockLandmark(0.5, 0.5))
        
        return [landmarks]
    
    def is_loaded(self):
        return self.model_loaded

    def _smooth_landmarks(self, landmarks):
        """对检测到的 landmarks 做指数移动平均平滑，减少抖动"""
        if not landmarks:
            self._prev_smoothed = None
            return landmarks

        # 提取当前帧坐标（归一化）
        curr = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)

        if self._prev_smoothed is not None and len(self._prev_smoothed) == len(curr):
            # EMA 融合
            smoothed = self._prev_smoothed * (1 - self._smooth_alpha) + curr * self._smooth_alpha
        else:
            smoothed = curr.copy()

        self._prev_smoothed = smoothed.copy()

        # 将平滑坐标写回 landmark 对象
        out = []
        for i, lm in enumerate(landmarks):
            class SmoothLandmark:
                pass
            sm = SmoothLandmark()
            sm.x = float(np.clip(smoothed[i, 0], 0.0, 1.0))
            sm.y = float(np.clip(smoothed[i, 1], 0.0, 1.0))
            sm.z = getattr(lm, 'z', 0.0)
            sm.visibility = getattr(lm, 'visibility', 1.0)
            out.append(sm)
        return out

    def _draw_angle_arc(self, img, p1, p2, p3, angle, color=(0, 150, 255), radius=35,
                         label=''):
        center = p2
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])

        start_angle = angle1 * 180 / np.pi
        end_angle = angle2 * 180 / np.pi

        if end_angle < start_angle:
            start_angle, end_angle = end_angle, start_angle

        # 绘制角度弧线
        cv.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, 2)

        # 角度数值标签 — 放在弧线中间位置
        mid_angle = (start_angle + end_angle) / 2 * np.pi / 180
        label_offset_x = int((radius + 25) * np.cos(mid_angle))
        label_offset_y = int((radius + 25) * np.sin(mid_angle))
        text_x = center[0] + label_offset_x - 20
        text_y = center[1] + label_offset_y + 5

        # 用PIL绘制角度值（支持所有字符）
        angle_text = f"{int(angle)}°"
        pil_img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            font_angle = ImageFont.truetype('C:/Windows/Fonts/simhei.ttf', 18)
            font_label = ImageFont.truetype('C:/Windows/Fonts/simhei.ttf', 14)
        except Exception:
            font_angle = ImageFont.load_default()
            font_label = ImageFont.load_default()

        # 测量角度文本尺寸
        angle_bbox = draw.textbbox((0, 0), angle_text, font=font_angle)
        tw = angle_bbox[2] - angle_bbox[0]
        th = angle_bbox[3] - angle_bbox[1]

        # 黑色背景框
        draw.rectangle([(text_x - 4, text_y - th - 4),
                       (text_x + tw + 4, text_y + 4)],
                      fill=(0, 0, 0))
        # 彩色角度值
        draw.text((text_x, text_y - th), angle_text, font=font_angle,
                  fill=(color[2], color[1], color[0]))

        # 关节名称标签 — 在关节位置上方显示
        if label:
            label_bbox = draw.textbbox((0, 0), label, font=font_label)
            lw = label_bbox[2] - label_bbox[0]
            lh = label_bbox[3] - label_bbox[1]
            lx = center[0] - lw // 2
            ly = center[1] - 18
            draw.rectangle([(lx - 3, ly - lh - 3),
                          (lx + lw + 3, ly + 3)],
                         fill=(0, 0, 0))
            draw.text((lx, ly - lh), label, font=font_label, fill=(255, 255, 255))

        # 将PIL图像转回OpenCV
        img[:, :] = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)[:, :]
    
    def _get_connection_color(self, start_idx, end_idx):
        face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        arm_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        leg_indices = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        torso_indices = [11, 12, 23, 24]

        if start_idx in torso_indices and end_idx in torso_indices:
            return (0, 215, 255)    # 黄色 BGR — 躯干
        elif start_idx in arm_indices and end_idx in arm_indices:
            return (255, 200, 50)   # 青色 BGR — 手臂（明亮蓝绿）
        elif start_idx in leg_indices and end_idx in leg_indices:
            return (200, 50, 255)   # 粉色 BGR — 腿部
        elif start_idx in face_indices or end_idx in face_indices:
            return (0, 165, 255)    # 橙色 BGR — 头部
        return (0, 255, 100)        # 亮绿 — 其他连接

    def _draw_pose_on_frame(self, frame, landmarks, angles=None):
        """
        在帧上绘制彩色骨架、关节标注点和角度标签。
        landmarks: MediaPipe 格式的 landmark 列表（包含 x,y 属性）
        angles: 可选，预先计算好的角度字典
        返回: 标注后的帧
        """
        img = frame.copy()
        h, w, _ = frame.shape
        points = []
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))

        # 1. 绘制骨架连线（厚线条）
        for connection in POSE_CONNECTIONS:
            s, e = connection
            if s < len(points) and e < len(points):
                color = self._get_connection_color(s, e)
                cv.line(img, points[s], points[e], color, 4)

        # 2. 绘制关节点（带黑色边框）
        for pt in points:
            cv.circle(img, pt, 8, (0, 0, 0), -1)      # 黑色外圈
            cv.circle(img, pt, 6, (255, 255, 255), -1)  # 白色内点

        # 3. 计算并绘制关键角度标记
        if angles is None:
            wrapped = [landmarks] if not hasattr(landmarks[0], 'x') else landmarks
            angles = self.calculate_joint_angles(wrapped)

        # 先绘制所有角度弧线（OpenCV）
        joint_configs = [
            ('left_elbow', 11, 13, 15, (255, 200, 50), ''),
            ('right_elbow', 12, 14, 16, (255, 200, 50), ''),
            ('left_knee', 23, 25, 27, (200, 50, 255), ''),
            ('right_knee', 24, 26, 28, (200, 50, 255), ''),
            ('left_hip', 11, 23, 25, (0, 215, 255), ''),
            ('right_hip', 12, 24, 26, (0, 215, 255), ''),
            ('left_shoulder', 13, 11, 23, (0, 215, 255), ''),
            ('right_shoulder', 14, 12, 24, (0, 215, 255), ''),
            ('left_ankle', 25, 27, 29, (200, 50, 255), ''),
            ('right_ankle', 26, 28, 30, (200, 50, 255), ''),
        ]

        # 收集所有PIL文本元素：(x, y, text, font_size, fill_rgb, is_angle_value, is_centered)
        text_elements = []

        for name, p1, p2, p3, color, joint_label in joint_configs:
            if p1 < len(points) and p2 < len(points) and p3 < len(points):
                angle = angles.get(name, 0)
                center = points[p2]
                v1 = (points[p1][0] - center[0], points[p1][1] - center[1])
                v2 = (points[p3][0] - center[0], points[p3][1] - center[1])
                a1 = np.arctan2(v1[1], v1[0]) * 180 / np.pi
                a2 = np.arctan2(v2[1], v2[0]) * 180 / np.pi
                if a2 < a1:
                    a1, a2 = a2, a1
                cv.ellipse(img, center, (35, 35), 0, a1, a2, color, 2)

                # 角度值
                mid_a = (a1 + a2) / 2 * np.pi / 180
                ox = int(60 * np.cos(mid_a))
                oy = int(60 * np.sin(mid_a))
                text_elements.append((center[0] + ox, center[1] + oy, f"{int(angle)}°",
                                      18, (color[2], color[1], color[0]), True, True))
                # 关节名
                if joint_label:
                    text_elements.append((center[0], center[1] - 18, joint_label,
                                          16, (255, 255, 255), False, True))

        # 4. 左上角图例
        for i, (b, g, r, label_text) in enumerate([
            (0, 215, 255, '躯干'),
            (255, 200, 50, '手臂'),
            (200, 50, 255, '腿部'),
        ]):
            y_pos = 25 + i * 30
            cv.rectangle(img, (10, y_pos - 10), (30, y_pos + 5), (b, g, r), -1)
            text_elements.append((35, y_pos, label_text, 18, (255, 255, 255), False, False))

        # 用PIL一次性绘制所有文本
        try:
            pil_img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            loaded = {}
            for tx, ty, txt, sz, clr, is_angle, is_ctrd in text_elements:
                if sz not in loaded:
                    loaded[sz] = ImageFont.truetype('C:/Windows/Fonts/simhei.ttf', sz)
                bb = draw.textbbox((0, 0), txt, font=loaded[sz])
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
                if is_angle:
                    # 角度值：左对齐，背景框+彩色文字
                    draw.rectangle([(tx - 4, ty - th - 4), (tx + tw + 4, ty + 4)], fill=(0, 0, 0))
                    draw.text((tx, ty - th), txt, font=loaded[sz], fill=clr)
                else:
                    # 标签/图例：背景框+白色文字，居中或左对齐
                    nx = tx - tw // 2 if is_ctrd else tx
                    draw.rectangle([(nx - 3, ty - th - 3), (nx + tw + 3, ty + 3)], fill=(0, 0, 0))
                    draw.text((nx, ty - th), txt, font=loaded[sz], fill=(255, 255, 255))
            return cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
        except Exception:
            return img
    
    def find_pose_landmarks(self, frame, draw=True):
        if not self.model_loaded:
            return frame.copy(), np.zeros_like(frame), None
        
        processed_frame = frame.copy()
        skeleton_img = np.zeros_like(frame)
        pose_landmarks = None
        angles = {}
        
        if self.use_mock:
            pose_landmarks = self._generate_mock_landmarks(frame)

            if draw and pose_landmarks:
                angles = self.calculate_joint_angles(pose_landmarks)
                processed_frame = self._draw_pose_on_frame(frame, pose_landmarks[0], angles)
                # 同步更新 skeleton_img
                h, w, _ = frame.shape
                landmarks = pose_landmarks[0]
                points = []
                for landmark in landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append((x, y))
                for connection in POSE_CONNECTIONS:
                    s, e = connection
                    if s < len(points) and e < len(points):
                        cv.line(skeleton_img, points[s], points[e], (0, 255, 0), 2)
                for pt in points:
                    cv.circle(skeleton_img, pt, 4, (0, 255, 0), -1)
        
        elif self.landmarker:
            try:
                img_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_RGB)
                detection_result = self.landmarker.detect(mp_image)

                if detection_result.pose_landmarks and draw:
                    for raw_landmarks in detection_result.pose_landmarks:
                        # 时序平滑，减少抖动
                        smoothed = self._smooth_landmarks(raw_landmarks)
                        angles = self.calculate_joint_angles([smoothed])
                        processed_frame = self._draw_pose_on_frame(frame, smoothed, angles)
                        # 同步 skeleton_img（同样用平滑后的坐标）
                        h, w, _ = frame.shape
                        points = []
                        for lm in smoothed:
                            x = int(lm.x * w)
                            y = int(lm.y * h)
                            points.append((x, y))
                        for connection in POSE_CONNECTIONS:
                            s, e = connection
                            if s < len(points) and e < len(points):
                                cv.line(skeleton_img, points[s], points[e], (0, 255, 0), 2)
                        for pt in points:
                            cv.circle(skeleton_img, pt, 4, (0, 255, 0), -1)

                pose_landmarks = detection_result.pose_landmarks
            except Exception as e:
                print(f"检测错误 (Task API): {e}")
        
        elif self.mp_pose:
            try:
                img_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                results = self.mp_pose.process(img_RGB)

                if results.pose_landmarks and draw:
                    raw_landmarks = results.pose_landmarks.landmark
                    # 时序平滑
                    smoothed = self._smooth_landmarks(raw_landmarks)
                    angles = self.calculate_joint_angles([results.pose_landmarks])
                    processed_frame = self._draw_pose_on_frame(frame, smoothed, angles)
                    # 同步 skeleton_img
                    h, w, _ = frame.shape
                    points = []
                    for lm in smoothed:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        points.append((x, y))
                    for connection in POSE_CONNECTIONS:
                        s, e = connection
                        if s < len(points) and e < len(points):
                            cv.line(skeleton_img, points[s], points[e], (0, 255, 0), 2)
                    for pt in points:
                        cv.circle(skeleton_img, pt, 4, (0, 255, 0), -1)

                pose_landmarks = [results.pose_landmarks] if results.pose_landmarks else None
            except Exception as e:
                print(f"检测错误 (Legacy API): {e}")
        
        return processed_frame, skeleton_img, pose_landmarks

    def detect_landmarks(self, frame):
        """仅检测并返回姿态关键点（不绘制）"""
        if not self.model_loaded:
            return None

        if self.use_mock:
            mock_result = self._generate_mock_landmarks(frame)
            return mock_result[0] if mock_result else None

        if self.landmarker:
            try:
                img_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_RGB)
                detection_result = self.landmarker.detect(mp_image)
                if detection_result.pose_landmarks:
                    return detection_result.pose_landmarks[0]
            except Exception as e:
                print(f"detect_landmarks error (Task API): {e}")
                return None

        if self.mp_pose:
            try:
                img_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                results = self.mp_pose.process(img_RGB)
                if results.pose_landmarks:
                    return results.pose_landmarks.landmark
            except Exception as e:
                print(f"detect_landmarks error (Legacy API): {e}")

        return None

    def calculate_joint_angles(self, landmarks):
        if not landmarks:
            return {}

        # 兼容两种输入格式：
        #   [list]   — wrapped (来自 find_pose_landmarks)
        #   list     — unwrapped (来自 detect_landmarks)
        if len(landmarks) > 0 and hasattr(landmarks[0], 'x'):
            # 已经是单层 landmark 列表 → 包装一层
            landmarks = [landmarks]

        angles = {}

        if landmarks and len(landmarks) > 0:
            pose_landmarks = landmarks[0]
        else:
            return {}
        
        def get_angle(p1, p2, p3):
            try:
                a = ((p1.x - p2.x), (p1.y - p2.y))
                b = ((p3.x - p2.x), (p3.y - p2.y))
                dot = a[0] * b[0] + a[1] * b[1]
                mag_a = (a[0]**2 + a[1]**2)**0.5
                mag_b = (b[0]**2 + b[1]**2)**0.5
                if mag_a == 0 or mag_b == 0:
                    return 0
                angle = abs(np.arccos(dot / (mag_a * mag_b)) * 180 / np.pi)
                return angle
            except:
                return 0
        
        try:
            angles['left_elbow'] = get_angle(
                pose_landmarks[11], pose_landmarks[13], pose_landmarks[15])
            angles['right_elbow'] = get_angle(
                pose_landmarks[12], pose_landmarks[14], pose_landmarks[16])
            angles['left_shoulder'] = get_angle(
                pose_landmarks[13], pose_landmarks[11], pose_landmarks[23])
            angles['right_shoulder'] = get_angle(
                pose_landmarks[14], pose_landmarks[12], pose_landmarks[24])
            angles['left_hip'] = get_angle(
                pose_landmarks[11], pose_landmarks[23], pose_landmarks[25])
            angles['right_hip'] = get_angle(
                pose_landmarks[12], pose_landmarks[24], pose_landmarks[26])
            angles['left_knee'] = get_angle(
                pose_landmarks[23], pose_landmarks[25], pose_landmarks[27])
            angles['right_knee'] = get_angle(
                pose_landmarks[24], pose_landmarks[26], pose_landmarks[28])
            angles['left_ankle'] = get_angle(
                pose_landmarks[25], pose_landmarks[27], pose_landmarks[29])
            angles['right_ankle'] = get_angle(
                pose_landmarks[26], pose_landmarks[28], pose_landmarks[30])
            angles['torso'] = get_angle(
                pose_landmarks[11], pose_landmarks[12], pose_landmarks[24])
        except Exception as e:
            print(f"计算角度错误: {e}")
        
        return angles
    
    def _calculate_additional_params(self, landmarks, angles):
        if not landmarks:
            return {}

        try:
            # 兼容 wrapped ([list]) 和 unwrapped (list) 两种输入
            if len(landmarks) > 0 and hasattr(landmarks[0], 'x'):
                pose = landmarks  # unwrapped: 直接就是 landmark 列表
            else:
                pose = landmarks[0] if len(landmarks) > 0 else None  # wrapped

            if pose is None:
                return {}

            h = 1.0
            
            nose = (pose[0].x, pose[0].y)
            left_shoulder = (pose[11].x, pose[11].y)
            right_shoulder = (pose[12].x, pose[12].y)
            left_hip = (pose[23].x, pose[23].y)
            right_hip = (pose[24].x, pose[24].y)
            left_knee = (pose[25].x, pose[25].y)
            right_knee = (pose[26].x, pose[26].y)
            left_ankle = (pose[27].x, pose[27].y)
            right_ankle = (pose[28].x, pose[28].y)
            left_heel = (pose[29].x, pose[29].y)
            right_heel = (pose[30].x, pose[30].y)
            
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                              (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_center = ((left_hip[0] + right_hip[0]) / 2, 
                         (left_hip[1] + right_hip[1]) / 2)
            knee_center = ((left_knee[0] + right_knee[0]) / 2, 
                          (left_knee[1] + right_knee[1]) / 2)
            ankle_center = ((left_ankle[0] + right_ankle[0]) / 2, 
                           (left_ankle[1] + right_ankle[1]) / 2)
            
            torso_vector = (hip_center[0] - shoulder_center[0], hip_center[1] - shoulder_center[1])
            torso_length = (torso_vector[0]**2 + torso_vector[1]**2)**0.5
            if torso_length > 0:
                torso_angle_vertical = abs(np.arccos(abs(torso_vector[1]) / torso_length) * 180 / np.pi - 90)
            else:
                torso_angle_vertical = 0
            
            body_vector = (ankle_center[0] - shoulder_center[0], ankle_center[1] - shoulder_center[1])
            body_length = (body_vector[0]**2 + body_vector[1]**2)**0.5
            if body_length > 0:
                body_lean_angle = abs(np.arccos(abs(body_vector[1]) / body_length) * 180 / np.pi - 90)
            else:
                body_lean_angle = 0
            
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            hip_width = abs(right_hip[0] - left_hip[0])
            knee_width = abs(right_knee[0] - left_knee[0])
            ankle_width = abs(right_ankle[0] - left_ankle[0])
            
            knee_valgus_left = abs(left_knee[0] - left_hip[0]) / max(hip_width, 0.01)
            knee_valgus_right = abs(right_knee[0] - right_hip[0]) / max(hip_width, 0.01)
            
            head_tilt = abs(nose[0] - shoulder_center[0]) / max(shoulder_width, 0.01) * 100
            
            pelvic_tilt = abs((left_hip[1] - right_hip[1]) / max(hip_width, 0.01)) * 100
            
            stride_length = abs(left_heel[1] - right_heel[1]) if hasattr(pose[29], 'y') else 0.1
            stride_ratio = stride_length / max(torso_length, 0.01)
            
            knee_height_diff = abs(left_knee[1] - right_knee[1])
            ankle_height_diff = abs(left_ankle[1] - right_ankle[1])
            
            avg_knee_angle = (angles.get('left_knee', 0) + angles.get('right_knee', 0)) / 2
            avg_hip_angle = (angles.get('left_hip', 0) + angles.get('right_hip', 0)) / 2
            
            overstriding_score = 1 if avg_knee_angle > 120 and stride_ratio > 2 else 0
            
            return {
                'torso_stability': torso_angle_vertical,
                'body_lean_angle': body_lean_angle,
                'knee_valgus_avg': (knee_valgus_left + knee_valgus_right) / 2,
                'head_tilt': head_tilt,
                'pelvic_tilt': pelvic_tilt,
                'arm_shoulder_ratio': (angles.get('left_elbow', 0) + angles.get('right_elbow', 0)) / 
                                     max((angles.get('left_shoulder', 0) + angles.get('right_shoulder', 0)), 1),
                'step_width_ratio': knee_width / max(hip_width, 0.01),
                'ankle_width_ratio': ankle_width / max(hip_width, 0.01),
                'stride_ratio': stride_ratio,
                'knee_height_diff': knee_height_diff,
                'ankle_height_diff': ankle_height_diff,
                'avg_knee_angle': avg_knee_angle,
                'avg_hip_angle': avg_hip_angle,
                'overstriding_score': overstriding_score
            }
        except Exception as e:
            print(f"计算附加参数错误: {e}")
            return {}
    
    def get_key_parameters(self, landmarks):
        if not landmarks:
            return None
        
        angles = self.calculate_joint_angles(landmarks)
        if not angles:
            return None
        
        additional_params = self._calculate_additional_params(landmarks, angles)
        
        elbow_avg = (angles.get('left_elbow', 0) + angles.get('right_elbow', 0)) / 2
        knee_avg = (angles.get('left_knee', 0) + angles.get('right_knee', 0)) / 2
        shoulder_avg = (angles.get('left_shoulder', 0) + angles.get('right_shoulder', 0)) / 2
        hip_avg = (angles.get('left_hip', 0) + angles.get('right_hip', 0)) / 2
        
        return {
            'elbow_flexion_avg': elbow_avg,
            'shoulder_angle_avg': shoulder_avg,
            'hip_angle_avg': hip_avg,
            'knee_flexion_avg': knee_avg,
            'ankle_angle_avg': (angles.get('left_ankle', 0) + angles.get('right_ankle', 0)) / 2,
            'torso_angle': angles.get('torso', 0),
            'arm_swing_symmetry': abs(angles.get('left_elbow', 0) - angles.get('right_elbow', 0)),
            'leg_symmetry': abs(angles.get('left_knee', 0) - angles.get('right_knee', 0)),
            'torso_stability': additional_params.get('torso_stability', 0),
            'body_lean_angle': additional_params.get('body_lean_angle', 0),
            'knee_valgus': additional_params.get('knee_valgus_avg', 0),
            'head_tilt': additional_params.get('head_tilt', 0),
            'pelvic_tilt': additional_params.get('pelvic_tilt', 0),
            'arm_shoulder_ratio': additional_params.get('arm_shoulder_ratio', 0),
            'step_width_ratio': additional_params.get('step_width_ratio', 0),
            'ankle_width_ratio': additional_params.get('ankle_width_ratio', 0),
            'stride_ratio': additional_params.get('stride_ratio', 0),
            'knee_height_diff': additional_params.get('knee_height_diff', 0),
            'ankle_height_diff': additional_params.get('ankle_height_diff', 0),
            'avg_knee_angle': additional_params.get('avg_knee_angle', 0),
            'avg_hip_angle': additional_params.get('avg_hip_angle', 0),
            'overstriding_score': additional_params.get('overstriding_score', 0),
            'all_angles': angles
        }
    
    def detect_gait_phase(self, landmarks):
        """检测当前帧的步态相位"""
        if not landmarks or len(landmarks) < 33:
            return 'unknown', 0
        
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        left_ankle_y = left_ankle.y
        right_ankle_y = right_ankle.y
        left_knee_y = left_knee.y
        right_knee_y = right_knee.y
        
        left_hip_y = left_hip.y
        right_hip_y = right_hip.y
        
        left_leg_length = abs(left_hip_y - left_ankle_y)
        right_leg_length = abs(right_hip_y - right_ankle_y)
        
        left_knee_angle = self._calculate_single_angle(left_hip, left_knee, landmarks[27])
        right_knee_angle = self._calculate_single_angle(right_hip, right_knee, landmarks[28])
        
        if left_ankle_y >= right_ankle_y - 0.02:
            stance_foot = 'left'
            swing_foot = 'right'
            stance_knee_angle = left_knee_angle
            swing_knee_angle = right_knee_angle
            stance_ankle_y = left_ankle_y
        else:
            stance_foot = 'right'
            swing_foot = 'left'
            stance_knee_angle = right_knee_angle
            swing_knee_angle = left_knee_angle
            stance_ankle_y = right_ankle_y
        
        if stance_knee_angle > 155:
            phase = 'contact'
            phase_value = 0.0
        elif stance_knee_angle > 140:
            phase = 'stance'
            phase_value = 0.25
        elif swing_knee_angle < 70:
            phase = 'push_off'
            phase_value = 0.5
        elif swing_knee_angle < 90:
            phase = 'swing'
            phase_value = 0.75
        else:
            phase = 'flight'
            phase_value = 1.0
        
        return phase, stance_foot
    
    def _calculate_single_angle(self, p1, p2, p3):
        """计算三个点形成的角度"""
        import math
        v1 = (p1.x - p2.x, p1.y - p2.y)
        v2 = (p3.x - p2.x, p3.y - p2.y)
        
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 180
        
        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def extract_gait_phases(self, frame_list):
        """从帧序列中提取步态相位序列"""
        phases = []
        phase_frames = {
            'contact_left': [],
            'contact_right': [],
            'stance_left': [],
            'stance_right': [],
            'swing_left': [],
            'swing_right': [],
            'flight': []
        }
        
        for i, frame in enumerate(frame_list):
            landmarks = self.detect_landmarks(frame)
            if landmarks:
                phase, stance_foot = self.detect_gait_phase(landmarks)
                phases.append((i, phase, stance_foot))
                
                phase_key = f"{phase}_{stance_foot}" if stance_foot and phase in ['contact', 'stance', 'swing'] else phase
                if phase_key in phase_frames:
                    phase_frames[phase_key].append(i)
                elif phase in phase_frames:
                    phase_frames[phase].append(i)
        
        return phases, phase_frames
    
    def align_gait_phases(self, std_phases, test_phases):
        """对齐两个视频的步态相位，返回匹配的帧索引对"""
        aligned_pairs = []
        
        std_phase_dict = {}
        for i, (frame_idx, phase, stance_foot) in enumerate(std_phases):
            key = f"{phase}_{stance_foot}" if stance_foot else phase
            if key not in std_phase_dict:
                std_phase_dict[key] = []
            std_phase_dict[key].append(frame_idx)
        
        test_phase_dict = {}
        for i, (frame_idx, phase, stance_foot) in enumerate(test_phases):
            key = f"{phase}_{stance_foot}" if stance_foot else phase
            if key not in test_phase_dict:
                test_phase_dict[key] = []
            test_phase_dict[key].append(frame_idx)
        
        for phase_key in std_phase_dict:
            if phase_key in test_phase_dict:
                std_frames = std_phase_dict[phase_key]
                test_frames = test_phase_dict[phase_key]
                
                min_len = min(len(std_frames), len(test_frames))
                for j in range(min_len):
                    aligned_pairs.append((std_frames[j], test_frames[j], phase_key))
        
        return aligned_pairs
    
    def extract_motion_curve(self, frame_list):
        """从帧序列中提取关键角度的运动曲线"""
        knee_angles = []
        hip_angles = []
        torso_angles = []
        elbow_angles = []
        
        for frame in frame_list:
            landmarks = self.detect_landmarks(frame)
            if landmarks:
                params = self.get_key_parameters(landmarks)
                if params:
                    knee_angles.append(params.get('knee_flexion_avg', 90))
                    hip_angles.append(params.get('hip_angle_avg', 160))
                    torso_angles.append(params.get('body_lean_angle', 10))
                    elbow_angles.append(params.get('elbow_flexion_avg', 90))
        
        return {
            'knee_flexion': knee_angles,
            'hip_extension': hip_angles,
            'torso_lean': torso_angles,
            'elbow_flexion': elbow_angles
        }
    
    def calculate_cosine_similarity(self, curve1, curve2):
        """计算两条曲线的余弦相似度"""
        if len(curve1) == 0 or len(curve2) == 0:
            return 0.0
        
        min_len = min(len(curve1), len(curve2))
        if min_len == 0:
            return 0.0
        
        curve1_norm = curve1[:min_len]
        curve2_norm = curve2[:min_len]
        
        dot_product = sum(c1 * c2 for c1, c2 in zip(curve1_norm, curve2_norm))
        norm1 = sum(c1 ** 2 for c1 in curve1_norm) ** 0.5
        norm2 = sum(c2 ** 2 for c2 in curve2_norm) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def calculate_pearson_correlation(self, curve1, curve2):
        """计算两条曲线的皮尔逊相关系数"""
        if len(curve1) < 2 or len(curve2) < 2:
            return 0.0
        
        min_len = min(len(curve1), len(curve2))
        curve1_norm = curve1[:min_len]
        curve2_norm = curve2[:min_len]
        
        mean1 = sum(curve1_norm) / min_len
        mean2 = sum(curve2_norm) / min_len
        
        numerator = sum((c1 - mean1) * (c2 - mean2) for c1, c2 in zip(curve1_norm, curve2_norm))
        denominator1 = sum((c1 - mean1) ** 2 for c1 in curve1_norm) ** 0.5
        denominator2 = sum((c2 - mean2) ** 2 for c2 in curve2_norm) ** 0.5
        
        if denominator1 == 0 or denominator2 == 0:
            return 0.0
        
        return numerator / (denominator1 * denominator2)
    
    def compare_curve_similarity(self, standard_curves, test_curves):
        """综合计算两条运动曲线的相似度"""
        weights = {
            'knee_flexion': 0.35,
            'hip_extension': 0.25,
            'torso_lean': 0.25,
            'elbow_flexion': 0.15
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            std_curve = standard_curves.get(key, [])
            test_curve = test_curves.get(key, [])
            
            cos_sim = self.calculate_cosine_similarity(std_curve, test_curve)
            pearson = self.calculate_pearson_correlation(std_curve, test_curve)
            
            combined_sim = (cos_sim + pearson) / 2
            total_score += combined_sim * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight
    
    def check_fatal_errors(self, test_curves):
        """检查测试曲线中的致命错误姿势"""
        penalty = 0
        
        knee_angles = test_curves.get('knee_flexion', [])
        torso_angles = test_curves.get('torso_lean', [])
        stride_ratios = test_curves.get('stride_ratio', [1.5] * len(knee_angles))
        
        for i, knee_angle in enumerate(knee_angles):
            torso = torso_angles[i] if i < len(torso_angles) else 10
            stride = stride_ratios[i] if i < len(stride_ratios) else 1.5
            
            if knee_angle > 170 and stride > 2.2:
                penalty += 2
            
            if torso > 25:
                penalty += 1
            
            if knee_angle > 160 and stride > 2.0:
                penalty += 1
        
        max_penalty = len(knee_angles)
        return min(penalty / max_penalty * 50, 50) if max_penalty > 0 else 0
    
    def frame_combine(self, frame, src):
        if frame.shape[0] != src.shape[0]:
            height = min(frame.shape[0], src.shape[0])
            frame = cv.resize(frame, (int(frame.shape[1] * height / frame.shape[0]), height))
            src = cv.resize(src, (int(src.shape[1] * height / src.shape[0]), height))
        dst = np.hstack((frame, src))
        return dst
    
    def _gaussian_score(self, value, ideal_center, sigma):
        """高斯型评分函数"""
        import math
        if sigma == 0:
            return 100.0 if value == ideal_center else 0.0
        return 100.0 * math.exp(-((value - ideal_center) ** 2) / (2 * sigma ** 2))
    
    def _calculate_deviation_score(self, test_value, std_value, tolerance_ratio=0.15):
        """计算测试值与标准值的偏差评分"""
        if std_value == 0:
            return 100.0
        
        tolerance = abs(std_value) * tolerance_ratio
        deviation = abs(test_value - std_value)
        
        if deviation <= tolerance * 0.3:
            return 100.0
        elif deviation <= tolerance:
            return 100.0 - (deviation / tolerance) * 30
        elif deviation <= tolerance * 2:
            return 70.0 - (deviation - tolerance) / tolerance * 40
        else:
            return max(10.0, 30.0 - (deviation - tolerance * 2) / tolerance * 20)
    
    def _dynamic_score(self, value, ideal_min, ideal_max, good_min, good_max, unit_penalty=0.8, severe_penalty=1.5):
        """
        国际标准分段阶梯评分函数（向后兼容）：
        - 精英区间(ideal_min~ideal_max)：得分 96-100
        - 良好区间(good_min~good_max 或 ideal两侧扩展)：得分 70-96
        - 需改进区间：得分从 70 快速扣减到 20
        """
        if ideal_min <= value <= ideal_max:
            center = (ideal_min + ideal_max) / 2
            half_range = (ideal_max - ideal_min) / 2
            if half_range == 0:
                return 98.0
            deviation = abs(value - center) / half_range
            return 100 - deviation * 4
        
        if value < ideal_min:
            if value >= good_min:
                excess = ideal_min - value
                range_good = ideal_min - good_min
                ratio = excess / range_good if range_good > 0 else 1.0
                return 96 - ratio * 26
            else:
                excess = good_min - value
                return max(20, 70 - excess * severe_penalty)
        else:
            if value <= good_max:
                excess = value - ideal_max
                range_good = good_max - ideal_max
                ratio = excess / range_good if range_good > 0 else 1.0
                return 96 - ratio * 26
            else:
                excess = value - good_max
                return max(20, 70 - excess * severe_penalty)


    def compare_poses(self, standard_params, test_params):
        """
        标准视频vs测试视频对比评分（真实验证）
        - curve模式：使用运动曲线相似度 + 数值偏差综合评分
        - single模式：使用逐帧参数偏差评分（对比标准参数）
        """
        if not standard_params or not test_params:
            return 80.0

        if isinstance(standard_params, dict) and 'knee_flexion' in standard_params:
            return self._compare_curve_based(standard_params, test_params)
        else:
            return self._compare_single_frame(standard_params, test_params)

    def _compare_single_frame(self, standard_params, test_params):
        """
        逐帧参数对比 —— 测试参数与标准参数的偏差评分
        分值含义：
          ≥90：与标准跑姿高度一致
          75-89：较为接近标准
          60-74：有可见偏差
          <60：偏差显著，需针对性改进
        """
        # 对比项目：(参数名, 权重, 基础容差)
        compare_items = [
            ('knee_flexion_avg', 0.18, 12),
            ('hip_angle_avg', 0.12, 10),
            ('ankle_angle_avg', 0.10, 8),
            ('body_lean_angle', 0.12, 4),
            ('torso_stability', 0.10, 5),
            ('elbow_flexion_avg', 0.08, 10),
            ('stride_ratio', 0.08, 0.20),
            ('pelvic_tilt', 0.06, 3),
            ('leg_symmetry', 0.06, 8),
            ('arm_swing_symmetry', 0.05, 8),
            ('head_tilt', 0.05, 4),
        ]

        total_score = 0.0
        total_weight = 0.0

        for param_key, weight, tolerance in compare_items:
            std_val = standard_params.get(param_key, None)
            test_val = test_params.get(param_key, None)

            if std_val is None or test_val is None:
                continue

            if std_val == 0:
                continue

            # 偏差绝对值 / 有效容差
            effective_tolerance = max(abs(std_val) * 0.15, tolerance)
            deviation = abs(test_val - std_val)
            ratio = deviation / effective_tolerance if effective_tolerance > 0 else 99

            if ratio <= 0.3:
                score = 100.0
            elif ratio <= 0.7:
                score = 100.0 - (ratio - 0.3) / 0.4 * 20      # 100 → 80
            elif ratio <= 1.5:
                score = 80.0 - (ratio - 0.7) / 0.8 * 25        # 80 → 55
            elif ratio <= 2.5:
                score = 55.0 - (ratio - 1.5) / 1.0 * 20        # 55 → 35
            else:
                score = max(15.0, 35.0 - (ratio - 2.5) * 5)    # 35 → 15

            total_score += score * weight
            total_weight += weight

        if total_weight == 0:
            return 80.0

        base_score = total_score / total_weight

        # === 致命错误扣分 ===
        penalty = 0

        # 躯干过度前倾
        test_lean = test_params.get('body_lean_angle', 0)
        if test_lean > 25:
            penalty += 20
        elif test_lean > 20:
            penalty += 10

        # 膝关节过伸 + 大步幅 = 刹车效应
        test_knee = test_params.get('knee_flexion_avg', 90)
        test_stride = test_params.get('stride_ratio', 1.5)
        if test_knee > 165 and test_stride > 2.0:
            penalty += 18
        elif test_knee > 155 and test_stride > 2.2:
            penalty += 12

        # 膝关节过度折叠（跑步效率低）
        if test_knee < 35:
            penalty += 15

        # 躯干稳定性差
        test_torso_stab = test_params.get('torso_stability', 0)
        if test_torso_stab > 20:
            penalty += 12

        final_score = base_score - penalty
        final_score = max(25.0, min(99.5, final_score))

        return round(final_score, 1)

    def _compare_curve_based(self, standard_curves, test_curves):
        """
        基于运动曲线相似度的综合评分
        结合 曲线形态相似度(40%) + 关节数值偏差(60%)
        """
        # 1) 曲线形态相似度
        shape_similarity = self.compare_curve_similarity(standard_curves, test_curves)

        # 2) 逐关节数值偏差评分
        weights = {
            'knee_flexion': 0.35,
            'hip_extension': 0.25,
            'torso_lean': 0.25,
            'elbow_flexion': 0.15
        }

        # 各关节可接受最大RMSE
        max_acceptable_rmse = {
            'knee_flexion': 25,
            'hip_extension': 20,
            'torso_lean': 8,
            'elbow_flexion': 20
        }

        total_dev_score = 0.0
        total_dev_weight = 0.0

        for key, weight in weights.items():
            std_curve = standard_curves.get(key, [])
            test_curve = test_curves.get(key, [])

            min_len = min(len(std_curve), len(test_curve))
            if min_len < 2:
                continue

            # 均方根误差 (RMSE)
            squared_errors = [(std_curve[i] - test_curve[i]) ** 2 for i in range(min_len)]
            rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5

            max_err = max_acceptable_rmse.get(key, 20)

            # RMSE → 分数
            if rmse <= max_err * 0.3:
                dev_score = 95.0
            elif rmse <= max_err * 0.7:
                dev_score = 95.0 - (rmse - max_err * 0.3) / (max_err * 0.4) * 25
            elif rmse <= max_err * 1.5:
                dev_score = 70.0 - (rmse - max_err * 0.7) / (max_err * 0.8) * 35
            else:
                dev_score = max(20.0, 35.0 - (rmse - max_err * 1.5) / max_err * 15)

            total_dev_score += dev_score * weight
            total_dev_weight += weight

        avg_dev_score = total_dev_score / total_dev_weight if total_dev_weight > 0 else 70.0

        # 3) 综合：形状相似度(40%) + 数值偏差(60%)
        shape_component = max(0, shape_similarity) * 100 * 0.40
        dev_component = avg_dev_score * 0.60
        base_score = shape_component + dev_component

        # 4) 致命错误扣分
        fatal_penalty = self.check_fatal_errors(test_curves)
        final_score = base_score - fatal_penalty

        final_score = max(20.0, min(99.5, final_score))

        return round(final_score, 1)

    # ==================== 标准跑姿库相关方法 ====================
    
    def analyze_video_for_radar(self, frame_list, running_type=None, gender=None):
        """分析视频并生成雷达图数据"""
        params_list = []

        for frame in frame_list:
            landmarks = self.detect_landmarks(frame)
            if landmarks:
                params = self.get_key_parameters(landmarks)
                if params:
                    params_list.append(params)

        if not params_list:
            return None

        return self._calculate_radar_scores(params_list, running_type, gender)

    def analyze_video_with_skeleton(self, frame_list, running_type=None, gender=None,
                                     max_radar_frames=60):
        """分析视频，返回雷达数据及每帧骨架坐标（用于前端canvas绘制）

        Args:
            frame_list: 所有待处理帧列表
            running_type: 跑步类型
            gender: 性别
            max_radar_frames: 用于雷达评分的最多帧数（骨架数据会用全部帧）
        """
        params_list = []
        skeleton_data = []  # 每帧的归一化关节点坐标 [[x,y], ...]

        for frame in frame_list:
            landmarks = self.detect_landmarks(frame)
            if landmarks:
                # 始终收集骨架数据（用于前端canvas绘制）
                lm_data = [[lm.x, lm.y] for lm in landmarks]
                skeleton_data.append(lm_data)

                # 仅前 max_radar_frames 帧用于雷达评分
                if len(params_list) < max_radar_frames:
                    params = self.get_key_parameters(landmarks)
                    if params:
                        params_list.append(params)

        if not params_list:
            return None, None

        radar_result = self._calculate_radar_scores(params_list, running_type, gender)
        return radar_result, skeleton_data
    
    def _calculate_radar_scores(self, params_list, running_type=None, gender=None):
        """计算五个维度的雷达图得分 —— 基于标准跑姿库的偏差评分"""
        # 提取所有帧的关键参数
        torso_lean_values = [p.get('body_lean_angle', 0) for p in params_list]
        knee_flex_values = [p.get('knee_flexion_avg', 90) for p in params_list]
        hip_angle_values = [p.get('hip_angle_avg', 160) for p in params_list]
        elbow_flex_values = [p.get('elbow_flexion_avg', 90) for p in params_list]
        knee_sym_values = [p.get('leg_symmetry', 0) for p in params_list]
        pelvic_tilt_values = [p.get('pelvic_tilt', 0) for p in params_list]
        head_tilt_values = [p.get('head_tilt', 0) for p in params_list]
        stride_values = [p.get('stride_ratio', 1.5) for p in params_list]
        torso_stab_values = [p.get('torso_stability', 0) for p in params_list]

        # 计算统计量
        avg_torso_lean = np.mean(torso_lean_values)
        torso_lean_variance = np.var(torso_lean_values)
        avg_knee_flex = np.mean(knee_flex_values)
        knee_flex_min = min(knee_flex_values)
        knee_flex_max = max(knee_flex_values)
        avg_hip_angle = np.mean(hip_angle_values)
        avg_elbow_flex = np.mean(elbow_flex_values)
        knee_sym_avg = np.mean(knee_sym_values)
        pelvic_tilt_avg = np.mean(pelvic_tilt_values)
        head_tilt_avg = np.mean(head_tilt_values)
        avg_stride = np.mean(stride_values)
        avg_torso_stab = np.mean(torso_stab_values)

        # 检测跑步类型和性别
        if running_type is None or gender is None:
            sample_params = {
                'knee_min_angle': knee_flex_min,
                'stride_ratio': avg_stride,
                'shoulder_hip_ratio': params_list[0].get('shoulder_hip_ratio', 1.1)
            }
            auto_running_type = detect_running_type(sample_params)
            auto_gender = detect_gender(sample_params)
            final_running_type = running_type if running_type else auto_running_type
            final_gender = gender if gender else auto_gender
        else:
            final_running_type = running_type
            final_gender = gender

        # 获取标准数据
        standard_data = get_standard_pose(final_running_type, final_gender, 'side_view')
        std_params = standard_data.get('gait_cycle_params', {})
        std_thresholds = standard_data.get('ideal_ranges', {})

        # 提取标准关键值（各相位典型值）
        std_knee_swing = std_params.get('knee_flexion', {}).get('swing', 55)
        std_knee_push = std_params.get('knee_flexion', {}).get('push_off', 168)
        std_hip_push = std_params.get('hip_extension', {}).get('push_off', 170)
        std_torso = std_params.get('torso_lean', {}).get('contact', 6)
        std_stride = std_params.get('stride_ratio', 1.5)

        # 计算周期平均值（用于与全周期平均值参数比较，而非单相位值）
        std_knee_cycle = np.mean([
            std_params.get('knee_flexion', {}).get(p, 150)
            for p in ['contact', 'stance', 'push_off', 'swing']
        ])
        std_hip_cycle = np.mean([
            std_params.get('hip_extension', {}).get(p, 160)
            for p in ['contact', 'stance', 'push_off', 'swing']
        ])
        std_torso_cycle = np.mean([
            std_params.get('torso_lean', {}).get(p, 6)
            for p in ['contact', 'stance', 'push_off', 'swing']
        ])
        std_elbow_cycle = np.mean([
            std_params.get('elbow_flexion', {}).get(p, 85)
            for p in ['contact', 'stance', 'push_off', 'swing']
        ])

        # 维度1: 核心稳定度
        core_score = self._calculate_core_stability_score(
            torso_lean_variance, pelvic_tilt_avg, head_tilt_avg,
            avg_torso_lean, std_torso, std_params
        )

        # 维度2: 下肢折叠效率
        leg_score = self._calculate_leg_fold_score(
            knee_flex_min, knee_flex_max, avg_hip_angle,
            std_knee_swing, std_knee_push, std_hip_push, std_thresholds
        )

        # 维度3: 着地品质（使用周期平均值作为基准，而非单相位值）
        landing_score = self._calculate_landing_score(
            avg_knee_flex, avg_torso_lean, avg_stride,
            std_knee_cycle, std_torso_cycle, std_stride
        )

        # 维度4: 推进力（使用周期平均值作为基准）
        propulsion_score = self._calculate_propulsion_score(
            avg_hip_angle, avg_knee_flex, avg_stride,
            std_hip_cycle, std_knee_cycle, std_stride
        )

        # 维度5: 对称性
        symmetry_score = self._calculate_symmetry_score(
            knee_sym_avg, elbow_flex_values, knee_flex_values
        )

        # ===== 评分基线提升（上调约50分） =====
        # 低分段提升幅度大，高分段提升幅度小，保持区分度
        def _boost(s):
            if s >= 90:
                return min(100, s + 5)
            elif s >= 60:
                return 80 + (s - 60) * 0.5    # 60→80, 90→95
            elif s >= 30:
                return 55 + (s - 30) * 0.83   # 30→55, 60→80
            else:
                return 30 + s * 0.83          # 0→30, 30→55

        core_score = _boost(core_score)
        leg_score = _boost(leg_score)
        landing_score = _boost(landing_score)
        propulsion_score = _boost(propulsion_score)
        symmetry_score = _boost(symmetry_score)

        # 综合评分
        weights = {
            'core_stability': 0.20,
            'leg_fold_efficiency': 0.25,
            'landing_quality': 0.20,
            'propulsion': 0.18,
            'symmetry': 0.17
        }

        total_score = (
            core_score * weights['core_stability'] +
            leg_score * weights['leg_fold_efficiency'] +
            landing_score * weights['landing_quality'] +
            propulsion_score * weights['propulsion'] +
            symmetry_score * weights['symmetry']
        )

        # 检测问题
        specific_errors = self._detect_specific_errors(params_list, std_params)
        issues = self._detect_issues(
            core_score, leg_score, landing_score, propulsion_score, symmetry_score,
            specific_errors=specific_errors
        )

        # 生成优化建议
        optimization_plan = self._generate_optimization_plan(issues)

        return {
            'radar_scores': {
                'core_stability': round(core_score, 1),
                'leg_fold_efficiency': round(leg_score, 1),
                'landing_quality': round(landing_score, 1),
                'propulsion': round(propulsion_score, 1),
                'symmetry': round(symmetry_score, 1)
            },
            'total_score': round(total_score, 1),
            'running_type': final_running_type,
            'gender': final_gender,
            'params_summary': {
                'avg_torso_lean': round(avg_torso_lean, 1),
                'knee_flex_min': round(knee_flex_min, 1),
                'knee_flex_max': round(knee_flex_max, 1),
                'avg_hip_angle': round(avg_hip_angle, 1),
                'stride_ratio': round(avg_stride, 2)
            },
            'issues': issues,
            'specific_errors': specific_errors,
            'optimization_plan': optimization_plan,
            'phase_scores': self._calculate_phase_scores(params_list, std_params),
            'standard_params': std_params
        }
    
    def _calculate_core_stability_score(self, torso_variance, pelvic_tilt, head_tilt,
                                         avg_torso_lean, std_torso, std_params):
        """
        核心稳定度 —— 基于标准跑姿库评分（宽松阈值版）
        指标：躯干晃动方差、骨盆倾斜、头部晃动、躯干前倾偏差
        """
        # 躯干晃动方差（越小越稳定）
        if torso_variance <= 4:
            torso_score = 88
        elif torso_variance <= 10:
            torso_score = 78 - (torso_variance - 4) * 3          # 78 → 60
        elif torso_variance <= 18:
            torso_score = 60 - (torso_variance - 10) * 2.5       # 60 → 40
        else:
            torso_score = max(25, 40 - (torso_variance - 18) * 2)

        # 骨盆倾斜（越小越好）
        if pelvic_tilt <= 4:
            pelvic_score = 88
        elif pelvic_tilt <= 10:
            pelvic_score = 78 - (pelvic_tilt - 4) * 3
        elif pelvic_tilt <= 18:
            pelvic_score = 60 - (pelvic_tilt - 10) * 2.5
        else:
            pelvic_score = max(25, 40 - (pelvic_tilt - 18) * 2)

        # 头部晃动
        if head_tilt <= 5:
            head_score = 88
        elif head_tilt <= 12:
            head_score = 78 - (head_tilt - 5) * 3
        elif head_tilt <= 20:
            head_score = 60 - (head_tilt - 12) * 2.5
        else:
            head_score = max(25, 40 - (head_tilt - 20) * 2)

        # 躯干前倾角偏离标准的程度
        lean_dev = abs(avg_torso_lean - std_torso)
        if lean_dev <= 4:
            lean_score = 88
        elif lean_dev <= 10:
            lean_score = 76 - (lean_dev - 4) * 3                 # 76 → 58
        elif lean_dev <= 18:
            lean_score = 58 - (lean_dev - 10) * 2.5              # 58 → 38
        else:
            lean_score = max(20, 38 - (lean_dev - 18) * 2)

        return (torso_score * 0.30 + pelvic_score * 0.20 +
                head_score * 0.15 + lean_score * 0.35)

    def _calculate_leg_fold_score(self, knee_min, knee_max, avg_hip_angle,
                                    std_swing, std_push, std_hip_push, thresholds):
        """
        下肢折叠效率 —— 基于标准跑姿库评分（宽松阈值版）
        - 摆动期折叠：knee_min 应接近 std_swing（越小越好，但不可过度）
        - 蹬伸期伸展：knee_max 应接近 std_push（越大越好）
        - 髋关节伸展：avg_hip_angle 应接近 std_hip_push
        """
        std_knee_min = thresholds.get('knee_flexion_min', std_swing - 10)
        std_knee_max = thresholds.get('knee_flexion_max', std_push + 5)

        # 折叠充分度：knee_min 足够小说明折叠充分
        target_fold = std_swing
        if knee_min <= target_fold + 20:
            fold_score = 88
        elif knee_min <= target_fold + 45:
            fold_score = 76 - (knee_min - target_fold - 20) * 0.53  # 76 → 63
        elif knee_min <= target_fold + 70:
            fold_score = 63 - (knee_min - target_fold - 45) * 0.6   # 63 → 48
        else:
            fold_score = max(20, 48 - (knee_min - target_fold - 70) * 0.6)

        # 伸展充分度：knee_max 足够大说明蹬伸充分
        if knee_max >= std_push - 12:
            ext_score = 88
        elif knee_max >= std_push - 30:
            ext_score = 76 - (std_push - 12 - knee_max) * 0.67
        elif knee_max >= std_push - 50:
            ext_score = 60 - (std_push - 30 - knee_max) * 0.6
        else:
            ext_score = max(20, 48 - (std_push - 50 - knee_max) * 0.5)

        # 髋关节伸展
        if avg_hip_angle >= std_hip_push - 12:
            hip_score = 88
        elif avg_hip_angle >= std_hip_push - 30:
            hip_score = 76 - (std_hip_push - 12 - avg_hip_angle) * 0.67
        elif avg_hip_angle >= std_hip_push - 50:
            hip_score = 60 - (std_hip_push - 30 - avg_hip_angle) * 0.6
        else:
            hip_score = max(20, 48 - (std_hip_push - 50 - avg_hip_angle) * 0.5)

        return (fold_score * 0.40 + ext_score * 0.30 + hip_score * 0.30)

    def _calculate_landing_score(self, knee_angle, torso_lean, stride_ratio,
                                  std_knee, std_torso, std_stride):
        """
        着地品质 —— 基于标准跑姿库评分（宽松阈值版）
        - 膝角应接近该跑姿的周期平均值
        - 步幅比应接近标准值（过大=跨步刹车，过小=步频过低）
        - 躯干前倾应接近标准值
        """
        # 着地膝角与标准周期平均值的偏差
        knee_dev = abs(knee_angle - std_knee)
        if knee_dev <= 10:
            knee_score = 88
        elif knee_dev <= 22:
            knee_score = 78 - (knee_dev - 10) * 1.5               # 78 → 60
        elif knee_dev <= 36:
            knee_score = 60 - (knee_dev - 22) * 1.5               # 60 → 39
        else:
            knee_score = max(20, 39 - (knee_dev - 36) * 1)

        # 步幅比与标准值的偏差
        stride_dev = abs(stride_ratio - std_stride)
        if stride_dev <= 0.20:
            stride_score = 88
        elif stride_dev <= 0.45:
            stride_score = 78 - (stride_dev - 0.20) * 44          # 78 → 67
        elif stride_dev <= 0.75:
            stride_score = 67 - (stride_dev - 0.45) * 57          # 67 → 50
        else:
            stride_score = max(20, 50 - (stride_dev - 0.75) * 50)

        # 躯干前倾与标准值的偏差
        lean_dev = abs(torso_lean - std_torso)
        if lean_dev <= 4:
            lean_score = 88
        elif lean_dev <= 10:
            lean_score = 76 - (lean_dev - 4) * 3                  # 76 → 58
        elif lean_dev <= 18:
            lean_score = 58 - (lean_dev - 10) * 2.5               # 58 → 38
        else:
            lean_score = max(20, 38 - (lean_dev - 18) * 2)

        return (knee_score * 0.40 + stride_score * 0.35 + lean_score * 0.25)

    def _calculate_propulsion_score(self, avg_hip_angle, avg_knee_flex, avg_stride,
                                     std_hip_avg, std_knee_avg, std_stride):
        """
        推进力 —— 基于标准跑姿库评分（宽松阈值版）
        - 髋关节伸展幅度应接近标准周期平均值
        - 膝关节角度应接近标准周期平均值
        - 步幅比应匹配跑步类型
        """
        # 髋角评分（蹬伸幅度）
        hip_dev = abs(avg_hip_angle - std_hip_avg)
        if hip_dev <= 12:
            hip_score = 86
        elif hip_dev <= 25:
            hip_score = 76 - (hip_dev - 12) * 1.2                # 76 → 60.4
        elif hip_dev <= 40:
            hip_score = 60 - (hip_dev - 25) * 1.0                # 60 → 45
        else:
            hip_score = max(20, 45 - (hip_dev - 40) * 0.8)

        # 膝角评分（蹬伸效率）
        knee_dev = abs(avg_knee_flex - std_knee_avg)
        if knee_dev <= 14:
            knee_score = 86
        elif knee_dev <= 28:
            knee_score = 76 - (knee_dev - 14) * 1.0              # 76 → 62
        elif knee_dev <= 44:
            knee_score = 62 - (knee_dev - 28) * 0.9              # 62 → 47.6
        else:
            knee_score = max(20, 47 - (knee_dev - 44) * 0.8)

        # 步幅比与标准值的偏差
        stride_dev = abs(avg_stride - std_stride)
        if stride_dev <= 0.20:
            coord_score = 86
        elif stride_dev <= 0.45:
            coord_score = 76 - (stride_dev - 0.20) * 40          # 76 → 66
        elif stride_dev <= 0.75:
            coord_score = 66 - (stride_dev - 0.45) * 53          # 66 → 50
        else:
            coord_score = max(20, 50 - (stride_dev - 0.75) * 50)

        return (hip_score * 0.40 + knee_score * 0.35 + coord_score * 0.25)

    def _calculate_symmetry_score(self, knee_sym, elbow_values, knee_values):
        """
        对称性 —— 双侧动作一致性评分（宽松阈值版）
        """
        # 膝关节左右对称性
        if knee_sym <= 10:
            knee_sym_score = 88
        elif knee_sym <= 22:
            knee_sym_score = 78 - (knee_sym - 10) * 1.3           # 78 → 62.4
        elif knee_sym <= 36:
            knee_sym_score = 62 - (knee_sym - 22) * 1.1           # 62 → 46.6
        else:
            knee_sym_score = max(20, 46 - (knee_sym - 36) * 0.8)

        # 肘关节对称性（利用时间序列差异估算）
        if len(elbow_values) >= 2:
            left_elb = elbow_values[0]
            right_elb = elbow_values[-1] if len(elbow_values) > 1 else left_elb
            elbow_diff = abs(left_elb - right_elb)
        else:
            elbow_diff = 5

        if elbow_diff <= 14:
            elbow_sym_score = 88
        elif elbow_diff <= 28:
            elbow_sym_score = 78 - (elbow_diff - 14) * 1.1        # 78 → 62.6
        elif elbow_diff <= 42:
            elbow_sym_score = 62 - (elbow_diff - 28) * 1.0        # 62 → 48
        else:
            elbow_sym_score = max(20, 48 - (elbow_diff - 42) * 0.8)

        # 触地时间对称性（左右膝角曲线的相关性）
        if len(knee_values) >= 10:
            mid = len(knee_values) // 2
            left_knee = knee_values[:mid]
            right_knee = knee_values[mid:]
            min_len = min(len(left_knee), len(right_knee))
            if min_len >= 3:
                correlation = np.corrcoef(left_knee[:min_len], right_knee[:min_len])[0, 1]
                if np.isnan(correlation):
                    correlation = 0.7
            else:
                correlation = 0.7
        else:
            correlation = 0.7

        if correlation >= 0.78:
            time_sym_score = 88
        elif correlation >= 0.55:
            time_sym_score = 68 + (correlation - 0.55) * 130      # 68 → 85
        elif correlation >= 0.30:
            time_sym_score = 45 + (correlation - 0.30) * 120      # 45 → 69
        else:
            time_sym_score = max(20, 40 - (0.30 - correlation) * 80)

        return (knee_sym_score * 0.40 + elbow_sym_score * 0.25 + time_sym_score * 0.35)
    
    def _detect_specific_errors(self, params_list, std_params=None):
        """
        检测具体错误跑姿模式 —— 基于用户提供的跑姿评估框架
        使用分位数和帧比例过滤，避免偶发动作（低头、转头等）导致误报。
        返回: [{'type': str, 'label': str, 'severity': str, 'detail': str, 'frames_pct': float}, ...]
        """
        MIN_FRAMES = 8
        if not params_list or len(params_list) < MIN_FRAMES:
            return []

        n = len(params_list)
        specific_errors = []

        # ---- 鲁棒性：帧比例阈值 ----
        # 只有当超过此比例的帧出现异常时才判定为错误
        # 偶发的低头/转头/侧倾等正常动作会被过滤掉
        ERROR_PCT_THRESHOLD = 18   # 至少 18% 的帧异常才触发
        SEVERE_PCT_THRESHOLD = 45  # 超过 45% 的帧异常判为严重

        # ---- 鲁棒性：分位数替代极值 ----
        # 使用 10% 分位数替代 min()，忽略短暂的一次性异常
        # 使用中位数（50%分位数）替代均值，对异常值更不敏感

        # 1. 躯干前倾角 —— 分位数过滤偶发低头/转身
        body_lean_vals = [p.get('body_lean_angle', 5) for p in params_list]
        p10_lean = np.percentile(body_lean_vals, 10)   # 排除底部10%的瞬低值
        p50_lean = np.percentile(body_lean_vals, 50)   # 中位数（鲁棒中心）

        # 帧比例：真正低于阈值或高于阈值的帧占比
        lean_too_low_pct = sum(1 for v in body_lean_vals if v < 3) / n * 100
        lean_too_high_pct = sum(1 for v in body_lean_vals if v > 12) / n * 100

        # 2. 头部姿态 —— 分位数过滤偶发转头/低头
        head_tilt_vals = [p.get('head_tilt', 0) for p in params_list]
        p50_head_tilt = np.percentile(head_tilt_vals, 50)
        head_tilt_high_pct = sum(1 for v in head_tilt_vals if v > 12) / n * 100

        # 3. 膝关节内外翻 —— 分位数过滤偶发步态偏差
        knee_valgus_vals = [p.get('knee_valgus', 0) for p in params_list]
        p50_knee_valgus = np.percentile(knee_valgus_vals, 50)
        knee_valgus_high_pct = sum(1 for v in knee_valgus_vals if v > 15) / n * 100

        # 4. 步幅比 —— 分位数过滤偶发的跨步
        stride_vals = [p.get('stride_ratio', 1.5) for p in params_list]
        p50_stride = np.percentile(stride_vals, 50)
        overstride_count = sum(1 for p in params_list if p.get('overstriding_score', 0) == 1)
        overstride_pct = overstride_count / n * 100

        # 5. 肘部角度 —— 分位数过滤偶尔的摆臂变化
        elbow_vals = [p.get('elbow_flexion_avg', 90) for p in params_list]
        p50_elbow = np.percentile(elbow_vals, 50)

        # 6. 躯干稳定性 —— 使用分位数范围评估（而非方差）
        torso_stab_vals = [p.get('torso_stability', 0) for p in params_list]
        # 用 P90-P10 范围代替方差，对单次晃动不敏感
        stab_range = np.percentile(torso_stab_vals, 90) - np.percentile(torso_stab_vals, 10)

        # 获取标准值
        std_stride = 1.5
        if std_params:
            std_stride = std_params.get('stride_ratio', 1.5)

        # ============================================================
        # 错误检测（所有检测均需要超过帧比例阈值才触发）
        # ============================================================

        # 1. 步幅过大检测
        stride_dev = p50_stride - std_stride  # 使用中位数而非均值
        if stride_dev > 0.4 or overstride_pct > ERROR_PCT_THRESHOLD:
            severity = 'severe' if (stride_dev > 0.7 or overstride_pct > SEVERE_PCT_THRESHOLD) else 'moderate'
            pct = max(overstride_pct, stride_dev / 1.0 * 30)  # 估算异常帧占比
            specific_errors.append({
                'type': 'overstriding',
                'label': '步幅过大',
                'severity': severity,
                'detail': (
                    f'步幅中位数 {p50_stride:.2f}，超出标准值 {std_stride:.1f} 达 {stride_dev:.1f}。'
                    '刻意跨大步会导致脚落地时明显伸在身体前方，形成"制动"动作，'
                    '大幅增加膝关节和踝关节冲击力，长期易受伤且跑步效率极低。'
                    '建议缩短步幅，让脚掌落在身体重心正下方。'
                ),
                'frames_pct': round(min(pct, 100), 1)
            })
        elif stride_dev > 0.25 and overstride_pct > 8:
            specific_errors.append({
                'type': 'overstriding',
                'label': '步幅偏大',
                'severity': 'mild',
                'detail': (
                    f'步幅中位数 {p50_stride:.2f}，略高于标准值 {std_stride:.1f}。'
                    '注意保持适中步幅，避免刻意跨大步。'
                ),
                'frames_pct': round(overstride_pct, 1)
            })

        # 2. 躯干姿态检测（基于帧比例，偶发一次低头不触发）
        if lean_too_low_pct > ERROR_PCT_THRESHOLD:
            # 躯干在相当比例的帧中过于直立或后仰
            if lean_too_low_pct > SEVERE_PCT_THRESHOLD or p10_lean < 1:
                specific_errors.append({
                    'type': 'torso_leaning_back',
                    'label': '身体后仰',
                    'severity': 'severe' if p10_lean < 1 else 'moderate',
                    'detail': (
                        f'躯干前倾角中位数 {p50_lean:.1f}°，{lean_too_low_pct:.0f}% 的帧前倾角低于3°。'
                        '存在明显身体后仰或过度直立现象，会导致重心不稳、呼吸不畅。'
                        '标准跑姿应保持上半身直立微前倾约5-10°，耳朵、肩膀、髋骨在一条直线上，核心收紧。'
                    ),
                    'frames_pct': round(lean_too_low_pct, 1)
                })
            else:
                specific_errors.append({
                    'type': 'torso_upright',
                    'label': '躯干过于直立',
                    'severity': 'mild',
                    'detail': (
                        f'躯干前倾角中位数 {p50_lean:.1f}°，{lean_too_low_pct:.0f}% 的帧偏直立。'
                        '建议略微增加前倾至5-10°，利用重力推进提高跑步效率。'
                    ),
                    'frames_pct': round(lean_too_low_pct, 1)
                })
        elif lean_too_high_pct > ERROR_PCT_THRESHOLD:
            specific_errors.append({
                'type': 'torso_overlean',
                'label': '躯干过度前倾',
                'severity': 'severe' if lean_too_high_pct > SEVERE_PCT_THRESHOLD else 'moderate',
                'detail': (
                    f'躯干前倾角中位数 {p50_lean:.1f}°，{lean_too_high_pct:.0f}% 的帧前倾角超过12°。'
                    '过度前倾会增加腰部负担。建议适当抬高上身，保持核心收紧，耳朵-肩膀-髋骨三点一线。'
                ),
                'frames_pct': round(lean_too_high_pct, 1)
            })

        # 3. 摆臂检测（基于中位数偏差 × 帧比例）
        elbow_dev = abs(p50_elbow - 90)
        elbow_bad_pct = sum(1 for v in elbow_vals if abs(v - 90) > 25) / n * 100
        if elbow_dev > 25 and elbow_bad_pct > ERROR_PCT_THRESHOLD:
            severity = 'severe' if (elbow_dev > 35 or elbow_bad_pct > SEVERE_PCT_THRESHOLD) else 'moderate'
            specific_errors.append({
                'type': 'arm_swing_angle',
                'label': '摆臂角度异常',
                'severity': severity,
                'detail': (
                    f'肘部角度中位数 {p50_elbow:.0f}°，{elbow_bad_pct:.0f}% 的帧偏离标准90°超过25°。'
                    '摆臂应以肩关节为轴前后自然摆动，肘部保持约90°夹角，'
                    '摆幅控制在腰侧至胸前之间，前不露肘、后不露手。'
                    '避免左右横摆、交叉过中线。'
                ),
                'frames_pct': round(elbow_bad_pct, 1)
            })
        elif elbow_dev > 15:
            elbow_mild_pct = sum(1 for v in elbow_vals if abs(v - 90) > 15) / n * 100
            if elbow_mild_pct > ERROR_PCT_THRESHOLD:
                specific_errors.append({
                    'type': 'arm_swing_angle',
                    'label': '摆臂角度偏大',
                    'severity': 'mild',
                    'detail': (
                        f'肘部角度中位数 {p50_elbow:.0f}°，{elbow_mild_pct:.0f}% 的帧偏离标准。'
                        '建议调整至约90°，合适的肘部角度可提高摆臂效率。'
                    ),
                    'frames_pct': round(elbow_mild_pct, 1)
                })

        # 4. 膝盖内扣/外撇检测（基于帧比例）
        if knee_valgus_high_pct > ERROR_PCT_THRESHOLD:
            severity = 'severe' if (p50_knee_valgus > 25 or knee_valgus_high_pct > SEVERE_PCT_THRESHOLD) else 'moderate'
            specific_errors.append({
                'type': 'knee_valgus',
                'label': '膝盖内扣/外撇',
                'severity': severity,
                'detail': (
                    f'膝关节内外翻角中位数 {p50_knee_valgus:.1f}%，'
                    f'{knee_valgus_high_pct:.0f}% 的帧存在明显内扣或外撇。'
                    '跑步时膝关节应保持正向稳定，内扣/外撇会导致关节受力不均，'
                    '增加膝盖和韧带损伤风险。建议加强臀中肌和髋部外展肌群力量。'
                ),
                'frames_pct': round(knee_valgus_high_pct, 1)
            })

        # 5. 头部姿态检测（基于帧比例，偶发低头看路不触发）
        if head_tilt_high_pct > ERROR_PCT_THRESHOLD:
            severity = 'severe' if (p50_head_tilt > 20 or head_tilt_high_pct > SEVERE_PCT_THRESHOLD) else 'moderate'
            specific_errors.append({
                'type': 'head_position',
                'label': '头部姿态异常',
                'severity': severity,
                'detail': (
                    f'头部倾斜角中位数 {p50_head_tilt:.1f}%，'
                    f'{head_tilt_high_pct:.0f}% 的帧头部姿态偏离正常范围。'
                    '存在频繁低头、仰头或头部侧倾。'
                    '跑步时应平视前方，头部稳定，颈部自然放松，不低头不仰头。'
                ),
                'frames_pct': round(head_tilt_high_pct, 1)
            })

        # 6. 重心稳定性检测（分位数范围代替方差）
        if stab_range > 8:  # P90-P10 范围超过 8°
            severity = 'severe' if stab_range > 14 else 'moderate'
            specific_errors.append({
                'type': 'instability',
                'label': '重心稳定性不足',
                'severity': severity,
                'detail': (
                    f'躯干稳定性范围 {stab_range:.1f}°（P10-P90），存在明显左右摇晃或上下起伏。'
                    '标准跑姿应保持动作连贯协调，重心稳定前移，无大幅摇晃或上下起伏。'
                ),
                'frames_pct': round(stab_range / 14 * 100, 1)
            })

        # 按严重程度排序
        sev_order = {'severe': 0, 'moderate': 1, 'mild': 2}
        specific_errors.sort(key=lambda x: sev_order.get(x['severity'], 3))

        return specific_errors

    def _detect_issues(self, core_score, leg_score, landing_score, propulsion_score, symmetry_score,
                        specific_errors=None):
        """检测各维度问题"""
        issues = []

        def add_issue(dimension, score):
            if score < 35:
                issues.append({'dimension': dimension, 'level': 'severe', 'score': round(score, 1)})
            elif score < 55:
                issues.append({'dimension': dimension, 'level': 'moderate', 'score': round(score, 1)})
            elif score < 75:
                issues.append({'dimension': dimension, 'level': 'mild', 'score': round(score, 1)})

        add_issue('core_stability', core_score)
        add_issue('leg_fold_efficiency', leg_score)
        add_issue('landing_quality', landing_score)
        add_issue('propulsion', propulsion_score)
        add_issue('symmetry', symmetry_score)

        # 合并具体错误模式到 issues
        if specific_errors:
            for err in specific_errors:
                severity = err.get('severity', 'moderate')
                issues.append({
                    'dimension': None,
                    'level': severity,
                    'score': None,
                    'specific_error': err['label'],
                    'detail': err['detail']
                })

        # 按严重程度排序
        level_order = {'severe': 0, 'moderate': 1, 'mild': 2}
        issues.sort(key=lambda x: level_order[x['level']])

        return issues
    
    def _generate_optimization_plan(self, issues):
        """生成优化建议"""
        plan = {
            'priority': [],      # 优先改进
            'attention': [],     # 需要关注
            'enhance': []        # 锦上添花
        }

        for issue in issues:
            dim_key = issue.get('dimension')
            level = issue['level']
            specific_error = issue.get('specific_error')

            if dim_key is None:
                # 具体错误模式 — 使用 detail 作为建议
                detail = issue.get('detail', '')
                item = {
                    'dimension': specific_error or '具体问题',
                    'score': None,
                    'advice': detail,
                    'exercises': [],
                    'specific_error': True
                }
                if level == 'severe' or level == 'moderate':
                    plan['priority'].append(item)
                else:
                    plan['attention'].append(item)
                continue

            rules = get_optimization_rules(dim_key)

            if level in rules:
                advice = rules[level].get('advice', '')
                exercises = rules[level].get('exercises', [])

                item = {
                    'dimension': RADAR_DIMENSIONS[dim_key]['name'],
                    'score': issue['score'],
                    'advice': advice,
                    'exercises': exercises
                }

                if level == 'severe':
                    plan['priority'].append(item)
                elif level == 'moderate':
                    plan['attention'].append(item)
                else:
                    plan['enhance'].append(item)

        # 生成阶段性训练计划
        plan['training_plan'] = generate_training_plan(issues)

        return plan
    
    def _calculate_phase_scores(self, params_list, std_params):
        """计算各步态相位得分"""
        phase_scores = {
            'contact': [],
            'stance': [],
            'push_off': [],
            'swing': []
        }
        
        for params in params_list:
            knee_angle = params.get('knee_flexion_avg', 90)
            
            # 简化的相位判断
            if knee_angle > 155:
                phase = 'contact'
            elif knee_angle > 140:
                phase = 'stance'
            elif knee_angle < 80:
                phase = 'push_off'
            else:
                phase = 'swing'
            
            # 计算该帧的得分
            score = self._calculate_single_frame_score(params, std_params)
            phase_scores[phase].append(score)
        
        # 计算各相位平均得分
        result = {}
        for phase, scores in phase_scores.items():
            if scores:
                result[phase] = round(sum(scores) / len(scores), 1)
            else:
                result[phase] = 50
        
        return result
    
    def _calculate_single_frame_score(self, params, std_params):
        """计算单帧得分"""
        knee_angle = params.get('knee_flexion_avg', 90)
        hip_angle = params.get('hip_angle_avg', 160)
        torso_lean = params.get('body_lean_angle', 10)
        
        score = 0
        weight_sum = 0
        
        # 膝角评分
        std_knee = std_params.get('knee_flexion', {}).get('contact', 160)
        if std_knee:
            score += self._gaussian_score(knee_angle, std_knee, 15) * 0.4
            weight_sum += 0.4
        
        # 髋角评分
        std_hip = std_params.get('hip_extension', {}).get('contact', 155)
        if std_hip:
            score += self._gaussian_score(hip_angle, std_hip, 10) * 0.3
            weight_sum += 0.3
        
        # 躯干前倾评分
        std_torso = std_params.get('torso_lean', {}).get('contact', 8)
        if std_torso:
            score += self._gaussian_score(torso_lean, std_torso, 5) * 0.3
            weight_sum += 0.3
        
        return score / weight_sum if weight_sum > 0 else 50


    def process_video_with_overlay(self, video_path, output_path, max_frames=None):
        """逐帧处理视频，绘制骨架叠加并输出新视频文件"""
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        if max_frames and max_frames < total_frames:
            total_frames = max_frames

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = self.detect_landmarks(frame)
            if landmarks:
                angles = self.calculate_joint_angles([landmarks])
                processed = self._draw_pose_on_frame(frame, landmarks, angles)
            else:
                processed = frame.copy()

            out.write(processed)
            frame_count += 1

            if frame_count % 30 == 0:
                print(f"处理视频帧: {frame_count}/{total_frames}")

        cap.release()
        out.release()
        print(f"视频处理完成: {output_path}")
        return output_path


if __name__ == '__main__':
    print("人体姿态检测模块测试")
    detector = PoseDetector()
    print(f"模型加载状态: {detector.is_loaded()}")
