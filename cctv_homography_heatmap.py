#!/usr/bin/env python3
# cctv_homography_heatmap.py - 호모그래피 변환을 이용한 CCTV 히트맵 시스템

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import json
import math
import torch
from datetime import datetime, timedelta

# === 맵 설정 (YAML 파일 기준) ===
MAP_IMAGE_PATH = "CCTV_MAP.png"
# MAP_RESOLUTION = 0.05  # meters per pixel (YAML 값과 동일)
# MAP_ORIGIN = (5.225, 3.925)  # X, Y in meters (from CCTV_MAP.yaml)

# === 설정 복원 (coordinate_picker.py 기준) ===
MAP_RESOLUTION = 0.05
MAP_ORIGIN = (-10.0, -10.0)

# === 카메라 이미지 좌표와 맵 좌표 매핑 ===
# 각 카메라에 대해 이미지 상의 픽셀 좌표와 대응하는 맵 좌표를 정의합니다
# 순서는 반드시 일치해야 합니다!

# 카메라 이미지 상의 픽셀 좌표
IMAGE_COORDS = {
    "cctv1": np.array([[261, 252], [373, 236], [637, 352], [267, 479]], dtype=np.float32),
    "cctv2": np.array([[43, 254], [36, 478], [630, 477], [277, 225]], dtype=np.float32),
    "cctv3": np.array([[277, 236], [159, 478], [471, 478], [417, 240]], dtype=np.float32)
}

# 실제 맵 상의 좌표 (미터 단위) - 사용자가 제공한 값 (coordinate_picker 로 측정)
MAP_COORDS = {
    "cctv1": np.array([[-5.200, 5.750], [-8.100, 5.900], [-8.150, -1.850], [-3.300, -1.900]], dtype=np.float32),
    "cctv2": np.array([[4.200, 8.200], [4.100, 1.950], [0.250, 0.150], [0.250, 8.200]], dtype=np.float32),
    "cctv3": np.array([[-8.000, 5.750], [-0.200, 5.750], [-0.100, 3.600], [-8.050, 3.550]], dtype=np.float32)
}

# 카메라 위치 정보 (coordinate_picker 로 업데이트된 값)
CAMERA_POSITIONS = {
    "cctv1": (-2.350, -4.450, 3.0),
    "cctv2": (3.900, -1.600, 3.0),
    "cctv3": (3.050, 4.550, 3.0)
}

class CCTVHeatmapNode(Node):
    def __init__(self):
        super().__init__('cctv_heatmap_node')
        
        # 로거 설정
        self.get_logger().info("Initializing CCTV Homography Heatmap System...")
        
        # 브릿지 초기화
        self.bridge = CvBridge()
        
        # 맵 이미지 로드
        self.map_img = cv2.imread(MAP_IMAGE_PATH)
        if self.map_img is None:
            self.get_logger().error(f"Could not load map image: {MAP_IMAGE_PATH}")
            raise FileNotFoundError(f"Map image not found: {MAP_IMAGE_PATH}")
        
        # 맵 크기 얻기 (픽셀 단위)
        self.map_height_pixels, self.map_width_pixels = self.map_img.shape[:2]
        self.get_logger().info(f"Map dimensions: {self.map_width_pixels}x{self.map_height_pixels} pixels")
        self.get_logger().info(f"Map resolution: {MAP_RESOLUTION} m/pixel")
        self.get_logger().info(f"Map origin: {MAP_ORIGIN} m")
        
        # 히트맵 레이어 초기화
        self.heatmap = np.zeros((self.map_height_pixels, self.map_width_pixels, 3), dtype=np.float32)
        
        # 탐지된 사람들의 위치 (시간에 따른 감쇠를 위해 타임스탬프 포함)
        self.detected_people = []  # [(cam_name, map_x, map_y, confidence, timestamp), ...]
        
        # 최대 저장 기간 (30초)
        self.max_history_seconds = 30
        
        # 호모그래피 행렬 계산
        self.homography_matrices = {}
        self.calculate_homography_matrices()
        
        # YOLO 모델 로드
        self.load_yolo_model()
        
        # 각 카메라 구독
        self.camera_subs = {}
        self.detection_pubs = {}
        
        camera_names = ['cctv1', 'cctv2', 'cctv3']
        for cam_name in camera_names:
            # 이미지 구독
            self.camera_subs[cam_name] = self.create_subscription(
                Image,
                f'/{cam_name}',
                lambda msg, cam=cam_name: self.image_callback(msg, cam),
                10
            )
            
            # 탐지 결과 발행
            self.detection_pubs[cam_name] = self.create_publisher(
                String,
                f'/{cam_name}/detections',
                10
            )
            
            self.get_logger().info(f'Subscribed to camera topic: /{cam_name}')
        
        # 히트맵 업데이트 타이머 (10Hz)
        self.create_timer(0.1, self.update_and_show_heatmap)
        
        # 오래된 데이터 정리 타이머 (1초마다)
        self.create_timer(1.0, self.cleanup_old_detections)
        
        # 창 초기화
        self.window_name = "CCTV Heatmap (Homography)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 800)
        
        self.get_logger().info("CCTV Homography Heatmap System Initialized Successfully!")

    def map_to_pixel(self, map_x, map_y):
        """맵 좌표(미터)를 이미지 픽셀 좌표로 변환 (coordinate_picker 기준)"""
        pixel_x = int((map_x - MAP_ORIGIN[0]) / MAP_RESOLUTION)
        # Y축 변환 복원: coordinate_picker와 동일하게 Y축 뒤집지 않음
        pixel_y = int((map_y - MAP_ORIGIN[1]) / MAP_RESOLUTION)
        # pixel_y = self.map_height_pixels - 1 - int((map_y - MAP_ORIGIN[1]) / MAP_RESOLUTION) # YAML 기준
        return pixel_x, pixel_y

    def pixel_to_map(self, pixel_x, pixel_y):
        """이미지 픽셀 좌표를 맵 좌표(미터)로 변환 (coordinate_picker 기준)"""
        map_x = pixel_x * MAP_RESOLUTION + MAP_ORIGIN[0]
        map_y = pixel_y * MAP_RESOLUTION + MAP_ORIGIN[1] # Y축 계산 복원
        # map_y = (self.map_height_pixels - 1 - pixel_y) * MAP_RESOLUTION + MAP_ORIGIN[1] # YAML 기준
        return map_x, map_y
    
    def calculate_homography_matrices(self):
        """각 카메라에 대한 호모그래피 행렬 계산"""
        for cam_name in IMAGE_COORDS.keys():
            src_points = IMAGE_COORDS[cam_name]  # 이미지 픽셀 좌표
            dst_points = MAP_COORDS[cam_name]    # 맵 좌표 (미터 단위)
            
            # 호모그래피 행렬 계산 (이미지 -> 맵)
            H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            
            if H is not None:
                self.homography_matrices[cam_name] = H
                self.get_logger().info(f"Homography matrix calculated for camera '{cam_name}'")
            else:
                self.get_logger().error(f"Failed to calculate homography matrix for camera '{cam_name}'")
    
    def load_yolo_model(self):
        """YOLO 모델 로드"""
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.classes = [0]  # person class only
            self.get_logger().info("YOLO model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def image_to_map_coords_homography(self, cam_name, img_x, img_y):
        """호모그래피 변환을 사용하여 이미지 픽셀 좌표를 맵 좌표로 변환"""
        if cam_name not in self.homography_matrices or self.homography_matrices[cam_name] is None:
            self.get_logger().warn(f"Homography matrix not available for camera '{cam_name}'")
            return None, None
        
        # 픽셀 좌표를 변환하기 위해 (1, 1, 2) 형태로 변환
        point = np.array([[img_x, img_y]], dtype=np.float32).reshape(-1, 1, 2)
        
        # 호모그래피 행렬을 적용하여 맵 좌표로 변환
        transformed_point = cv2.perspectiveTransform(point, self.homography_matrices[cam_name])
        
        if transformed_point is None or transformed_point.size == 0:
             self.get_logger().warn(f"Perspective transform failed for point ({img_x}, {img_y}) on camera '{cam_name}'")
             return None, None
             
        map_x, map_y = transformed_point[0][0]
        
        return float(map_x), float(map_y)
    
    def is_point_in_polygon(self, point, polygon):
        """점이 다각형 내부에 있는지 확인"""
        x, y = point
        polygon_np = np.array(polygon, dtype=np.int32) # Use int32 for pointPolygonTest
        result = cv2.pointPolygonTest(polygon_np, (float(x), float(y)), False)
        return result >= 0
    
    def image_callback(self, msg, cam_name):
        """카메라 이미지를 받아서 사람을 탐지하고 결과를 발행"""
        if self.model is None:
            self.get_logger().warn("YOLO model not loaded, skipping detection.")
            return
            
        try:
            # ROS 이미지 메시지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img_height, img_width = cv_image.shape[:2]
            
            # YOLO로 사람 탐지
            results = self.model(cv_image)
            
            # 탐지 결과 처리
            detections_data = [] # Data to publish
            detection_viz_image = cv_image.copy()
            
            # 이미지에 호모그래피 영역 그리기
            polygon_pts = IMAGE_COORDS[cam_name].astype(np.int32)
            cv2.polylines(detection_viz_image, [polygon_pts], True, (0, 0, 255), 2)
            cv2.putText(detection_viz_image, f"{cam_name} - Detection Area", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            n_detected_in_area = 0
            for det in results.xyxy[0]:  # Process detections for the first image
                if det[5] == 0:  # person class
                    confidence = float(det[4])
                    x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                    foot_x = float((x1 + x2) / 2)
                    foot_y = float(y2) # Bottom center
                    
                    # Check if foot point is inside the defined polygon area
                    in_area = self.is_point_in_polygon((foot_x, foot_y), IMAGE_COORDS[cam_name])
                    
                    map_x, map_y = None, None
                    if in_area:
                        # Convert image coords to map coords using homography
                        map_x, map_y = self.image_to_map_coords_homography(cam_name, foot_x, foot_y)
                        
                        if map_x is not None and map_y is not None:
                            n_detected_in_area += 1
                            # Add detection to list for publishing
                            detections_data.append({
                                "x": map_x,
                                "y": map_y,
                                "confidence": confidence,
                                "timestamp": datetime.now().isoformat(),
                                "in_area": True
                            })
                            # Add detection for heatmap visualization
                            self.detected_people.append((cam_name, map_x, map_y, confidence, datetime.now()))
                            
                            # Visualization: Blue box, show map coords
                            box_color = (255, 0, 0)
                            location_text = f"Map:({map_x:.1f}, {map_y:.1f})"
                        else:
                             # Homography failed for this point
                             box_color = (0, 165, 255) # Orange for warning
                             location_text = "Transform Error"
                             in_area = False # Treat as outside if transform failed
                    else:
                        # Outside detection area: Green box
                        box_color = (0, 255, 0)
                        location_text = "Outside Area"
                    
                    # Draw bounding box
                    cv2.rectangle(detection_viz_image, (x1, y1), (x2, y2), box_color, 2)
                    # Draw foot position
                    cv2.circle(detection_viz_image, (int(foot_x), int(foot_y)), 5, (0, 0, 255), -1)
                    # Draw location text
                    cv2.putText(detection_viz_image, location_text,
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    # Draw confidence
                    cv2.putText(detection_viz_image, f"{confidence:.2f}",
                              (x2 - 50, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            # Show detection image window (optional)
            cv2.imshow(f"Detections - {cam_name}", detection_viz_image)
            
            # Publish detection results if any detections were made in the area
            if n_detected_in_area > 0:
                detection_msg = String()
                detection_msg.data = json.dumps({
                    "detections": detections_data,
                    "camera": cam_name,
                    "timestamp": datetime.now().isoformat()
                })
                self.detection_pubs[cam_name].publish(detection_msg)
                self.get_logger().info(f"[{cam_name}] Detected {n_detected_in_area} person(s) in area.")
            
        except Exception as e:
            self.get_logger().error(f"Error processing image from {cam_name}: {e}", exc_info=True)
    
    def cleanup_old_detections(self):
        """오래된 탐지 데이터 제거"""
        now = datetime.now()
        threshold = now - timedelta(seconds=self.max_history_seconds)
        original_count = len(self.detected_people)
        self.detected_people = [det for det in self.detected_people if det[4] > threshold]
        removed = original_count - len(self.detected_people)
        #if removed > 0:
        #    self.get_logger().info(f"Removed {removed} old detections. Current count: {len(self.detected_people)}")
    
    def update_heatmap(self):
        """히트맵 업데이트"""
        # 히트맵 감쇄
        decay = 0.98
        self.heatmap = self.heatmap * decay
        
        # 강도 맵 초기화
        intensity_map = np.zeros((self.map_height_pixels, self.map_width_pixels), dtype=np.float32)
        
        for cam_name, map_x, map_y, confidence, timestamp in self.detected_people:
            # 시간 가중치
            age = (datetime.now() - timestamp).total_seconds()
            if age > self.max_history_seconds:
                continue
            time_weight = 1.0 - (age / self.max_history_seconds) * 0.7
            
            # 맵 좌표를 픽셀 좌표로 변환 (수정된 함수 사용)
            pixel_x, pixel_y = self.map_to_pixel(map_x, map_y)
            
            # 픽셀 범위 체크
            if 0 <= pixel_x < self.map_width_pixels and 0 <= pixel_y < self.map_height_pixels:
                radius = int(30 * time_weight) # 반경 조정
                # 원형 그라데이션
                for r in range(radius, 0, -1):
                    intensity = time_weight * (1 - r / radius)
                    cv2.circle(intensity_map, (pixel_x, pixel_y), r, intensity, -1)
        
        # 가우시안 블러 적용
        intensity_map = cv2.GaussianBlur(intensity_map, (15, 15), 0)
        
        # 히트맵 누적 (최대값 사용)
        intensity_map_3ch = np.stack([intensity_map] * 3, axis=2)
        self.heatmap = np.maximum(self.heatmap, intensity_map_3ch)
    
    def update_and_show_heatmap(self):
        """히트맵 업데이트 및 화면에 표시"""
        try:
            # 히트맵 업데이트
            self.update_heatmap()
            
            # 결과 표시용 맵 복사
            result_map = self.map_img.copy()
            
            # 카메라별 색상
            colors = {
                "cctv1": (0, 255, 255),   # Yellow
                "cctv2": (255, 0, 255),   # Magenta
                "cctv3": (255, 255, 0)    # Cyan
            }
            
            # 각 카메라 영역 및 위치 표시
            for cam_name in MAP_COORDS.keys():
                color = colors.get(cam_name, (255, 255, 255))
                
                # 다각형 영역 그리기 (맵 좌표 -> 픽셀 변환 사용)
                polygon_pts_map = MAP_COORDS[cam_name]
                polygon_pts_pixel = []
                for map_pt in polygon_pts_map:
                    px, py = self.map_to_pixel(map_pt[0], map_pt[1])
                    polygon_pts_pixel.append([px, py])
                
                polygon_pts_pixel = np.array(polygon_pts_pixel, dtype=np.int32)
                cv2.polylines(result_map, [polygon_pts_pixel], True, color, 2)
                
                # 카메라 위치 표시 (맵 좌표 -> 픽셀 변환 사용)
                cam_pos_map = CAMERA_POSITIONS[cam_name]
                cam_px, cam_py = self.map_to_pixel(cam_pos_map[0], cam_pos_map[1])
                
                if 0 <= cam_px < self.map_width_pixels and 0 <= cam_py < self.map_height_pixels:
                    cv2.circle(result_map, (cam_px, cam_py), 8, (0, 0, 0), -1) # Black border
                    cv2.circle(result_map, (cam_px, cam_py), 6, color, -1) # Inner color
                    cv2.putText(result_map, cam_name, (cam_px + 10, cam_py + 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(result_map, cam_name, (cam_px + 10, cam_py + 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                else:
                    self.get_logger().warn(f"Camera '{cam_name}' pixel position ({cam_px}, {cam_py}) is outside map bounds ({self.map_width_pixels}x{self.map_height_pixels})")
            
            # 히트맵 표시
            normalized = np.clip(self.heatmap, 0, 1)
            single_channel = np.max(normalized, axis=2) # Use max intensity across channels
            heatmap_colored = cv2.applyColorMap((single_channel * 255).astype(np.uint8), cv2.COLORMAP_JET)
            mask = single_channel > 0.05 # Mask for areas with significant heat
            
            alpha = 0.7 # Heatmap transparency
            for i in range(3): # Apply heatmap overlay channel by channel
                result_map[:,:,i] = np.where(
                    mask,
                    result_map[:,:,i] * (1 - alpha) + heatmap_colored[:,:,i] * alpha,
                    result_map[:,:,i]
                )
            
            # 현재 탐지된 사람 위치 표시 (최근 5초)
            now = datetime.now()
            for cam_name, map_x, map_y, confidence, timestamp in self.detected_people:
                if (now - timestamp).total_seconds() <= 5:
                    pixel_x, pixel_y = self.map_to_pixel(map_x, map_y)
                    if 0 <= pixel_x < self.map_width_pixels and 0 <= pixel_y < self.map_height_pixels:
                        color = colors.get(cam_name, (255, 255, 255))
                        # 사람 위치 표시
                        cv2.circle(result_map, (pixel_x, pixel_y), 6, (0, 0, 0), -1)  # Black border
                        cv2.circle(result_map, (pixel_x, pixel_y), 4, color, -1)  # Inner color
                    #else:
                    #    self.get_logger().debug(f"Detected person pixel ({pixel_x}, {pixel_y}) outside map bounds")
            
            # 텍스트 정보 추가 (영어로 변경)
            texts = [
                f"Detections: {len(self.detected_people)}",
                "Exit: 'q'",
                "Save: 's'",
                "Reset Heatmap: 'r'"
            ]
            
            for i, text in enumerate(texts):
                # Draw black outline first
                cv2.putText(result_map, text, (10, 30 + 30*i),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                # Draw white text over it
                cv2.putText(result_map, text, (10, 30 + 30*i),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            
            # 결과 이미지 표시
            cv2.imshow(self.window_name, result_map)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("Exiting...")
                rclpy.shutdown()
            elif key == ord('s'):
                # 이미지 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"heatmap_{timestamp}.png"
                cv2.imwrite(filename, result_map)
                self.get_logger().info(f"Image saved: {filename}")
            elif key == ord('r'):
                # 히트맵 초기화
                self.heatmap.fill(0)
                self.detected_people.clear()
                self.get_logger().info("Heatmap reset")
        
        except Exception as e:
            self.get_logger().error(f"Error during heatmap update/display: {e}", exc_info=True)

def main(args=None):
    rclpy.init(args=args)
    node = CCTVHeatmapNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main() 