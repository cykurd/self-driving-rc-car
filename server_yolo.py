''' picar-x wifi server with yolov8 human detection'''

import socket
import json
import threading
import time
import cv2
import base64
import numpy as np
from picamera2 import Picamera2
from picarx import Picarx

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print('ultralytics not installed. run: pip3 install ultralytics --break-system-packages')


class PiCarServer:
    def __init__(self):
        self.px = Picarx()
        self.camera = None

        # server config
        self.cmd_port = 5000
        self.video_port = 5001

        # state
        self.running = False
        self.mode = 'manual'

        # video stats
        self.frames_sent = 0

        # drift correction
        self.drift_correction = 8

        # human detection setup
        self.yolo_model = None
        self.human_detection_enabled = False
        self.last_human_detection = 0
        self.detection_confidence_threshold = 0.45
        self.detection_timeout = 2.0

        # camera pan/tilt state
        self.current_pan = 0
        self.current_tilt = 0
        self.max_pan = 40

        if YOLO_AVAILABLE:
            self.init_yolo()

    def init_yolo(self):
        '''initialize trained human detection model'''
        try:
            from pathlib import Path
            script_dir = Path(__file__).parent
            model_path = script_dir / 'best.pt'

            if not model_path.exists():
                print(f'model not found at {model_path}, using yolov8n.pt')
                model_path = 'yolov8n.pt'
            else:
                print(f'loading human detection model from {model_path}')

            self.yolo_model = YOLO(str(model_path))

            dummy_frame = np.zeros((640, 480, 3), dtype=np.uint8)
            self.yolo_model(dummy_frame, verbose=False)

            print('human detection model initialized')
        except Exception as e:
            print(f'model initialization failed: {e}')
            self.yolo_model = None

    def init_camera(self):
        '''initialize camera'''
        try:
            print('initializing camera')
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={'size': (640, 480), 'format': 'RGB888'}
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(2)
            print('camera initialized')
            return True
        except Exception as e:
            print(f'camera initialization failed: {e}')
            return False

    def handle_commands(self, conn, addr):
        '''handle command socket'''
        print(f'command client connected: {addr}')

        try:
            while self.running:
                data = conn.recv(4096)
                if not data:
                    break

                try:
                    cmd = json.loads(data.decode())
                    self.process_command(cmd)
                    conn.send(b'OK')
                except Exception as e:
                    print(f'command error: {e}')
                    conn.send(b'ERR')
        except Exception as e:
            print(f'command handler error: {e}')
        finally:
            conn.close()
            print('command client disconnected')

    def process_command(self, cmd):
        '''process commands'''
        cmd_type = cmd.get('type')

        if cmd_type == 'move':
            if self.mode == 'manual':
                speed = cmd.get('speed', 0)
                angle = cmd.get('angle', 0)
                self.px.set_dir_servo_angle(angle)
                self.px.forward(speed)

        elif cmd_type == 'stop':
            self.px.forward(0)
            self.px.set_dir_servo_angle(0)

        elif cmd_type == 'camera':
            pan = cmd.get('pan', 0)
            tilt = cmd.get('tilt', 0)
            self.current_pan = pan
            self.current_tilt = tilt
            self.px.set_cam_pan_angle(pan)
            self.px.set_cam_tilt_angle(tilt)

        elif cmd_type == 'drift':
            self.drift_correction = cmd.get('value', 0)
            print(f'drift correction updated to {self.drift_correction:+d}')

        elif cmd_type == 'mode':
            new_mode = cmd.get('mode', 'manual')
            if new_mode == 'follow_human':
                if self.yolo_model is not None:
                    self.mode = 'follow_human'
                    self.human_detection_enabled = True
                    self.reset_camera_and_stop()
                    print('mode switched to follow human')
                else:
                    print('model unavailable, staying in manual')
                    self.mode = 'manual'
                    self.human_detection_enabled = False
            else:
                self.mode = 'manual'
                self.human_detection_enabled = False
                self.px.forward(0)
                self.px.set_dir_servo_angle(0)
                print('mode switched to manual')

    def reset_camera_and_stop(self):
        '''reset camera and stop movement'''
        self.current_pan = 0
        self.current_tilt = 0
        self.px.set_cam_pan_angle(0)
        self.px.set_cam_tilt_angle(0)
        self.px.forward(0)
        self.px.set_dir_servo_angle(0)

    def detect_human(self, frame_bgr):
        '''return single best human detection above threshold'''
        if self.yolo_model is None:
            return None

        results = self.yolo_model(
            frame_bgr,
            verbose=False,
            conf=self.detection_confidence_threshold,
            imgsz=640
        )

        best = None
        best_conf = 0

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                if class_id == 0 and conf > self.detection_confidence_threshold:
                    if conf > best_conf:
                        best_conf = conf
                        best = box

        if best is None:
            return None

        x1, y1, x2, y2 = best.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        area = (x2 - x1) * (y2 - y1)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return cx, cy, area, (x1, y1, x2, y2)

    def follow_human(self, frame_bgr):
        '''adjust speed based on bounding box area only'''
        detection = self.detect_human(frame_bgr)
        if not detection:
            if time.time() - self.last_human_detection > self.detection_timeout:
                self.px.forward(0)
                self.px.set_dir_servo_angle(0)
            return None

        self.last_human_detection = time.time()
        cx, cy, area, bbox = detection

        target_area = 35000
        tolerance = 3000

        if area > target_area + tolerance:
            speed = 20
            steering = 0
            status = 'too close, reversing'
        elif area < target_area - tolerance:
            speed = -25
            steering = self.drift_correction
            status = 'too far, moving forward'
        else:
            speed = 0
            steering = 0
            status = 'good distance, holding'
        
        print(f'human tracking: {status} (area: {area}, drift: {steering})')
        self.px.set_dir_servo_angle(steering)
        self.px.forward(speed)
        return cx, cy, area, bbox

    def stream_video(self, conn, addr):
        '''stream video frames to client'''
        print(f'video client connected: {addr}')
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, 60,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1
        ]

        try:
            while self.running:
                frame_bgr = self.camera.capture_array()

                if self.human_detection_enabled and self.mode == 'follow_human':
                    detection = self.follow_human(frame_bgr)
                    if detection:
                        cx, cy, area, bbox = detection
                        if bbox:
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.circle(frame_bgr, (cx, cy), 10, (0, 255, 255), -1)

                frame_small = cv2.resize(frame_bgr, (320, 240))
                success, buffer = cv2.imencode('.jpg', frame_small, encode_params)
                if not success:
                    time.sleep(0.05)
                    continue

                jpg = base64.b64encode(buffer).decode('utf-8')
                packet = {'frame': jpg, 't': time.time()}
                conn.sendall((json.dumps(packet) + '\n').encode('utf-8'))

                self.frames_sent += 1
                if self.frames_sent % 30 == 0:
                    print(f'video frames sent: {self.frames_sent}')

                time.sleep(0.05)

        except Exception as e:
            print(f'video stream error: {e}')
        finally:
            conn.close()
            print(f'video connection closed, {self.frames_sent} frames sent')
            self.frames_sent = 0

    def start(self):
        '''start server'''
        print('starting picar-x wifi server')

        if not self.init_camera():
            return

        self.running = True

        cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        cmd_sock.bind(('0.0.0.0', self.cmd_port))
        cmd_sock.listen(1)

        video_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        video_sock.bind(('0.0.0.0', self.video_port))
        video_sock.listen(1)

        print('server ready')
        print(f'command port: {self.cmd_port}')
        print(f'video port: {self.video_port}')
        print('waiting for client connection')

        try:
            while self.running:
                cmd_conn, cmd_addr = cmd_sock.accept()
                threading.Thread(target=self.handle_commands, args=(cmd_conn, cmd_addr), daemon=True).start()

                video_conn, video_addr = video_sock.accept()
                threading.Thread(target=self.stream_video, args=(video_conn, video_addr), daemon=True).start()

        except KeyboardInterrupt:
            print('shutting down')
        finally:
            self.cleanup()

    def cleanup(self):
        '''stop motors, stop camera, and close server'''
        self.running = False
        self.px.forward(0)
        self.px.set_dir_servo_angle(0)

        if self.camera:
            self.camera.stop()

        print('server stopped')


if __name__ == '__main__':
    server = PiCarServer()
    server.start()
