''' picar-x wifi control client with human tracking support'''

import socket
import json
import threading
import cv2
import numpy as np
import base64
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from pynput import keyboard
except ImportError:
    import os
    os.system('pip3 install pynput')
    from pynput import keyboard


class PiCarClient:
    def __init__(self, pi_ip):
        # robot connection info
        self.pi_ip = pi_ip
        self.cmd_port = 5000
        self.video_port = 5001

        # sockets
        self.cmd_sock = None
        self.video_sock = None
        self.running = False

        # drift correction
        self.drift_correction = 8

        # control states
        self.speed = 0
        self.angle = 0
        self.pan = 0
        self.tilt = 0

        # display speed value for overlay
        self.display_speed = 0

        # driving mode
        self.current_mode = 'manual'

        # video state
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.frames_received = 0
        self.last_frame_time = time.time()
        self.fps = 0

        # matplotlib UI
        self.fig = None
        self.ax = None
        self.img_display = None

    def connect(self):
        # connect to robot command and video servers
        try:
            self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.cmd_sock.connect((self.pi_ip, self.cmd_port))
            print(f'Connected to commands at {self.pi_ip}:{self.cmd_port}')

            self.video_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.video_sock.connect((self.pi_ip, self.video_port))
            self.video_sock.settimeout(5.0)
            print(f'Connected to video at {self.pi_ip}:{self.video_port}')

            self.running = True
            return True

        except Exception as e:
            print(f'connection failed: {e}')
            return False

    def send_command(self, cmd: dict):
        # send json command to server
        try:
            msg = json.dumps(cmd).encode()
            self.cmd_sock.settimeout(0.5)
            self.cmd_sock.send(msg)
            try:
                self.cmd_sock.recv(1024)
            except socket.timeout:
                pass
        except Exception as e:
            print(f'command error: {e}')

    def receive_video(self):
        # receive video stream
        buffer = ''
        print('[video] starting video receiver')

        # receive video stream
        while self.running:
            try:
                chunk = self.video_sock.recv(65536) # receive chunk of data
                if not chunk:
                    print('[video] connection closed') # if connection is closed, break
                    break

                # decode chunk of data
                buffer += chunk.decode(errors='ignore')

                # while there is a newline in the buffer, split the buffer into a line and the rest of the buffer
                while '\n' in buffer: 
                    line, buffer = buffer.split('\n', 1) 
                    if not line.strip():
                        continue # if the line is empty, continue
                    
                    # try to load the line as JSON
                    try:
                        frame_data = json.loads(line)
                        img_bytes = base64.b64decode(frame_data['frame']) # decode base64 encoded image
                        img_np = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR) # decode image

                        if img is None or img.size == 0:
                            continue # if the image is None or the size is 0, continue

                        # convert image to RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_rgb = self.enhance_image(img_rgb) # enhance image
                        self.add_overlay(img_rgb)

                        # update frame with thread safety
                        with self.frame_lock:
                            self.current_frame = img_rgb # set current frame
                            self.frames_received += 1 # increment frames received

                            # calculate fps every 30 frames
                            if self.frames_received % 30 == 0:
                                now = time.time() # get current time
                                self.fps = 30 / (now - self.last_frame_time) # calculate fps
                                self.last_frame_time = now # set last frame time
                                print(f'[video] frames: {self.frames_received} | fps: {self.fps:.1f}') # print fps

                    # if the line is not JSON, skip it
                    except Exception:
                        continue

            # if the socket times out, continue
            except socket.timeout:
                continue 

            # if there is an exception, print the error and break
            except Exception as e:
                print(f'[video] stream error: {e}')
                break

        # print total frames received
        print(f'[video] stopped (total frames: {self.frames_received})')

    def enhance_image(self, img):
        # basic enhancement and sharpening
        img_float = img.astype(np.float32)

        # clip the image to the range 0-255
        img_float = np.clip((img_float - 128) * 1.3 + 128, 0, 255)
        img = img_float.astype(np.uint8)

        # create a sharpening kernel for the image
        # this sharpens the image by emphasizing edges
        # because if the middle pixel is 5/4ths brighter than the surrounding pixels, it will be emphasized
        kernel = np.array(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, 5.0, -0.5],
                [-0.5, -0.5, -0.5],
            ]
        )

        # apply the kernel to the image
        img = cv2.filter2D(img, -1, kernel)
        return img

    def add_overlay(self, img):
        # overlay control info on video
        overlay = img.copy()

        # add a black semi-transparent background to the overlay
        cv2.rectangle(overlay, (5, 5), (280, 140), (0, 0, 0), -1)
        img[:] = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        # add the current mode to the overlay
        cv2.putText(img, f'mode: {self.current_mode}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        # add the current speed to the overlay
        cv2.putText(img, f'speed: {self.display_speed}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # add the current angle to the overlay (in degrees)
        cv2.putText(img, f'angle: {self.angle}', (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # add the current pan and tilt to the overlay (in degrees)
        cv2.putText(img, f'pan: {self.pan}  tilt: {self.tilt}', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # add the current drift correction to the overlay (in degrees)
        if self.drift_correction != 0:
            cv2.putText(img, f'drift: {self.drift_correction:+d}', (10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    def update_plot(self, frame_num):
        # update video frame shown in matplotlib
        # with thread safety
        with self.frame_lock:
            # if the current frame and image display are not None
            if self.current_frame is not None:
                if self.img_display is None:

                    # then set the image display to the current frame
                    self.img_display = self.ax.imshow(self.current_frame)
                    # turn off the axis
                    self.ax.axis('off')
                    # set the title to the current frame
                    self.ax.set_title('picar-x fpv - press esc or q to quit',
                                      fontsize=12, color='white',
                                      backgroundcolor='black', pad=10)
                else:
                    self.img_display.set_array(self.current_frame)

        return [self.img_display] if self.img_display else []

    def send_move(self):
        # send movement with drift correction if needed
        corrected_angle = self.angle # get the current angle
        if self.angle == 0 and self.speed != 0:
            corrected_angle = self.drift_correction # if the angle is 0 and the speed is not 0, add the drift correction

        # send the movement command with the corrected angle
        self.send_command(
            {
                'type': 'move',
                'speed': self.speed,
                'angle': corrected_angle,
            }
        )

    def send_camera(self):
        # send camera pan and tilt command
        self.send_command(
            {
                'type': 'camera',
                'pan': self.pan,
                'tilt': self.tilt,
            }
        )

    def on_press(self, key):
        # keyboard input handling
        try:
            if key.char == 'w':
                self.speed = -50
                self.display_speed = 50
                self.send_move()
            elif key.char == 's':
                self.speed = 50
                self.display_speed = -50
                self.send_move()
            elif key.char == 'a':
                self.angle = -30
                self.send_move()
            elif key.char == 'd':
                self.angle = 30
                self.send_move()
            elif key.char == 'f':
                self.send_command({'type': 'stop'})
                print('emergency stop')

            elif key.char == '=': # this is the + button on the keyboard
                self.drift_correction += 1 # increase the drift correction 1 degree (to the right)
                self.send_command({'type': 'drift', 'value': self.drift_correction})
                print(f'drift correction: {self.drift_correction:+d}')
            elif key.char == '-': # this is the minus button on the keyboard next to +/=
                self.drift_correction -= 1 # decrease the drift correction 1 degree (to the left)
                self.send_command({'type': 'drift', 'value': self.drift_correction})
                print(f'drift correction: {self.drift_correction:+d}')
            elif key.char == '0':
                self.drift_correction = 0
                self.send_command({'type': 'drift', 'value': self.drift_correction})
                print('drift correction reset')

            elif key.char == '1':
                self.current_mode = 'manual'
                self.send_command({'type': 'mode', 'mode': 'manual'})
                print('mode: manual')
            # had other modes too previously but they were not useful
            elif key.char == '4':
                self.current_mode = 'follow_human'
                self.send_command({'type': 'mode', 'mode': 'follow_human'})
                print('mode: follow human')

            elif key.char == 'i':
                self.tilt = min(35, self.tilt + 5) # increase the tilt 5 degrees (up)
                self.send_camera()
            elif key.char == 'k':
                self.tilt = max(-35, self.tilt - 5) # decrease the tilt 5 degrees (down)
                self.send_camera()
            elif key.char == 'j':
                self.pan = max(-35, self.pan - 5) # decrease the pan 5 degrees (left)
                self.send_camera()
            elif key.char == 'l':
                self.pan = min(35, self.pan + 5) # increase the pan 5 degrees (right)
                self.send_camera()

            elif key.char == 'h':
                # center the camera
                self.pan = 0
                self.tilt = 0
                self.send_camera()
                print('camera centered')

            elif key.char == 'q':
                # quit the program
                print('quitting')
                self.running = False
                plt.close(self.fig)
                return False

        except AttributeError:
            pass

    def on_release(self, key):
        # keyboard release handling
        # keep going until button is released
        try:
            if key.char in ['w', 's']: # if the button is w or s, stop the car
                self.speed = 0
                self.display_speed = 0
                self.send_move()
            elif key.char in ['a', 'd']: # if the button is a or d, stop the car
                self.angle = 0
                self.send_move()
        except Exception:
            # if the button is esc, quit the program
            if key == keyboard.Key.esc:
                print('quitting')
                self.running = False
                plt.close(self.fig)
                return False

    def start(self):
        # start client app
        # if the connection fails, return
        if not self.connect():
            return

        print('\n' + '=' * 70)
        print('picar-x remote control with human tracking')
        print('=' * 70)
        print('movement:')
        print('  w/s     - forward/backward')
        print('  a/d     - turn left/right')
        print('  f       - emergency stop')
        print('\ncamera:')
        print('  i/k     - tilt up/down')
        print('  j/l     - pan left/right')
        print('  h       - center camera')
        print('\ndrift correction:')
        print('  +/-     - adjust steering compensation')
        print('  0       - reset drift correction')
        print('\nmodes:')
        print('  1       - manual control')
        print('  4       - follow human')
        print('\nquit:')
        print('  q/esc   - exit')
        print('=' * 70)
        print(f'\ncurrent drift correction: {self.drift_correction:+d}')
        print('starting video stream')

        # start the video thread
        video_thread = threading.Thread(target=self.receive_video, daemon=True)
        video_thread.start()

        # start the keyboard listener thread
        keyboard_listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )
        keyboard_listener.start()

        print('opening video window')

        # if 's' is in the save key map, remove it 
        # this is because anytime i would move backwards, it would save the image and stall out the car
        if 's' in plt.rcParams.get('keymap.save', []):
            plt.rcParams['keymap.save'].remove('s')

        self.fig, self.ax = plt.subplots(figsize=(10, 7.5))
        # set the window title to 'picar-x control'
        self.fig.canvas.manager.set_window_title('picar-x control')
        # set the background color to black
        self.fig.patch.set_facecolor('black')
        # set the toolbar to not be visible (this has stock commands like save image, zoom artificailly, etc which we don't need)
        self.fig.canvas.toolbar_visible = False
        # create the animation
        # this is the function that will be called to update the video frame
        # the interval is 33ms, which is 30fps
        # the blit flag is True, which means that only the changed parts of the image will be redrawn
        # the cache_frame_data flag is False, which means that the frame data will not be cached
        anim = FuncAnimation(self.fig, self.update_plot, interval=33, blit=True, cache_frame_data=False)

        # show the figure
        try:
            plt.show()
        except KeyboardInterrupt:
            pass

        self.cleanup()

    def cleanup(self):
        ''' this is still only half working because the matplotlib window is not closing properly
        but the idea is to shut down cleanly when the program is exited'''
        # shutdown
        print('cleaning up')
        self.running = False

        # close the command socket
        if self.cmd_sock:
            try:
                self.send_command({'type': 'stop'})
            except Exception:
                pass
            self.cmd_sock.close()

        # close the video socket
        if self.video_sock:
            self.video_sock.close()

        # close all matplotlib figures
        plt.close('all')
        print('disconnected')

if __name__ == '__main__':
    # check if the pi ip address is provided (should always be the same for my use case)
    if len(sys.argv) < 2:
        print('usage: python3 picar_client_yolo.py <pi_ip_address>')
        print('example: python3 picar_client_yolo.py 192.168.1.155')
        sys.exit(1)

    # create the client object and start the client
    client = PiCarClient(sys.argv[1])
    client.start()
