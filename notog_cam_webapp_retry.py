# To Run on the hostaa
'''python
PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot
'''

# To Run the teleop:
'''python
PYTHONPATH=src python -m examples.xlerobot.teleoperate_Keyboard
'''
import os
import sys
import termios
import tty
import os
import sys
import termios
import tty
import select
import threading
import cv2
import threading
import cv2
from flask import Flask, Response, request, render_template
from flask_cors import CORS

import time
import numpy as np
import math

from lerobot.robots.xlerobot import XLerobotConfig, XLerobot
# from lerobot.robots.xlerobot import XLerobotClient, XLerobotClientConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.model.SO101Robot import SO101Kinematics

HOLD_TIMEOUT_S = 0.25   # stop motion ~250ms after key repeats stop
TICK_HZ = 50            # send commands at 50 Hz while held

import atexit

_TERMIOS_OLD = None

def termios_noecho_cbreak_on():
    """Put terminal in cbreak + no-echo so we can read single keys cleanly."""
    global _TERMIOS_OLD
    fd = sys.stdin.fileno()
    _TERMIOS_OLD = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    new = termios.tcgetattr(fd)
    new[3] = new[3] & ~termios.ECHO  # lflags: disable echo
    termios.tcsetattr(fd, termios.TCSADRAIN, new)

def termios_restore():
    global _TERMIOS_OLD
    if _TERMIOS_OLD is None:
        return
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _TERMIOS_OLD)
    _TERMIOS_OLD = None

def read_key_nonblocking():
    """Nonblocking single-key reader (raw TTY).

    Returns:
      - a single printable character (e.g. 'a', '7', ';')
      - 'UP'/'DOWN'/'LEFT'/'RIGHT' if arrow keys are pressed
      - None if no input is available
    """
    fd = sys.stdin.fileno()
    dr, _, _ = select.select([fd], [], [], 0)
    if not dr:
        return None

    try:
        data = os.read(fd, 16)
    except BlockingIOError:
        return None

    if not data:
        return None

    # Arrow escape sequences: ESC [ A/B/C/D
    if data.startswith(b"\x1b[") and len(data) >= 3:
        code = data[2:3]
        return {b"A": "UP", b"B": "DOWN", b"C": "RIGHT", b"D": "LEFT"}.get(code, "\x1b")

    # Lone ESC
    if data == b"\x1b":
        return "\x1b"

    return chr(data[0])

atexit.register(termios_restore)




# Keymaps (semantic action: key)
LEFT_KEYMAP = {
    'shoulder_pan+': 'q', 'shoulder_pan-': 'e',
    'wrist_roll+': 'r', 'wrist_roll-': 'f',
    'gripper+': 't', 'gripper-': 'g',
    'x+': 'w', 'x-': 's', 'y+': 'a', 'y-': 'd',
    'pitch+': 'z', 'pitch-': 'x',
    'reset': 'c',
    # For head motors
    "head_motor_1+": "<", "head_motor_1-": ">",
    "head_motor_2+": ",", "head_motor_2-": ".",
    
    'triangle': 'y',  # Rectangle trajectory key
}
RIGHT_KEYMAP = {
    'shoulder_pan+': '7', 'shoulder_pan-': '9',
    'wrist_roll+': '/', 'wrist_roll-': '*',
    'gripper+': '+', 'gripper-': '-',
    'x+': '8', 'x-': '2', 'y+': '4', 'y-': '6',
    'pitch+': '1', 'pitch-': '3',
    'reset': '0',

    'triangle': 'Y',  # Rectangle trajectory key
}

# Base keymap (no toggle). Chosen to avoid overlaps with LEFT_KEYMAP / RIGHT_KEYMAP / head keys.
# You press these keys; we translate them into whatever keys XLerobot expects internally.
BASE_KEYMAP_LOCAL = {
    "forward": "i",
    "backward": "k",
    "left": "j",
    "right": "l",
    "rotate_left": "u",
    "rotate_right": "o",
    "speed_down": "p",
    "speed_up": "h",
}

# Fallback mapping for XLerobot._from_keyboard_to_base_action() if robot.teleop_keys is not present.
# These are *internal* keys we feed to the helper (you still press BASE_KEYMAP_LOCAL keys).
_BASE_FALLBACK_EXPECTED = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "rotate_left": ",",
    "rotate_right": ".",
    "speed_down": "-",
    "speed_up": "=",
}


LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

# Head motor mapping
HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}

class RectangularTrajectory:
    """
    Generates a rectangular trajectory on the x-y plane with sinusoidal velocity profiles.
    The rectangle is divided into 4 line segments, each with smooth acceleration/deceleration.
    """
    def __init__(self, width=0.06, height=0.06, segment_duration=0.91):
        """
        Initialize rectangular trajectory parameters.
        
        Args:
            width: Rectangle width in meters
            height: Rectangle height in meters  
            segment_duration: Time for each line segment in seconds
        """
        self.width = width
        self.height = height
        self.segment_duration = segment_duration
        self.total_duration = 4 * segment_duration
        
    def get_trajectory_point(self, current_x, current_y, t):
        """
        Get the target x, y position at time t for the rectangular trajectory.
        
        Args:
            current_x: Starting x position
            current_y: Starting y position
            t: Time since trajectory start (0 to total_duration)
            
        Returns:
            tuple: (target_x, target_y)
        """
        # Determine which segment we're in
        segment = int(t / self.segment_duration)
        segment_t = t % self.segment_duration
        
        # Normalize segment time (0 to 1)
        normalized_t = segment_t / self.segment_duration
        
        # Sinusoidal velocity profile: smooth acceleration and deceleration
        # s(t) = 0.5 * (1 - cos(π * t)) gives smooth 0 to 1 transition
        smooth_t = 0.5 * (1 - math.cos(math.pi * normalized_t))
        
        # Define rectangle corners relative to starting position
        corners = [
            (current_x, current_y),                           # Start (bottom-left)
            (current_x + self.width, current_y),              # Bottom-right
            (current_x + self.width, current_y + self.height), # Top-right  
            (current_x, current_y + self.height),             # Top-left
            (current_x, current_y)                            # Back to start
        ]
        
        # Clamp segment to valid range
        segment = max(0, min(3, segment))
        
        # Interpolate between current corner and next corner
        start_corner = corners[segment]
        end_corner = corners[segment + 1]
        
        target_x = start_corner[0] + smooth_t * (end_corner[0] - start_corner[0])
        target_y = start_corner[1] + smooth_t * (end_corner[1] - start_corner[1])
        
        return target_x, target_y

class SimpleHeadControl:
    def __init__(self, initial_obs, kp=0.81):
        self.kp = kp
        self.degree_step = 1
        # Initialize head motor positions
        self.target_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

    def move_to_zero_position(self, robot):
        self.target_positions = self.zero_pos.copy()
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_keys(self, key_state):
        if key_state.get('head_motor_1+'):
            self.target_positions["head_motor_1"] += self.degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")
        if key_state.get('head_motor_1-'):
            self.target_positions["head_motor_1"] -= self.degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")
        if key_state.get('head_motor_2+'):
            self.target_positions["head_motor_2"] += self.degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")
        if key_state.get('head_motor_2-'):
            self.target_positions["head_motor_2"] -= self.degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")

    def p_control_action(self, robot):
        obs = robot.get_observation()
        action = {}
        for motor in self.target_positions:
            current = obs.get(f"{HEAD_MOTOR_MAP[motor]}.pos", 0.0)
            error = self.target_positions[motor] - current
            control = self.kp * error
            action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + control
        return action

class SimpleTeleopArm:
    def __init__(self, kinematics, joint_map, initial_obs, prefix="left", kp=0.81):
        self.kinematics = kinematics
        self.joint_map = joint_map
        self.prefix = prefix  # To distinguish left and right arm
        self.kp = kp
        # Initial joint positions
        self.joint_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }
        # Set initial x/y to fixed values
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        # Set the degree step and xy step
        self.degree_step = 3
        self.xy_step = 0.0081
        # Set target positions to zero for P control
        self.target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        self.zero_pos = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }
        
        # Rectangular trajectory instance
        self.rectangular_trajectory = RectangularTrajectory(
            width=0.06,          # 6cm wide rectangle
            height=0.06,         # 4cm tall rectangle  
            segment_duration=1.01 # 3 seconds per line segment
        )

    def move_to_zero_position(self, robot):
        print(f"[{self.prefix}] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()  # Use copy to avoid reference issues
        
        # Reset kinematic variables to their initial state
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Don't let handle_keys recalculate wrist_flex - set it explicitly
        self.target_positions["wrist_flex"] = 0.0
        
        action = self.p_control_action(robot)
        robot.send_action(action)

    def execute_rectangular_trajectory(self, robot, fps=30):
        """
        Execute a blocking rectangular trajectory on the x-y plane.
        
        Args:
            robot: Robot instance to send actions to
            fps: Control loop frequency
        """
        print(f"[{self.prefix}] Starting rectangular trajectory...")
        print(f"[{self.prefix}] Rectangle: {self.rectangular_trajectory.width:.3f}m x {self.rectangular_trajectory.height:.3f}m")
        print(f"[{self.prefix}] Duration: {self.rectangular_trajectory.total_duration:.3f}s total")
        
        # Store starting position
        start_x = self.current_x
        start_y = self.current_y
        
        # Execute trajectory
        start_time = time.time()
        dt = 1.0 / fps
        
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Check if trajectory is complete
            if elapsed_time >= self.rectangular_trajectory.total_duration:
                print(f"[{self.prefix}] Rectangular trajectory completed!")
                break
                
            # Get target position from trajectory
            target_x, target_y = self.rectangular_trajectory.get_trajectory_point(
                start_x, start_y, elapsed_time
            )
            
            # Update current position
            self.current_x = target_x
            self.current_y = target_y
            
            # Calculate inverse kinematics
            try:
                joint2, joint3 = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
                self.target_positions["shoulder_lift"] = joint2
                self.target_positions["elbow_flex"] = joint3
                
                # Update wrist_flex coupling
                self.target_positions["wrist_flex"] = (
                    -self.target_positions["shoulder_lift"]
                    -self.target_positions["elbow_flex"]
                    + self.pitch
                )
                
                # Get action
                action = self.p_control_action(robot)
                
                # Determine which arm is executing and send appropriate action structure
                if self.prefix == "left":
                    # Send left arm action with empty actions for other components
                    robot_action = {**action, **{}, **{}, **{}}
                elif self.prefix == "right":
                    # Send right arm action with empty actions for other components
                    robot_action = {**{}, **action, **{}, **{}}
                
                # Send action to robot
                robot.send_action(robot_action)
                
                # Get observation and log data
                obs = robot.get_observation()
                log_rerun_data(obs, robot_action)
                
            except Exception as e:
                print(f"[{self.prefix}] IK failed at x={self.current_x:.4f}, y={self.current_y:.4f}: {e}")
                break
                
            # Maintain control frequency
            # busy_wait(dt)
        
        print(f"[{self.prefix}] Trajectory execution finished.")

    def handle_keys(self, key_state):
        # Joint increments
        if key_state.get('shoulder_pan+'):
            self.target_positions["shoulder_pan"] += self.degree_step
            print(f"[{self.prefix}] shoulder_pan: {self.target_positions['shoulder_pan']}")
        if key_state.get('shoulder_pan-'):
            self.target_positions["shoulder_pan"] -= self.degree_step
            print(f"[{self.prefix}] shoulder_pan: {self.target_positions['shoulder_pan']}")
        if key_state.get('wrist_roll+'):
            self.target_positions["wrist_roll"] += self.degree_step
            print(f"[{self.prefix}] wrist_roll: {self.target_positions['wrist_roll']}")
        if key_state.get('wrist_roll-'):
            self.target_positions["wrist_roll"] -= self.degree_step
            print(f"[{self.prefix}] wrist_roll: {self.target_positions['wrist_roll']}")
        if key_state.get('gripper+'):
            self.target_positions["gripper"] += self.degree_step
            # Clamp to physical limits
            self.target_positions["gripper"] = min(self.target_positions["gripper"], 60.0)
            print(f"[{self.prefix}] gripper: {self.target_positions['gripper']}")
        if key_state.get('gripper-'):
            self.target_positions["gripper"] -= self.degree_step
            # Clamp to physical limits
            self.target_positions["gripper"] = max(self.target_positions["gripper"], -32.0)
            print(f"[{self.prefix}] gripper: {self.target_positions['gripper']}")
        if key_state.get('pitch+'):
            self.pitch += self.degree_step
            print(f"[{self.prefix}] pitch: {self.pitch}")
        if key_state.get('pitch-'):
            self.pitch -= self.degree_step
            print(f"[{self.prefix}] pitch: {self.pitch}")

        # XY plane (IK)
        moved = False
        if key_state.get('x+'):
            self.current_x += self.xy_step
            moved = True
            print(f"[{self.prefix}] x+: {self.current_x:.4f}, y: {self.current_y:.4f}")
        if key_state.get('x-'):
            self.current_x -= self.xy_step
            moved = True
            print(f"[{self.prefix}] x-: {self.current_x:.4f}, y: {self.current_y:.4f}")
        if key_state.get('y+'):
            self.current_y += self.xy_step
            moved = True
            print(f"[{self.prefix}] x: {self.current_x:.4f}, y+: {self.current_y:.4f}")
        if key_state.get('y-'):
            self.current_y -= self.xy_step
            moved = True
            print(f"[{self.prefix}] x: {self.current_x:.4f}, y-: {self.current_y:.4f}")
        if moved:
            joint2, joint3 = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
            self.target_positions["shoulder_lift"] = joint2
            self.target_positions["elbow_flex"] = joint3
            print(f"[{self.prefix}] shoulder_lift: {joint2}, elbow_flex: {joint3}")

        # Wrist flex is always coupled to pitch and the other two
        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"]
            -self.target_positions["elbow_flex"]
            + self.pitch
        )
        # print(f"[{self.prefix}] wrist_flex: {self.target_positions['wrist_flex']}")

    def p_control_action(self, robot):
        obs = robot.get_observation()
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        action = {}
        for j in self.target_positions:
            error = self.target_positions[j] - current[j]
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        return action
    

# ---------- MJPEG camera streaming (side-thread) ----------

STREAM_PORT = 8080

CAM_DEVS = {
    # adjust names as you like
    "cam_a": "/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:2:1.0-video-index0",
    "cam_b": "/dev/v4l/by-path/platform-xhci-hcd.1-usb-0:1.3:1.0-video-index0",
    "cam_c": "/dev/v4l/by-path/platform-xhci-hcd.1-usb-0:1.4:1.0-video-index0",
}

_app = Flask(__name__)
_caps = {}
_caps_lock = threading.Lock()

# Global state for web keys
WEB_PRESSED_KEYS = set()
WEB_KEYS_LOCK = threading.Lock()

#def _get_cap(name):


STREAM_PORT = 8080

CAM_DEVS = {
    # adjust names as you like
    "cam_a": "/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:2:1.0-video-index0",
    "cam_b": "/dev/v4l/by-path/platform-xhci-hcd.1-usb-0:1.3:1.0-video-index0",
    "cam_c": "/dev/v4l/by-path/platform-xhci-hcd.1-usb-0:1.4:1.0-video-index0",
}

_app = Flask(__name__)
CORS(_app)  # Enable CORS for all routes
_caps = {}
_caps_lock = threading.Lock()

def _get_cap(name):
    dev = CAM_DEVS[name]
    with _caps_lock:
        if name in _caps:
            return _caps[name]
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

        # Tune for CPU/latency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 20)

        # If the camera supports MJPEG, this often reduces CPU:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        _caps[name] = cap
        print(f"[CAM] {name} opened: {cap.isOpened()}")
        return cap

def _release_cap(name):
    """Release a camera capture object so it can be reopened."""
    with _caps_lock:
        if name in _caps:
            try:
                _caps[name].release()
            except Exception:
                pass
            del _caps[name]
            print(f"[CAM] {name} released for reconnection")

# Track consecutive failures per camera
_cam_fail_counts = {}
_CAM_FAIL_THRESHOLD = 10  # After this many failures, try to reconnect

def _gen_frames(name):
    global _cam_fail_counts
    _cam_fail_counts[name] = 0
    
    while True:
        cap = _get_cap(name)
        
        # Check if camera is even open
        if not cap.isOpened():
            print(f"[CAM] {name} is not opened, attempting reconnect...")
            _release_cap(name)
            time.sleep(1.0)
            continue
        
        ok, frame = cap.read()
        
        if not ok or frame is None:
            _cam_fail_counts[name] += 1
            
            if _cam_fail_counts[name] >= _CAM_FAIL_THRESHOLD:
                print(f"[CAM] {name} failed {_cam_fail_counts[name]} times, reconnecting...")
                _release_cap(name)
                _cam_fail_counts[name] = 0
                time.sleep(1.0)
            else:
                time.sleep(0.1)
            continue
        
        # Reset failure count on success
        _cam_fail_counts[name] = 0

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

@_app.route("/<cam_name>")
def stream_cam(cam_name):
    if cam_name not in CAM_DEVS:
        return "unknown camera", 404
    return Response(_gen_frames(cam_name),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@_app.route("/")
def index():
    # Make sure 'dashboard.html' is in a 'templates' folder relative to this script
    return render_template("dashboard.html")

@_app.route("/control", methods=["POST", "OPTIONS"])
def control():
    """
    Expects JSON: { "key": "w", "active": true/false }
    Supports CORS preflight requests.
    """
    # Handle CORS preflight
    if request.method == "OPTIONS":
        response = _app.make_default_options_response()
        return response
    
    data = request.json
    key = data.get("key")
    active = data.get("active")
    
    # Map array keys from JS to our internal names if needed, 
    # but for simplicity we assume the JS sends the actual char (e.g. 'w', 'a', 's', 'd')
    # Special handling for Arrow keys if we want to map them to 'head_motor...' logic
    
    if key and active is not None:
        with WEB_KEYS_LOCK:
            if active:
                WEB_PRESSED_KEYS.add(key)
            else:
                WEB_PRESSED_KEYS.discard(key)
        print(f"[WEB] Key '{key}' active={active}")
    
    return {"status": "ok"}

# Global references for arms (set in main())
_left_arm = None
_right_arm = None

# Toggle between keyboard and web control
WEB_CONTROL_ENABLED = True  # Press 'm' to toggle

# Gripper position constants (in degrees) - calibrated from actual testing
GRIPPER_OPEN_POS = 60.0     # Fully open
GRIPPER_CLOSED_POS = -32.0  # Fully closed (physical limit)
GRIPPER_MIN = -32.0         # Minimum allowed position (closed limit)
GRIPPER_MAX = 60.0          # Maximum allowed position (open limit)

@_app.route("/gripper", methods=["POST", "OPTIONS"])
def gripper_control():
    """
    Direct gripper position control.
    Expects JSON: { "hand": "left"|"right", "state": "OPEN"|"CLOSED" }
    This sets the gripper target position directly instead of simulating key presses.
    """
    global _left_arm, _right_arm
    
    # Handle CORS preflight
    if request.method == "OPTIONS":
        response = _app.make_default_options_response()
        return response
    
    data = request.json
    hand = data.get("hand")  # "left" or "right"
    state = data.get("state")  # "OPEN" or "CLOSED"
    
    if hand not in ["left", "right"] or state not in ["OPEN", "CLOSED"]:
        return {"status": "error", "message": "Invalid hand or state"}, 400
    
    # Get the target position based on state (clamped to physical limits)
    target_pos = GRIPPER_OPEN_POS if state == "OPEN" else GRIPPER_CLOSED_POS
    target_pos = max(GRIPPER_MIN, min(GRIPPER_MAX, target_pos))
    
    # Set the gripper target position directly
    arm = _left_arm if hand == "left" else _right_arm
    if arm is None:
        return {"status": "error", "message": "Arms not initialized"}, 500
    
    old_pos = arm.target_positions.get("gripper", 0)
    arm.target_positions["gripper"] = target_pos
    print(f"[GRIPPER] {hand} {state}: {old_pos:.1f} -> {target_pos:.1f}")
    
    return {"status": "ok", "hand": hand, "state": state, "position": target_pos}

@_app.route("/gripper_proportional", methods=["POST", "OPTIONS"])
def gripper_proportional():
    """
    Proportional gripper position control.
    Expects JSON: { "hand": "left"|"right", "openness": 0.0-1.0 }
    Maps openness (0=closed, 1=open) to gripper degrees (GRIPPER_MIN to GRIPPER_MAX).
    """
    global _left_arm, _right_arm, WEB_CONTROL_ENABLED
    
    # Handle CORS preflight
    if request.method == "OPTIONS":
        response = _app.make_default_options_response()
        return response
    
    # Skip if web control is disabled (keyboard mode)
    if not WEB_CONTROL_ENABLED:
        return {"status": "skipped", "reason": "web_control_disabled"}
    
    data = request.json
    hand = data.get("hand")  # "left" or "right"
    openness = data.get("openness")  # 0.0 to 1.0
    
    if hand not in ["left", "right"]:
        return {"status": "error", "message": "Invalid hand"}, 400
    
    if openness is None or not isinstance(openness, (int, float)):
        return {"status": "error", "message": "Invalid openness value"}, 400
    
    # Clamp openness to valid range
    openness = max(0.0, min(1.0, float(openness)))
    
    # Map openness to gripper position: 0.0 = GRIPPER_MIN, 1.0 = GRIPPER_MAX
    target_pos = GRIPPER_MIN + (openness * (GRIPPER_MAX - GRIPPER_MIN))
    
    # Set the gripper target position
    arm = _left_arm if hand == "left" else _right_arm
    if arm is None:
        return {"status": "error", "message": "Arms not initialized"}, 500
    
    arm.target_positions["gripper"] = target_pos
    
    return {"status": "ok", "hand": hand, "openness": openness, "position": target_pos}

# Arm joint limits (in degrees) - adjust based on robot calibration
ELBOW_MIN = -90.0   # Fully extended (straight arm) - negative for extension
ELBOW_MAX = 90.0    # Fully bent
SHOULDER_MIN = -90.0  # Arm down
SHOULDER_MAX = 90.0   # Arm up

@_app.route("/arm_joints", methods=["POST", "OPTIONS"])
def arm_joints():
    """
    Control arm joints directly.
    Expects JSON: { "hand": "left"|"right", "elbow_angle": 0-180, "shoulder_angle": 0-180 }
    Maps human angles to robot joints.
    """
    global _left_arm, _right_arm, WEB_CONTROL_ENABLED
    
    # Handle CORS preflight
    if request.method == "OPTIONS":
        response = _app.make_default_options_response()
        return response
    
    # Skip if web control is disabled (keyboard mode)
    if not WEB_CONTROL_ENABLED:
        return {"status": "skipped", "reason": "web_control_disabled"}
    
    data = request.json
    hand = data.get("hand")  # "left" or "right"
    elbow_angle = data.get("elbow_angle")  # 0-180 degrees
    shoulder_angle = data.get("shoulder_angle")  # 0-180 degrees
    
    if hand not in ["left", "right"]:
        return {"status": "error", "message": "Invalid hand"}, 400
    
    # Get the arm
    arm = _left_arm if hand == "left" else _right_arm
    if arm is None:
        return {"status": "error", "message": "Arms not initialized"}, 500
    
    result = {"status": "ok", "hand": hand}
    
    # Handle elbow angle if provided
    if elbow_angle is not None and isinstance(elbow_angle, (int, float)):
        elbow_angle = max(0, min(180, float(elbow_angle)))
        # Map: Human 180° (straight) → Robot -90° (extended)
        #      Human 90° (bent) → Robot 0°
        #      Human 0° (very bent) → Robot +90°
        robot_elbow = 90 - elbow_angle  # Human 180→-90, 90→0, 0→90
        robot_elbow = max(ELBOW_MIN, min(ELBOW_MAX, robot_elbow))
        arm.target_positions["elbow_flex"] = robot_elbow
        result["elbow"] = {"human": elbow_angle, "robot": robot_elbow}
    
    # Handle shoulder angle if provided (DISABLED for LEFT arm - motor fried)
    if shoulder_angle is not None and isinstance(shoulder_angle, (int, float)):
        if hand == "right":  # Only right arm has working shoulder
            shoulder_angle = max(0, min(180, float(shoulder_angle)))
            robot_shoulder = -(shoulder_angle - 90)  # Negated
            robot_shoulder = max(SHOULDER_MIN, min(SHOULDER_MAX, robot_shoulder))
            arm.target_positions["shoulder_lift"] = robot_shoulder
            result["shoulder"] = {"human": shoulder_angle, "robot": robot_shoulder}
        else:
            result["shoulder"] = {"status": "disabled", "reason": "left_motor_fried"}
    
    # Handle wrist angle if provided
    wrist_angle = data.get("wrist_angle")  # 0-180 degrees
    if wrist_angle is not None and isinstance(wrist_angle, (int, float)):
        wrist_angle = max(0, min(180, float(wrist_angle)))
        # Map: Human 90° (neutral) → Robot 0°
        #      Human 180° (up) → Robot -90°
        #      Human 0° (down) → Robot +90°
        robot_wrist = 90 - wrist_angle  # Similar to elbow mapping
        robot_wrist = max(-90, min(90, robot_wrist))  # Clamp to safe range
        arm.target_positions["wrist_flex"] = robot_wrist
        result["wrist"] = {"human": wrist_angle, "robot": robot_wrist}
        print(f"[{hand}] wrist_flex: {robot_wrist:.1f} (human: {wrist_angle:.1f})")
    else:
        # Debug: wrist_angle not received
        if wrist_angle is None:
            pass  # No wrist data sent (hand not visible)
    
    return result

def start_stream_server():
    # threaded=True lets multiple clients/cams connect
    # Important: host='0.0.0.0' to allow external connections
    _app.run(host="0.0.0.0", port=STREAM_PORT, threaded=True, use_reloader=False)

def main():
    # Teleop parameters
    FPS = 50
    # ip = "192.168.1.123"  # This is for zmq connection
    ip = "localhost"  # This is for local/wired connection
    robot_name = "my_xlerobot_pc"

    # For zmq connection
    # robot_config = XLerobotClientConfig(remote_ip=ip, id=robot_name)
    # robot = XLerobotClient(robot_config)    

    # For local/wired connection
    robot_config = XLerobotConfig(
    port1="/dev/xle_left",
    port2="/dev/xle_right",
)

    robot = XLerobot(robot_config)
    
    try:
        robot.connect()
        print(f"[MAIN] Successfully connected to robot")
    except Exception as e:
        print(f"[MAIN] Failed to connect to robot: {e}")
        print(robot_config)
        print(robot_config)
        print(robot)
        return

    # Start camera streaming in background
    t = threading.Thread(target=start_stream_server, daemon=True)
    t.start()
    print(f"[STREAM] MJPEG server running on port {STREAM_PORT}")
    print("[STREAM] URLs:")
    for k in CAM_DEVS:
        print(f"  http://<PI_IP>:{STREAM_PORT}/{k}")

    _RERUN_ON = os.environ.get('RERUN','on').lower() not in ('0','false','off','no')
    if _RERUN_ON:
        init_rerun(session_name="xlerobot_teleop_v2")

    #Init the keyboard instance
   # keyboard_config = KeyboardTeleopConfig()
   # keyboard = KeyboardTeleop(keyboard_config)
   # keyboard.connect()

    # Init the arm and head instances
    obs = robot.get_observation()
    kin_left = SO101Kinematics()
    kin_right = SO101Kinematics()
    left_arm = SimpleTeleopArm(kin_left, LEFT_JOINT_MAP, obs, prefix="left")
    right_arm = SimpleTeleopArm(kin_right, RIGHT_JOINT_MAP, obs, prefix="right")
    head_control = SimpleHeadControl(obs)
    
    # Set global references for web gripper control
    global _left_arm, _right_arm
    _left_arm = left_arm
    _right_arm = right_arm

    # Move both arms and head to zero position at start
    left_arm.move_to_zero_position(robot)
    right_arm.move_to_zero_position(robot)

    try:
        termios_noecho_cbreak_on()
        pressed_times = {}  # key -> last seen time (for hold behavior)
        dt = 1.0 / TICK_HZ
        while True:
            ch = read_key_nonblocking()
            now = time.time()

            if ch is not None:
                # quit on ESC or Ctrl-C
                if ch in ("\x1b", "\x03"):
                    break
                # record key as "held" (we rely on key-repeat to keep it alive)
                pressed_times[ch] = now

            # prune stale held keys
            stale = [k for k, t in pressed_times.items() if (now - t) > HOLD_TIMEOUT_S]
            for k in stale:
                pressed_times.pop(k, None)

            # Merge local keys and web keys
            local_keys = set(pressed_times.keys())
            with WEB_KEYS_LOCK:
                web_keys = WEB_PRESSED_KEYS.copy()
            
            pressed_keys = local_keys | web_keys


            # Use the same mapping code already in the example.
            # This expects keys like "w", "a", "s", "d", etc.

            left_key_state = {action: (key in pressed_keys) for action, key in LEFT_KEYMAP.items()}
            right_key_state = {action: (key in pressed_keys) for action, key in RIGHT_KEYMAP.items()}

            # Handle rectangular trajectory for left arm (y key)
            if left_key_state.get('triangle'):
                print("[MAIN] Left arm rectangular trajectory triggered!")
                left_arm.execute_rectangular_trajectory(robot, fps=FPS)
                continue

            # Handle rectangular trajectory for right arm (Y key)  
            if right_key_state.get('triangle'):
                print("[MAIN] Right arm rectangular trajectory triggered!")
                right_arm.execute_rectangular_trajectory(robot, fps=FPS)
                continue

            # Handle reset for left arm
            if left_key_state.get('reset'):
                left_arm.move_to_zero_position(robot)
                continue  

            # Handle reset for right arm
            if right_key_state.get('reset'):
                right_arm.move_to_zero_position(robot)
                continue

            left_arm.handle_keys(left_key_state)
            right_arm.handle_keys(right_key_state)
            head_control.handle_keys(left_key_state)  # Head controlled by left arm keymap

            left_action = left_arm.p_control_action(robot)
            right_action = right_arm.p_control_action(robot)
            head_action = head_control.p_control_action(robot)

            # Base action (no toggle): only feed base-related keys into the helper to avoid interference.
            held_base_intents = {intent for intent, key in BASE_KEYMAP_LOCAL.items() if key in pressed_keys}

            expected_map = getattr(robot, "teleop_keys", None)
            if isinstance(expected_map, dict):
                base_keys_for_helper = [expected_map[intent] for intent in held_base_intents if intent in expected_map]
            else:
                base_keys_for_helper = [_BASE_FALLBACK_EXPECTED[intent] for intent in held_base_intents if intent in _BASE_FALLBACK_EXPECTED]

            keyboard_keys = np.array(base_keys_for_helper, dtype=object)
            base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}

            action = {**left_action, **right_action, **head_action, **base_action}
            
            # Debug: Show all arm joint positions being sent
            import json
            arm_data = {
                "left": {
                    "shoulder": action.get('left_arm_shoulder_lift.pos'),
                    "elbow": action.get('left_arm_elbow_flex.pos'),
                    "gripper": action.get('left_arm_gripper.pos')
                },
                "right": {
                    "shoulder": action.get('right_arm_shoulder_lift.pos'),
                    "elbow": action.get('right_arm_elbow_flex.pos'),
                    "gripper": action.get('right_arm_gripper.pos')
                }
            }
            print(f"[JOINTS] {json.dumps(arm_data)}")
            
            robot.send_action(action)

            obs = robot.get_observation()
            # print(f"[MAIN] Observation: {obs}")
            if _RERUN_ON:
                log_rerun_data(obs, action)
            # busy_wait(1.0 / FPS)
            time.sleep(dt)
    finally:
        robot.disconnect()
    #    keyboard.disconnect()
        print("Teleoperation ended.")

if __name__ == "__main__":
    main()
