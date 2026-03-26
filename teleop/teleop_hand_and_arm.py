import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import cv2
import numpy as np
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.surround_camera import SurroundCamera
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller', 'waldo'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client (Unitree's eyes)
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")

        # apriltag head tracker
        head_tracker = None
        intr = camera_config['head_camera'].get('intrinsics')
        if intr:
            from teleop.utils.apriltag_tracker import AprilTagTracker
            head_tracker = AprilTagTracker(
                camera_params=(intr['fx'], intr['fy'], intr['cx'], intr['cy']),
            )
            logger_mp.info("AprilTag head tracker initialized.")
        else:
            logger_mp.warning("No camera intrinsics in cam_config — AprilTag head tracking disabled.")

        # surrounding camera (local RealSense D405)
        surround_cam = SurroundCamera()
        logger_mp.info("Surrounding camera initialized.")
        from teleop.utils.apriltag_tracker import AprilTagTracker
        surround_tracker = AprilTagTracker(
            camera_params=(surround_cam.intrinsics['fx'], surround_cam.intrinsics['fy'],
                           surround_cam.intrinsics['cx'], surround_cam.intrinsics['cy']),
        )
        logger_mp.info("AprilTag surround tracker initialized.")

        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     )

        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")
        
        # begin waldogate
        if args.input_mode != "waldo":
            # arm
            if args.arm == "G1_29":
                arm_ik = G1_29_ArmIK()
                arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
            elif args.arm == "G1_23":
                arm_ik = G1_23_ArmIK()
                arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
            elif args.arm == "H1_2":
                arm_ik = H1_2_ArmIK()
                arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
            elif args.arm == "H1":
                arm_ik = H1_ArmIK()
                arm_ctrl = H1_ArmController(simulation_mode=args.sim)

            # end-effector
            if args.ee == "dex3":
                from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
                left_hand_pos_array = Array('d', 75, lock = True)
                right_hand_pos_array = Array('d', 75, lock = True)
                dual_hand_data_lock = Lock()
                dual_hand_state_array = Array('d', 14, lock = False)
                dual_hand_action_array = Array('d', 14, lock = False)
                hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock,
                                              dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
            elif args.ee == "dex1":
                from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
                left_gripper_value = Value('d', 0.0, lock=True)
                right_gripper_value = Value('d', 0.0, lock=True)
                dual_gripper_data_lock = Lock()
                dual_gripper_state_array = Array('d', 2, lock=False)
                dual_gripper_action_array = Array('d', 2, lock=False)
                gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock,
                                                         dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
            elif args.ee == "inspire_dfx":
                from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
                left_hand_pos_array = Array('d', 75, lock = True)
                right_hand_pos_array = Array('d', 75, lock = True)
                dual_hand_data_lock = Lock()
                dual_hand_state_array = Array('d', 12, lock = False)
                dual_hand_action_array = Array('d', 12, lock = False)
                hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
            elif args.ee == "inspire_ftp":
                from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
                left_hand_pos_array = Array('d', 75, lock = True)
                right_hand_pos_array = Array('d', 75, lock = True)
                dual_hand_data_lock = Lock()
                dual_hand_state_array = Array('d', 12, lock = False)
                dual_hand_action_array = Array('d', 12, lock = False)
                hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
            elif args.ee == "brainco":
                from teleop.robot_control.robot_hand_brainco import Brainco_Controller
                left_hand_pos_array = None  # no left hand connected
                right_hand_pos_array = Array('d', 75, lock = True)
                dual_hand_data_lock = Lock()
                dual_hand_state_array = Array('d', 12, lock = False)
                dual_hand_action_array = Array('d', 12, lock = False)
                hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock,
                                               dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
            else:
                pass
        else:
            # TODO: Waldo init
            # Non-waldo creates the following. Waldo must produce equivalents:
            #
            # 1. arm_ik: IK solver that converts 4x4 SE(3) wrist poses to joint angles.
            #    - solve_ik(left_wrist_4x4, right_wrist_4x4, current_q, current_dq) -> (sol_q, sol_tauff)
            #    - sol_q shape depends on --arm:
            #        G1_29: (14,) = left[0:7] + right[7:14] - shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
            #        G1_23: (10,) = left[0:5] + right[5:10] - shoulder_pitch/roll/yaw, elbow, wrist_roll
            #        H1_2:  (14,) = left[0:7] + right[7:14] - shoulder_pitch/roll/yaw, elbow_pitch/roll, wrist_pitch/yaw
            #        H1:    (8,)  = left[0:4] + right[4:8]  - shoulder_pitch/roll/yaw, elbow
            #    - sol_tauff: feedforward torques (same shape), gravity compensation via pinocchio RNEA.
            #      Can be zeros if your controller handles gravity comp internally.
            #    Waldo bypasses IK entirely since you receive joint angles directly from ports.
            #
            # 2. arm_ctrl: DDS arm controller with 250Hz background thread publishing motor commands.
            #    Must expose these methods (used in main loop, recording, and shutdown):
            #    - ctrl_dual_arm(q_target, tauff_target): send joint angle targets to robot
            #    - get_current_dual_arm_q() -> np.array: current joint positions (same shape as sol_q)
            #    - get_current_dual_arm_dq() -> np.array: current joint velocities (same shape)
            #    - get_current_motor_q() -> np.array: all body motor positions (used in controller recording)
            #    - ctrl_dual_arm_go_home(): return arms to home position (called in finally block)
            #    - speed_gradual_max(t=5.0): ramp velocity limit 20->30 rad/s over t seconds (called at startup)
            #    Safety: velocity clipping at 20-30 rad/s, control_dt = 1/250s
            #
            # 3. hand_ctrl: child process reading hand skeleton from shared memory, running DexPilot
            #    retargeting, and publishing motor commands via DDS. Motor counts per ee type:
            #      dex3:        7 motors/hand - thumb(3) + index(2) + middle(2), MotorCmds_ via DDS
            #      dex1:        1 motor/hand  - single gripper, linear map from pinch/trigger distance
            #      inspire_dfx: 6 motors/hand - fingers(4) + thumb_bend(1) + thumb_rot(1), normalized [0,1]
            #      inspire_ftp: 6 motors/hand - same joints, proprietary protocol scaled [0-1000]
            #      brainco:     6 motors/hand - thumb(2) + fingers(4), inverted normalization [0,1]
            #    Waldo bypasses retargeting since you receive hand joint angles directly from ports.

            # Waldo arm controller: receives joint angles from stream_arm_zmq.py via ZMQ (port 5557),
            # computes feedforward torques, and publishes motor commands via DDS.
            from teleop.robot_control.waldo_rt_arm import Waldo_Arm_Controller
            arm_ctrl = Waldo_Arm_Controller(motion_mode=args.motion, simulation_mode=args.sim)

            # Waldo hand controller: receives joint angles from inference_server via ZMQ,
            # publishes to brainco motors via DDS, updates recording arrays internally.
            if args.ee == "brainco":
                from teleop.robot_control.waldo_rt_brainco import Waldo_Brainco_Controller
                dual_hand_data_lock = Lock()
                dual_hand_state_array = Array('d', 12, lock=False)   # [output] left(6) + right(6) hand state
                dual_hand_action_array = Array('d', 12, lock=False)  # [output] left(6) + right(6) hand action
                dual_hand_dq_array = Array('d', 12, lock=False)      # [output] left(6) + right(6) hand velocity
                dual_hand_tau_array = Array('d', 12, lock=False)     # [output] left(6) + right(6) hand current
                hand_ctrl = Waldo_Brainco_Controller(
                    dual_hand_data_lock=dual_hand_data_lock,
                    dual_hand_state_array=dual_hand_state_array,
                    dual_hand_action_array=dual_hand_action_array,
                    dual_hand_dq_array=dual_hand_dq_array,
                    dual_hand_tau_array=dual_hand_tau_array,
                    simulation_mode=args.sim,
                )
            else:
                pass

        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            cam_h, cam_w = camera_config['head_camera']['image_shape']

            # arm joint names per robot type (left then right, matching qpos order)
            ARM_JOINT_NAMES = {
                "G1_29": {
                    "left_arm":  ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw"],
                    "right_arm": ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"],
                },
                "G1_23": {
                    "left_arm":  ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll"],
                    "right_arm": ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll"],
                },
                "H1_2": {
                    "left_arm":  ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow_pitch", "left_elbow_roll", "left_wrist_pitch", "left_wrist_yaw"],
                    "right_arm": ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow_pitch", "right_elbow_roll", "right_wrist_pitch", "right_wrist_yaw"],
                },
                "H1": {
                    "left_arm":  ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow"],
                    "right_arm": ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"],
                },
            }
            # end-effector joint names per hand type
            EE_JOINT_NAMES = {
                "dex3": {
                    "left_ee":  ["left_thumb_0", "left_thumb_1", "left_thumb_2", "left_middle_0", "left_middle_1", "left_index_0", "left_index_1"],
                    "right_ee": ["right_thumb_0", "right_thumb_1", "right_thumb_2", "right_index_0", "right_index_1", "right_middle_0", "right_middle_1"],
                },
                "dex1": {
                    "left_ee":  ["left_gripper"],
                    "right_ee": ["right_gripper"],
                },
                "inspire_dfx": {
                    "left_ee":  ["left_pinky", "left_ring", "left_middle", "left_index", "left_thumb_bend", "left_thumb_rotation"],
                    "right_ee": ["right_pinky", "right_ring", "right_middle", "right_index", "right_thumb_bend", "right_thumb_rotation"],
                },
                "inspire_ftp": {
                    "left_ee":  ["left_pinky", "left_ring", "left_middle", "left_index", "left_thumb_bend", "left_thumb_rotation"],
                    "right_ee": ["right_pinky", "right_ring", "right_middle", "right_index", "right_thumb_bend", "right_thumb_rotation"],
                },
                "brainco": {
                    "left_ee":  ["left_thumb", "left_thumb_aux", "left_index", "left_middle", "left_ring", "left_pinky"],
                    "right_ee": ["right_thumb", "right_thumb_aux", "right_index", "right_middle", "right_ring", "right_pinky"],
                },
            }
            arm_names = ARM_JOINT_NAMES.get(args.arm, {"left_arm": [], "right_arm": []})
            ee_names = EE_JOINT_NAMES.get(args.ee, {"left_ee": [], "right_ee": []})
            joint_names = {**arm_names, **ee_names, "body": []}

            # build metadata from camera_config and args
            head_cam_cfg = camera_config.get('head_camera', {})
            camera_metadata = {
                "id": "head_camera",
                "serial_number": head_cam_cfg.get('serial_number'),
                "frame": "world",
                "color_intrinsics": head_cam_cfg.get('color_intrinsics'),
                "depth_intrinsics": head_cam_cfg.get('depth_intrinsics'),
                "depth_scale": head_cam_cfg.get('depth_scale'),
            }
            surround_metadata = {
                "id": "surround_camera",
                "serial_number": surround_cam.serial,
                "frame": "world",
                "color_intrinsics": surround_cam.color_intrinsics,
                "depth_intrinsics": surround_cam.depth_intrinsics,
                "depth_scale": surround_cam.depth_scale,
            }
            metadata = {
                "robot_config": {
                    "arm_type": args.arm,
                    "ee_type": args.ee,
                    "control_frequency": args.frequency,
                    "img_server_ip": args.img_server_ip,
                },
                "camera_config": {
                    "width": cam_w,
                    "height": cam_h,
                    "fps": head_cam_cfg.get('fps', args.frequency),
                    "cameras": [camera_metadata, surround_metadata],
                },
            }

            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency,
                                     image_size = [cam_w, cam_h],
                                     joint_names = joint_names,
                                     metadata = metadata,
                                     rerun_log = not args.headless)

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                #regardless, let it render to headset
                tv_wrapper.render_to_xr(head_img)
            elif camera_config['head_camera']['enable_zmq']:
                head_img = img_client.get_head_frame()

            if camera_config['head_camera'].get('enable_depth') and args.record:
                head_depth = img_client.get_head_depth_frame()

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")

        #begin waldogate
        if args.input_mode != "waldo":
            arm_ctrl.speed_gradual_max()
        else:
            # Waldo arm_ctrl exposes same interface, ramp velocity the same way
            arm_ctrl.speed_gradual_max()
        #end waldogate
        
        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                    if xr_need_local_img:
                        tv_wrapper.render_to_xr(head_img)
            if camera_config['head_camera'].get('enable_depth') and args.record:
                try:
                    head_depth = img_client.get_head_depth_frame()
                except Exception as e:
                    logger_mp.warning(f"[Depth] get_head_depth_frame failed: {e}")
                    head_depth = None
            # feed frame to apriltag head tracker
            if head_tracker is not None and head_img is not None and head_img.bgr is not None:
                head_tracker.update_frame(head_img.bgr)

            # get surrounding camera image
            surround_img = surround_cam.get_frame()
            surround_depth = surround_cam.get_depth_frame()
            # feed frame to apriltag surround tracker
            if surround_img is not None and surround_img.bgr is not None:
                surround_tracker.update_frame(surround_img.bgr)

            #if camera_config['left_wrist_camera']['enable_zmq']:
             #   if args.record:
              #      left_wrist_img = img_client.get_left_wrist_frame()
            #if camera_config['right_wrist_camera']['enable_zmq']:
             #   if args.record:
              #      right_wrist_img = img_client.get_right_wrist_frame()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    # snapshot measured frequencies into robot metadata
                    metadata['robot_config']['control_frequency'] = arm_ctrl.measured_control_hz
                    metadata['robot_config']['state_frequency'] = arm_ctrl.measured_state_hz
                    # snapshot AprilTag extrinsic into camera metadata
                    if head_tracker is not None:
                        pose, detected = head_tracker.get_pose()
                        if pose is not None:
                            metadata['camera_config']['cameras'][0]['extrinsics'] = pose.tolist()
                    surround_pose, surround_detected = surround_tracker.get_pose()
                    if surround_pose is not None:
                        metadata['camera_config']['cameras'][1]['extrinsics'] = surround_pose.tolist()
                    recorder.info.update(metadata)
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                        if args.input_mode == "waldo":
                            arm_ctrl.start()
                            if args.ee == "brainco":
                                hand_ctrl.start()
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    if args.input_mode == "waldo":
                        if args.ee == "brainco":
                            hand_ctrl.stop()
                        arm_ctrl.stop()
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)


            #begin waldogate
            if args.input_mode != "waldo":
                # get xr's tele data
                tele_data = tv_wrapper.get_tele_data()
                logger_mp.info(f"L_wrist: {tele_data.left_wrist_pose[:3,3].round(3)}  R_wrist: {tele_data.right_wrist_pose[:3,3].round(3)}  R_hand_nz: {np.count_nonzero(tele_data.right_hand_pos)}")
                if (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    if left_hand_pos_array is not None:
                        with left_hand_pos_array.get_lock():
                            left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                    with right_hand_pos_array.get_lock():
                        right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = tele_data.left_ctrl_triggerValue
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = tele_data.right_ctrl_triggerValue
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = tele_data.left_hand_pinchValue
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = tele_data.right_hand_pinchValue
                else:
                    pass
            else:
                # Waldo hands: no main-loop work needed. Waldo_Brainco_Controller runs its own
                # ZMQ subscribe -> DDS publish loop in background threads, and updates the
                # recording arrays (dual_hand_state_array / dual_hand_action_array) internally.
                pass
            #end waldogate

            # will be skipped if its waldo.
            # high level control
            if args.input_mode == "controller" and args.motion:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                  -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                  -tele_data.right_ctrl_thumbstickValue[0]* 0.3)
            #begin waldogate
            if args.input_mode != "waldo":
                # get current robot state data.
                current_lr_arm_q       = arm_ctrl.get_current_dual_arm_q()
                current_lr_arm_dq      = arm_ctrl.get_current_dual_arm_dq()
                current_lr_arm_ddq     = arm_ctrl.get_current_dual_arm_ddq()
                current_lr_arm_tau_est = arm_ctrl.get_current_dual_arm_tau_est()

                # solve ik using motor data and wrist pose, then use ik results to control arms.
                time_ik_start = time.time()
                sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
                time_ik_end = time.time()
                logger_mp.info(f"ik:{round(time_ik_end - time_ik_start, 4)}s  sol_q_elbows:[{sol_q[3]:.3f},{sol_q[10]:.3f}]  cur_q_elbows:[{current_lr_arm_q[3]:.3f},{current_lr_arm_q[10]:.3f}]  vel_limit:{arm_ctrl.arm_velocity_limit:.1f}")
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            else:
                # Waldo arms: no main-loop work needed. Waldo_Arm_Controller runs its own
                # ZMQ subscribe -> DDS publish loop in background threads.
                # Read state for recording variables used below.
                current_lr_arm_q       = arm_ctrl.get_current_dual_arm_q()
                current_lr_arm_dq      = arm_ctrl.get_current_dual_arm_dq()
                current_lr_arm_ddq     = arm_ctrl.get_current_dual_arm_ddq()
                current_lr_arm_tau_est = arm_ctrl.get_current_dual_arm_tau_est()
                current_lr_arm_ext_tau  = arm_ctrl.get_current_dual_arm_external_tau()
                current_lr_arm_comp_tau = arm_ctrl.get_current_dual_arm_compensation_tau()
                sol_q = arm_ctrl.get_arm_action()
                sol_dq = arm_ctrl.get_arm_action_velocity()
                sol_ddq = arm_ctrl.get_arm_action_acceleration()
                # cartesian positions via FK
                state_l_pos, state_l_aa, state_r_pos, state_r_aa = arm_ctrl.get_current_dual_arm_cartesian_pos()
                action_l_pos, action_l_aa, action_r_pos, action_r_aa = arm_ctrl.get_arm_action_cartesian_pos()
                # cartesian velocities via Jacobian
                state_l_twist, state_r_twist = arm_ctrl.get_current_dual_arm_cartesian_vel()
                action_l_twist, action_r_twist = arm_ctrl.get_arm_action_cartesian_vel()
                # cartesian accelerations via FK
                state_l_accel, state_r_accel = arm_ctrl.get_current_dual_arm_cartesian_accel()
                action_l_accel, action_r_accel = arm_ctrl.get_arm_action_cartesian_accel()
                # cartesian external wrench (follower only)
                state_l_ext_wrench, state_r_ext_wrench = arm_ctrl.get_current_dual_arm_cartesian_external_wrench()
                # motor temperatures (follower only)
                arm_temperatures = arm_ctrl.get_current_dual_arm_temperature()
                # output IDs and timestamps
                follower_tick = arm_ctrl.get_current_tick()
                follower_timestamp = time.time()
                leader_frame_count = arm_ctrl.get_action_frame_count()
                leader_timestamp = arm_ctrl.get_arm_action_timestamp()
            #end waldogate
            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                #begin waldogate
                elif args.input_mode == "waldo":
                    # TODO: Waldo recording
                    # Non-waldo reads end-effector state/action from shared memory arrays populated by
                    # the hand controller child processes. Each ee type has different joint counts:
                    #
                    # End-effector joint specifications (for reference):
                    #   dex3:        7 joints per hand (14 total) - thumb(3) + index(2) + middle(2)
                    #   dex1:        1 joint per hand (2 total) - single gripper motor
                    #   inspire_dfx: 6 joints per hand (12 total) - fingers(4) + thumb_bend(1) + thumb_rot(1)
                    #   inspire_ftp: 6 joints per hand (12 total) - same as dfx, different protocol
                    #   brainco:     6 joints per hand (12 total) - thumb(2) + fingers(4)
                    #
                    # Arm joint specifications:
                    #   G1_29/H1_2: 7 joints per arm (14 total) - shoulder(3) + elbow(1) + wrist(3)
                    #   G1_23:      5 joints per arm (10 total) - shoulder(3) + elbow(1) + wrist_roll(1)
                    #   H1:         4 joints per arm (8 total) - shoulder(3) + elbow(1)
                    #
                    # Waldo should populate these from your own controllers:
                    #   left_ee_state    - current left hand/gripper joint positions (list, length = ee joint count)
                    #   right_ee_state   - current right hand/gripper joint positions (list, length = ee joint count)
                    #   left_hand_action  - target left hand/gripper joint commands sent this frame (list, same length)
                    #   right_hand_action - target right hand/gripper joint commands sent this frame (list, same length)
                    #   left_arm_state   - current left arm joint positions (list, length = arm joint count)
                    #   right_arm_state  - current right arm joint positions (list, length = arm joint count)
                    #   left_arm_action  - target left arm joint commands sent this frame (list, same length)
                    #   right_arm_action - target right arm joint commands sent this frame (list, same length)
                    #   current_body_state  - full body motor positions (list, [] if unused)
                    #   current_body_action - locomotion commands (list, [] if unused)
                    # hand ee state/action from Waldo_Brainco_Controller's recording arrays
                    if args.ee == "brainco":
                        with dual_hand_data_lock:
                            left_ee_state = dual_hand_state_array[:6]
                            right_ee_state = dual_hand_state_array[-6:]
                            left_hand_action = dual_hand_action_array[:6]
                            right_hand_action = dual_hand_action_array[-6:]
                            left_ee_dq = list(dual_hand_dq_array[:6])
                            right_ee_dq = list(dual_hand_dq_array[-6:])
                            left_ee_tau = list(dual_hand_tau_array[:6])
                            right_ee_tau = list(dual_hand_tau_array[-6:])

                    current_body_state = []
                    current_body_action = []
                    left_arm_state = current_lr_arm_q[:7]
                    right_arm_state = current_lr_arm_q[-7:]
                    left_arm_dq      = current_lr_arm_dq[:7]
                    right_arm_dq     = current_lr_arm_dq[-7:]
                    left_arm_ddq     = current_lr_arm_ddq[:7]
                    right_arm_ddq    = current_lr_arm_ddq[-7:]
                    left_arm_tau_est      = current_lr_arm_tau_est[:7]
                    right_arm_tau_est     = current_lr_arm_tau_est[-7:]
                    left_arm_ext_tau       = current_lr_arm_ext_tau[:7]
                    right_arm_ext_tau      = current_lr_arm_ext_tau[-7:]
                    left_arm_comp_tau      = current_lr_arm_comp_tau[:7]
                    right_arm_comp_tau     = current_lr_arm_comp_tau[-7:]
                    left_arm_action = sol_q[:7]
                    right_arm_action = sol_q[-7:]
                    left_arm_action_dq   = sol_dq[:7]
                    right_arm_action_dq  = sol_dq[-7:]
                    left_arm_action_ddq  = sol_ddq[:7]
                    right_arm_action_ddq = sol_ddq[-7:]
                    sol_tauff = arm_ctrl.get_arm_tauff()
                    left_arm_action_tauff  = sol_tauff[:7]
                    right_arm_action_tauff = sol_tauff[-7:]
                #end waldogate
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []
                    left_arm_state  = current_lr_arm_q[:7]
                    right_arm_state = current_lr_arm_q[-7:]
                    left_arm_dq      = current_lr_arm_dq[:7]
                    right_arm_dq     = current_lr_arm_dq[-7:]
                    left_arm_ddq     = current_lr_arm_ddq[:7]
                    right_arm_ddq    = current_lr_arm_ddq[-7:]
                    left_arm_tau_est  = current_lr_arm_tau_est[:7]
                    right_arm_tau_est = current_lr_arm_tau_est[-7:]
                    left_arm_ext_tau       = []
                    right_arm_ext_tau      = []
                    left_arm_comp_tau      = []
                    right_arm_comp_tau     = []
                    left_arm_action = sol_q[:7]
                    right_arm_action = sol_q[-7:]
                    left_arm_action_dq     = []
                    right_arm_action_dq    = []
                    left_arm_action_ddq    = []
                    right_arm_action_ddq   = []
                    left_arm_action_tauff  = sol_tauff[:7]
                    right_arm_action_tauff = sol_tauff[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera'].get('enable_depth') and head_depth is not None:
                        # raw uint16 depth (lossless via .png)
                        depths['depth_0'] = head_depth
                        # colorized RGB for visualization
                        depth_clipped = np.clip(head_depth, 0, 3000)
                        depth_norm = (depth_clipped * (255.0 / 3000)).astype(np.uint8)
                        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
                        depth_colored[head_depth == 0] = 0
                        depths['depth_0_rgb'] = depth_colored
                    if head_img is not None and head_img.bgr is not None:
                        colors[f"color_{0}"] = head_img.bgr
                    else:
                        logger_mp.warning("Head image is None!")
                    # surrounding camera
                    if surround_img is not None and surround_img.bgr is not None:
                        colors["color_1"] = surround_img.bgr
                    if surround_depth is not None:
                        depths['depth_1'] = surround_depth
                        # D405: depth_scale≈0.0001, 1m = 10000 raw units
                        depth_clipped_s = np.clip(surround_depth, 0, 30000)
                        depth_norm_s = (depth_clipped_s * (255.0 / 30000)).astype(np.uint8)
                        depth_colored_s = cv2.applyColorMap(depth_norm_s, cv2.COLORMAP_TURBO)
                        depth_colored_s[surround_depth == 0] = 0
                        depths['depth_1_rgb'] = depth_colored_s
                    states = {
                        "left_arm": {
                            "qpos":               left_arm_state.tolist(),
                            "qvel":               left_arm_dq.tolist(),
                            "torque":             left_arm_tau_est.tolist(),
                            "ddq":                left_arm_ddq.tolist(),
                            "external_torque":    left_arm_ext_tau.tolist(),
                            "compensation_torque": left_arm_comp_tau.tolist(),
                            "cartesian_pos":  state_l_pos.tolist(),
                            "cartesian_axis_angle": state_l_aa.tolist(),
                            "cartesian_vel": state_l_twist.tolist(),
                            "cartesian_accel": state_l_accel.tolist(),
                            "cartesian_external_wrench": state_l_ext_wrench.tolist(),
                            "rotor_temperature": arm_temperatures[:7],
                        },
                        "right_arm": {
                            "qpos":               right_arm_state.tolist(),
                            "qvel":               right_arm_dq.tolist(),
                            "torque":             right_arm_tau_est.tolist(),
                            "ddq":                right_arm_ddq.tolist(),
                            "external_torque":    right_arm_ext_tau.tolist(),
                            "compensation_torque": right_arm_comp_tau.tolist(),
                            "cartesian_pos":  state_r_pos.tolist(),
                            "cartesian_axis_angle": state_r_aa.tolist(),
                            "cartesian_vel": state_r_twist.tolist(),
                            "cartesian_accel": state_r_accel.tolist(),
                            "cartesian_external_wrench": state_r_ext_wrench.tolist(),
                            "rotor_temperature": arm_temperatures[7:],
                        },
                        "left_ee": {
                            "qpos":   left_ee_state,
                            "qvel":   left_ee_dq if args.input_mode == "waldo" and args.ee == "brainco" else [],
                            "torque": left_ee_tau if args.input_mode == "waldo" and args.ee == "brainco" else [],
                        },
                        "right_ee": {
                            "qpos":   right_ee_state,
                            "qvel":   right_ee_dq if args.input_mode == "waldo" and args.ee == "brainco" else [],
                            "torque": right_ee_tau if args.input_mode == "waldo" and args.ee == "brainco" else [],
                        },
                        "body": {
                            "qpos": current_body_state,
                        },
                        "head": {},
                        "output_id": follower_tick,
                        "output_timestamp": follower_timestamp,
                    }
                    # apriltag head pose
                    if head_tracker is not None:
                        head_pose, head_detected = head_tracker.get_pose()
                    else:
                        head_pose, head_detected = None, False
                    if head_pose is not None:
                        states["head"] = {
                            "position": head_pose[:3, 3].tolist(),
                            "rotation": head_pose[:3, :3].tolist(),
                            "pose_matrix": head_pose.tolist(),
                            "detected": head_detected,
                        }
                    else:
                        states["head"] = {
                            "position": [0.0, 0.0, 0.0],
                            "rotation": [[0.0]*3]*3,
                            "pose_matrix": [[0.0]*4]*4,
                            "detected": False,
                        }
                    # apriltag surround pose
                    surround_pose, surround_detected = surround_tracker.get_pose()
                    if surround_pose is not None:
                        states["surround"] = {
                            "position": surround_pose[:3, 3].tolist(),
                            "rotation": surround_pose[:3, :3].tolist(),
                            "pose_matrix": surround_pose.tolist(),
                            "detected": surround_detected,
                        }
                    else:
                        states["surround"] = {
                            "position": [0.0, 0.0, 0.0],
                            "rotation": [[0.0]*3]*3,
                            "pose_matrix": [[0.0]*4]*4,
                            "detected": False,
                        }
                    actions = {
                        "left_arm": {
                            "qpos":               left_arm_action.tolist(),
                            "qvel":               left_arm_action_dq.tolist() if hasattr(left_arm_action_dq, 'tolist') else left_arm_action_dq,
                            "torque":             left_arm_action_tauff.tolist(),
                            "ddq":                left_arm_action_ddq.tolist() if hasattr(left_arm_action_ddq, 'tolist') else left_arm_action_ddq,
                            "compensation_torque": left_arm_action_tauff.tolist(),
                            "cartesian_pos":  action_l_pos.tolist(),
                            "cartesian_axis_angle": action_l_aa.tolist(),
                            "cartesian_vel": action_l_twist.tolist(),
                            "cartesian_accel": action_l_accel.tolist(),
                        },
                        "right_arm": {
                            "qpos":               right_arm_action.tolist(),
                            "qvel":               right_arm_action_dq.tolist() if hasattr(right_arm_action_dq, 'tolist') else right_arm_action_dq,
                            "torque":             right_arm_action_tauff.tolist(),
                            "ddq":                right_arm_action_ddq.tolist() if hasattr(right_arm_action_ddq, 'tolist') else right_arm_action_ddq,
                            "compensation_torque": right_arm_action_tauff.tolist(),
                            "cartesian_pos":  action_r_pos.tolist(),
                            "cartesian_axis_angle": action_r_aa.tolist(),
                            "cartesian_vel": action_r_twist.tolist(),
                            "cartesian_accel": action_r_accel.tolist(),
                        },
                        "left_ee": {
                            "qpos":   left_hand_action,
                            "qvel":   [],
                            "torque": [],
                        },
                        "right_ee": {
                            "qpos":   right_hand_action,
                            "qvel":   [],
                            "torque": [],
                        },
                        "body": {
                            "qpos": current_body_action,
                        },
                        "output_id": leader_frame_count,
                        "output_timestamp": leader_timestamp,
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        #begin waldogate
        try:
            if args.input_mode != "waldo":
                arm_ctrl.ctrl_dual_arm_go_home()
            else:
                # Waldo shutdown: return arms home, stop background threads
                arm_ctrl.ctrl_dual_arm_go_home()
                arm_ctrl.stop()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        #end waldogate
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if head_tracker is not None:
                head_tracker.stop()
        except Exception as e:
            logger_mp.error(f"Failed to stop head tracker: {e}")

        try:
            surround_tracker.stop()
        except Exception as e:
            logger_mp.error(f"Failed to stop surround tracker: {e}")

        try:
            surround_cam.close()
        except Exception as e:
            logger_mp.error(f"Failed to close surround camera: {e}")

        try:
            img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        #begin waldogate
        if args.input_mode != "waldo":
            try:
                tv_wrapper.close()
            except Exception as e:
                logger_mp.error(f"Failed to close televuer wrapper: {e}")
        #end waldogate

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
