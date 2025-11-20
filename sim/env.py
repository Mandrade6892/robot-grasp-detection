import time
import math
import os
import pybullet as p
import pybullet_data
import numpy as np
import imageio


def main(gui=True):
    # Start PyBullet
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(
    cameraDistance=1.0,   # zoom level
    cameraYaw=60,         # left/right rotation around target
    cameraPitch=-35,      # up/down tilt
    cameraTargetPosition=[0.5, 0, 0]  # point camera looks at
)


    # Set up the basic scene
    plane = p.loadURDF("plane.urdf")
    table = p.loadURDF("table/table.urdf", [0.5, 0, -0.65], p.getQuaternionFromEuler([0, 0, 0]))
    cube = p.loadURDF("cube_small.urdf", [0.5, 0, 0.02])

    # Load the Franka Panda robot arm
    robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

    # Get joint names and IDs so we can move the arm
    num_joints = p.getNumJoints(robot)
    joint_name_to_index = {p.getJointInfo(robot, i)[1].decode(): i for i in range(num_joints)}

    # Detect which joints belong to the fingers
    finger_joints = [i for i in range(num_joints) if "finger" in p.getJointInfo(robot, i)[1].decode()]

    # Function to open or close the gripper evenly
    def set_gripper(opening=0.04):
        pos_each = max(0.0, min(opening/2.0, 0.025))
        for j in finger_joints:
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=pos_each, force=20)

    # Move the arm so the hand is above the cube
    ee_link = joint_name_to_index.get("panda_hand", 11)
    target_pos = [0.5, 0.0, 0.25]
    target_quat = p.getQuaternionFromEuler([math.pi, 0, 0])
    ik = p.calculateInverseKinematics(robot, ee_link, target_pos, target_quat, maxNumIterations=200)

    # Apply the IK result to the arm joints
    arm_joint_indices = [joint_name_to_index[n] for n in [
        "panda_joint1","panda_joint2","panda_joint3","panda_joint4",
        "panda_joint5","panda_joint6","panda_joint7"
    ]]
    for idx, j in enumerate(arm_joint_indices):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=ik[idx], force=200)

    # Camera setup (used later for CNN input)
    cam_target = [0.5, 0, 0]
    cam_dist, cam_yaw, cam_pitch = 0.9, 60, -40
    width, height = 640, 480
    fov, near, far = 60, 0.01, 2.0
    aspect = width / height

    # Open the gripper first
    set_gripper(0.06)
    for _ in range(240):
        p.stepSimulation()
        time.sleep(1/240)

    # Slowly lower the arm toward the cube
    for step in range(180):
        z = 0.25 - 0.0005*step
        ik = p.calculateInverseKinematics(robot, ee_link, [0.5, 0.0, z], target_quat, maxNumIterations=100)
        for i, j in enumerate(arm_joint_indices):
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=ik[i], force=200)
        p.stepSimulation()
        time.sleep(1/240)

    # Close the gripper to grab the cube
    set_gripper(0.0)
    for _ in range(240):
        p.stepSimulation()
        time.sleep(1/240)

    # Lift the cube up a bit
    for step in range(240):
        z = 0.16 + 0.0005*step
        ik = p.calculateInverseKinematics(robot, ee_link, [0.5, 0.0, z], target_quat, maxNumIterations=100)
        for i, j in enumerate(arm_joint_indices):
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=ik[i], force=200)
        p.stepSimulation()
        time.sleep(1/240)

    # Take one camera snapshot (we'll later feed this into the CNN)
       # --- Camera mounted on the robot hand (eye-in-hand) ---

    # Get end-effector (hand) pose in world coordinates
    hand_state = p.getLinkState(robot, ee_link)
    hand_pos = hand_state[0]       # (x, y, z)
    hand_ori = hand_state[1]       # quaternion (x, y, z, w)

    # Convert orientation to rotation matrix
    rot = p.getMatrixFromQuaternion(hand_ori)
    rot = np.array(rot).reshape(3, 3)

    # Define camera "forward" and "up" directions relative to the hand
    # Here we use the hand's -Z as "forward" and Y as "up" (tweak if needed)
    forward = -rot[:, 2]  # third column, negated
    up_vec  =  rot[:, 1]  # second column

    # Camera eye position: a bit above the hand along its forward axis
    cam_eye = hand_pos + 0.10 * forward  # 10 cm in front of the hand
    cam_target = hand_pos                # look back at the hand / object

    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cam_eye,
        cameraTargetPosition=cam_target,
        cameraUpVector=up_vec
    )

    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=near,
        farVal=far
    )

    img_data = p.getCameraImage(
        width,
        height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    # Convert to numpy arrays
    rgba = np.reshape(img_data[2], (height, width, 4)).astype(np.uint8)
    rgb_image = rgba[:, :, :3]

    # Make sure folder exists
    os.makedirs("captures", exist_ok=True)

    # Build filename (timestamp so each run is unique)
    filename = os.path.join("captures", f"ee_cam_{int(time.time())}.png")
    imageio.imwrite(filename, rgb_image)

    print(f"Saved hand-mounted camera snapshot to: {filename}")

    
        # Optional: visualize or save snapshot
        #import matplotlib.pyplot as plt
        # plt.imshow(rgb_image)
        # plt.show()
    

    if gui:
        print("Simulation complete. Close the window to finish.")
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1/240)
    p.disconnect()

if __name__ == "__main__":
    main(gui=True)
