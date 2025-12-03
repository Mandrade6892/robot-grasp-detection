import time
import math
import os
import pybullet as p
import pybullet_data
import numpy as np
import imageio


def main(gui=True):
    # Connect to PyBullet
    cid = p.connect(p.GUI if gui else p.DIRECT)
    print("Connected to PyBullet with id:", cid)

    try:
        # Basic setup
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Nice GUI camera
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=60,
            cameraPitch=-35,
            cameraTargetPosition=[0.5, 0, 0]
        )

        # Scene
        plane = p.loadURDF("plane.urdf")
        table = p.loadURDF(
            "table/table.urdf",
            [0.5, 0, -0.65],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        cube = p.loadURDF("cube_small.urdf", [0.5, 0, 0.02])

        # Robot
        robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        num_joints = p.getNumJoints(robot)
        joint_name_to_index = {
            p.getJointInfo(robot, i)[1].decode(): i for i in range(num_joints)
        }
        finger_joints = [
            i for i in range(num_joints)
            if "finger" in p.getJointInfo(robot, i)[1].decode()
        ]

        def set_gripper(opening=0.04):
            """Open/close gripper symmetrically."""
            pos_each = max(0.0, min(opening / 2.0, 0.025))
            for j in finger_joints:
                p.setJointMotorControl2(
                    robot,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=pos_each,
                    force=20
                )

        # Move hand above cube
        ee_link = joint_name_to_index.get("panda_hand", 11)
        target_pos = [0.5, 0.0, 0.25]
        target_quat = p.getQuaternionFromEuler([math.pi, 0, 0])

        ik = p.calculateInverseKinematics(
            robot,
            ee_link,
            target_pos,
            target_quat,
            maxNumIterations=200
        )

        arm_joint_indices = [joint_name_to_index[n] for n in [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7"
        ]]

        for idx, j in enumerate(arm_joint_indices):
            p.setJointMotorControl2(
                robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=ik[idx],
                force=200
            )

        # Camera parameters (for image capture)
        width, height = 640, 480
        fov, near, far = 60, 0.01, 2.0
        aspect = width / height

        print("Phase 1: open gripper")
        set_gripper(0.06)
        for _ in range(240):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        print("Phase 2: approach cube")
        for step in range(180):
            z = 0.25 - 0.0005 * step
            ik = p.calculateInverseKinematics(
                robot,
                ee_link,
                [0.5, 0.0, z],
                target_quat,
                maxNumIterations=100
            )
            for i, j in enumerate(arm_joint_indices):
                p.setJointMotorControl2(
                    robot,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=ik[i],
                    force=200
                )
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        print("Phase 3: close gripper")
        set_gripper(0.0)
        for _ in range(240):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        print("Phase 4: lift cube")
        for step in range(240):
            z = 0.16 + 0.0005 * step
            ik = p.calculateInverseKinematics(
                robot,
                ee_link,
                [0.5, 0.0, z],
                target_quat,
                maxNumIterations=100
            )
            for i, j in enumerate(arm_joint_indices):
                p.setJointMotorControl2(
                    robot,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=ik[i],
                    force=200
                )
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        # -------- Overhead "eagle view" camera + saving images --------
        print("Phase 5: capture image from overhead camera")

        # Camera above the table, looking straight down
        cam_eye = [0.5, 0.0, 1.0]      # (x, y, z) position of camera
        cam_target = [0.5, 0.0, 0.0]   # look at center of table
        cam_up = [0, 1, 0]             # y-axis is "up"

        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=cam_eye,
            cameraTargetPosition=cam_target,
            cameraUpVector=cam_up
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
            renderer=p.ER_TINY_RENDERER  # stable for RGB + depth
        )

        # Convert to numpy arrays
        rgba = np.reshape(img_data[2], (height, width, 4)).astype(np.uint8)
        rgb_image = rgba[:, :, :3]
        depth_buffer = np.reshape(img_data[3], (height, width))

        # Make sure folder exists
        os.makedirs("captures", exist_ok=True)

        # Save RGB image
        timestamp = int(time.time())
        rgb_path = os.path.join("captures", f"eagle_rgb_{timestamp}.png")
        imageio.imwrite(rgb_path, rgb_image)

        # Save depth map (normalized for viewing)
        depth_norm = (depth_buffer - depth_buffer.min()) / (
            depth_buffer.max() - depth_buffer.min() + 1e-8
        )
        depth_img = (depth_norm * 255).astype(np.uint8)
        depth_path = os.path.join("captures", f"eagle_depth_{timestamp}.png")
        imageio.imwrite(depth_path, depth_img)

        print("Saved overhead RGB to:", rgb_path)
        print("Saved overhead depth to:", depth_path)

        # Keep GUI alive
        if gui:
            print("Simulation complete. Close the window to finish.")
            while p.isConnected():
                p.stepSimulation()
                time.sleep(1.0 / 240.0)

    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main(gui=True)
