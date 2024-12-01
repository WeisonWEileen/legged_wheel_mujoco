import copy
import mujoco
from dm_control import mjcf
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as Rotater
import common
from dm_control.mujoco import Physics
from lxml import etree
from dm_control.suite import base
from dm_control.rl import control
from dm_control import viewer
import mediapy as media
import matplotlib.pyplot as plt
import json
import time

# import mujoco_viewer
import mujoco.viewer
from utils.utils import *

np.set_printoptions(precision=5, suppress=True)

def compute_gains(model, data, configuration, Q, R, print_flag=False):
    # 1. Set configuration and find control that stabilizes it (ctrl0)
    newdata = mujoco.MjData(model)
    newdata = copy.copy(data)

    mujoco.mj_resetData(model, newdata)
    newdata.qpos = configuration
    # compute the control
    mujoco.mj_forward(model, newdata)
    newdata.qacc = 0
    mujoco.mj_inverse(model, newdata)

    # define control and configuration to linearize around
    # print(newdata.qfrc_actuator)
    qfrc_inverse_ = newdata.qfrc_inverse.copy()
    actuator_moment_inv_ = np.linalg.pinv(newdata.actuator_moment)
    ctrl0 = qfrc_inverse_ @ actuator_moment_inv_
    # ctrl0[:] = 0
    qpos0 = newdata.qpos.copy()
    if print_flag:
        print("qpos0:\n", qpos0)
        print("ctrl0:\n", ctrl0, "\nafrc_inv:\n", qfrc_inverse_)
        print("actuator_moment_inv:\n", actuator_moment_inv_)

    # 2. Compute LQR gains given weights
    mujoco.mj_resetData(model, newdata)
    newdata.ctrl = ctrl0
    newdata.qpos = qpos0

    # 3. Allocate the A and B matrices, compute them.
    A = np.zeros((2 * model.nv, 2 * model.nv))
    B = np.zeros((2 * model.nv, model.nu))
    epsilon = 1e-6
    flg_centered = True
    mujoco.mjd_transitionFD(model, newdata, epsilon, flg_centered, A, B, None, None)

    # print A, B, Q, R in format 'A:\n{}\n'
    if print_flag:
        print(
            "A:\n{}\nB:\n{}\nQ:\n{}\nR:\n{}".format(A.shape, B.shape, Q.shape, R.shape)
        )
        print("A:\n{}\nB:\n{}\nQ:\n{}\nR:\n{}".format(A, B, Q, R))
        # save np array into excel file
        save_A_excel(A, "./data/A.xlsx")
        save_B_excel(B, "./data/B.xlsx")

    # Solve discrete Riccati equation.
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    if print_flag:
        print("P:\n{}".format(P.shape))

    # Compute the feedback gain matrix K.
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    # print P, K in format 'A:\n{}\n'
    if print_flag:
        print("P:\n{}\nK:\n{}".format(P, K))
        save_B_excel(K.T, "./data/K.xlsx")

    return ctrl0, K


# Find the height-offset at which the vertical force is smallest.
def find_best_offset(model, data):
    data = copy.copy(data)
    height_offsets = np.linspace(-0.01, 0.01, 2001)
    pos_vertical0 = data.qpos[2]
    vertical_forces = []
    pos_list = []
    for offset in height_offsets:
        mujoco.mj_forward(model, data)
        data.qacc = 0
        # Offset the height by `offset`.
        pos_vertical_now = pos_vertical0 + offset
        data.qpos[2] = pos_vertical_now
        pos_list.append(data.qpos[2])
        mujoco.mj_inverse(model, data)
        vertical_forces.append(data.qfrc_inverse[2])
    pos_list = np.array(pos_list)
    print("pos_list:", pos_list)
    # Find the height-offset at which the vertical force is smallest.
    idx = np.argmin(np.abs(vertical_forces))
    best_offset = height_offsets[idx]
    # plot_smallest_vertical_force(height_offsets, vertical_forces, best_offset,model)
    return best_offset


def calc_delta_states(model, data, target_xyz):
    body_id = 5  # 1:root body, 5: left wheel
    # body_name = mujoco.mj_id2name(model, 1, body_id)
    print("body_id:", body_id)
    newdata = mujoco.MjData(model)
    newdata = copy.copy(data)

    mujoco.mj_resetData(model, newdata)
    newdata.qpos[:3] = target_xyz
    mujoco.mj_forward(model, newdata)

    xyz_0 = newdata.qpos.copy()[:3]
    jac_p = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))
    target_xyz_ = target_xyz.copy()
    mujoco.mj_jacSubtreeCom(model, newdata, jac_p, model.body("root").id)
    delta_xyz = target_xyz_ - xyz_0
    print(
        "target_xyz_:\n",
        target_xyz_,
        "\ndelta_xyz:\n",
        delta_xyz,
        "\njac_p:\n",
        jac_p,
        "\njac_r:\n",
        jac_r,
    )
    delta_state_p = np.linalg.pinv(jac_p) @ delta_xyz.reshape(-1, 1)
    print("delta_state_p:", delta_state_p.flatten())


def simulate():

    CART_POLE_MJCF = "legwheel_robot2.xml"  # "cartpole2.xml"
    sim_model = mujoco.MjModel.from_xml_path(CART_POLE_MJCF)
    sim_data = mujoco.MjData(sim_model)
    # renderer = mujoco.Renderer(sim_model, height=int(480 * 1.5), width=int(640 * 1.5))

    best_offset = find_best_offset(sim_model, sim_data)
    best_height = sim_data.qpos[2] + best_offset
    print(
        "best_height: {} + {} = {}]".format(sim_data.qpos[2], best_offset, best_height)
    )

    # Make a new camera, move it to a closer distance.
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(sim_model, camera)
    camera.lookat = [0, -0.3, 0.21]
    camera.azimuth = 90
    camera.elevation = -30
    camera.distance = 2
    q0 = copy.copy(sim_data.qpos)

    print(
        "sim_model.nv:",
        sim_model.nv,
        "sim_model.nu:",
        sim_model.nu,
        "sim_model.nq:",
        sim_model.nq,
        "sim_model.na:",
        sim_model.na,
    )
    print(
        "sim_data.qpos:",
        sim_data.qpos,
        "\nsim_data.qvel:",
        sim_data.qvel,
        "\nsim_data.qacc:",
        sim_data.qacc,
        "\nsim_data.ctrl:",
        sim_data.ctrl,
    )
    print(
        "sim_data.site_xpos:",
        sim_data.site_xpos,
        "\nsim_data.site_xmat:",
        sim_data.site_xmat,
    )

    # Parameters.
    DURATION = 10  # seconds
    BALANCE_STD = 0.2  # actuator units
    FRAMERATE = 30

    ang_euler = Rotater.from_quat([0, 0, 0, 1]).as_euler("xyz", degrees=True)
    ang_euler = [0, 0, 45]
    quat = Rotater.from_euler("xyz", ang_euler, degrees=True).as_quat()
    ang_tgt = [0, 0, -45]
    quat_tgt = Rotater.from_euler("xyz", ang_tgt, degrees=True).as_quat()
    print("ang_euler:", ang_euler, "quat:", quat)

    init_h = best_height
    tgt_h = init_h
    qpos0 = np.array(
        [
            0.0,
            0.0,
            init_h * np.cos(ang_euler[1] * np.pi / 180),
            quat[3],
            quat[0],
            quat[1],
            quat[2],
            0,
            0,
            0,
            0,
            0,
            0,
            0.0,
            0.0,
        ]
    )  # give initial pose
    target = np.array(
        [
            0.0,
            0.0,
            tgt_h,
            quat_tgt[3],
            quat_tgt[0],
            quat_tgt[1],
            quat_tgt[2],
            0.0,
            0.0,
            0,
            0,
            0,
            0,
            0.0,
            0.0,
        ]
    )  

    target[0] = 0.4  # x
    target[1] = -0.4  # y

    Q = np.eye(sim_model.nv * 2) * 1e-6
    if True:
        root_q_pos_idx = [0, 1, 2]
        root_q_ori_idx = [3, 4, 5]
        root_v_pos_idx = [14, 15, 16]
        root_v_ori_idx = [17, 18, 19]
        thigh_q_idx = [6, 10]
        thigh_v_idx = [20, 24]
        wheel_q_idx = [9, 13]
        wheel_v_idx = [23, 27]
        diagidx = root_q_pos_idx + root_q_ori_idx + root_v_pos_idx + root_v_ori_idx
    else:
        diagidx = list(range(sim_model.nv * 2))
    Q[diagidx, diagidx] = 10

    R = np.eye(sim_model.nu) * 1
    

    dq = np.zeros(sim_model.nv)

    # print(dq.shape)
    ctrl0, K = compute_gains(sim_model, sim_data, qpos0, Q, R)

    mujoco.mj_resetData(sim_model, sim_data)
    sim_data.qpos = qpos0

    # Enable contact force visualisation.
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    scene_option.frame = mujoco.mjtFrame.mjFRAME_WORLD

    # qhist = []
    # frames = []
    # dqlist = []
    # ctrlist = []
    # delta_ctrlist = []

    with mujoco.viewer.launch_passive(sim_model, sim_data) as viewer:
        # start_time = time.time()
        while True:
            step_start = time.time()
            # print("================={}===================".format(sim_data.time))
            if True:
                if sim_data.time % 0.1 < 0.001:
                    try:
                        ctrl0_, K_ = compute_gains(
                            sim_model, sim_data, sim_data.qpos, Q, R, False
                        )
                        ctrl0, K = ctrl0_, K_
                    except Exception as e:
                        print("Error:", e)
                        # continue

            mujoco.mj_differentiatePos(sim_model, dq, 1, target, sim_data.qpos)
            dx = np.hstack((dq, sim_data.qvel)).T

            # LQR control law.
            ctrl_noise = np.random.randn(sim_model.nu) / 5

            delta_ctrl = K @ dx
            sim_data.ctrl = ctrl0 - delta_ctrl  + ctrl_noise
            # print('ctrl:', sim_data.ctrl)
            # print("sim_data.time:", sim_data.time)

            mujoco.mj_step(sim_model, sim_data)

            # print("dq:\n",dq,"\nsim_data.qpos:\n",sim_data.qpos,"\nsim_data.qvel:\n",sim_data.qvel,"\nsim_data.ctrl:\n",sim_data.ctrl)

            viewer.sync()

            time_until_next_step = sim_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    simulate()
