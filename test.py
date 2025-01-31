from absl import app
import os

import numpy as np

import mujoco


def main(argv):
    xml_path = os.path.join(
        os.path.dirname(__file__),
        'models/unitree_go2/scene_mjx_torque.xml',
    )
    # Mujoco model:
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    q_init = np.asarray(mj_model.keyframe('home').qpos)
    qd_init = np.asarray(mj_model.keyframe('home').qvel)
    default_ctrl = np.asarray(mj_model.keyframe('home').ctrl)

    feet_site = [
        'front_left_foot',
        'front_right_foot',
        'hind_left_foot',
        'hind_right_foot',
    ]
    feet_site_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
        for f in feet_site
    ]
    feet_ids = np.asarray(feet_site_ids)

    imu_id = np.asarray(
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'imu'),
    )

    base_body = [
        'base_link',
    ]
    base_body_id = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY.value, b)
        for b in base_body
    ]
    base_id = np.asarray(base_body_id)

    calf_body = [
            'front_left_calf',
            'front_right_calf',
            'hind_left_calf',
            'hind_right_calf',
        ]
    calf_body_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY.value, c)
        for c in calf_body
    ]
    calf_ids = np.asarray(calf_body_ids)

    # Setup Data:
    data = mujoco.MjData(mj_model)
    data.qpos = q_init
    data.qvel = qd_init
    data.ctrl = default_ctrl

    # Minimal Steps:
    mujoco.mj_kinematics(mj_model, data)
    mujoco.mj_comPos(mj_model, data)
    mujoco.mj_comVel(mj_model, data)
    
    jacp = np.zeros((1,))
    jacr = np.zeros((1,))

    mj_jacobian = functools.partial(mujoco.mj_jac, model=mj_model, data=data)
    np.vectorize(mj_jacobian)(jacp, jacr, points, body_ids)

if __name__ == '__main__':
    app.run(main)
