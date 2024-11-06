from typing import Tuple
from absl import app
import os
import functools

import jax
import jax.numpy as jnp

import mujoco
from mujoco import mjx
from brax.mjx import pipeline

from src.controllers.osc import utilities as osc_utils
from src.controllers.osc import controller
from src.controllers.osc.controller import OSQPConfig

# Types:
from jaxopt.base import KKTSolution
from brax.mjx.base import State
from src.controllers.osc.controller import OptimizationData

import time

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

jax.config.update('jax_enable_x64', True)
# jax.config.update('jax_compiler_enable_remat_pass', False)
# jax.config.update('jax_disable_jit', True)

def main(argv):
    xml_path = os.path.join(
        os.path.dirname(__file__),
        'models/unitree_go2/scene_mjx_torque.xml',
    )
    # Mujoco model:
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    q_init = jnp.asarray(mj_model.keyframe('home').qpos)
    qd_init = jnp.asarray(mj_model.keyframe('home').qvel)
    default_ctrl = jnp.asarray(mj_model.keyframe('home').ctrl)

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
    feet_ids = jnp.asarray(feet_site_ids)

    imu_id = jnp.asarray(
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'imu'),
    )

    base_body = [
        'base_link',
    ]
    base_body_id = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY.value, b)
        for b in base_body
    ]
    base_id = jnp.asarray(base_body_id)

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
    calf_ids = jnp.asarray(calf_body_ids)

    # MJX Model:
    model = mjx.put_model(mj_model)
    data = mjx.make_data(model)

    # Initialize MJX Data:
    data = data.replace(
        qpos=q_init, qvel=qd_init, ctrl=jnp.zeros_like(default_ctrl),
    )
    data = mjx.forward(model, data)

    # JIT Functions:
    env_init = functools.partial(
        pipeline.init,
        act=None,
        ctrl=None,
        unused_debug=False,
    )
    env_step = functools.partial(
        pipeline.step,
        unused_debug=False,
    )
    init_fn = jax.jit(jax.vmap(env_init, in_axes=(None, 0, 0)))
    step_fn = jax.jit(jax.vmap(env_step, in_axes=(None, 0, 0)))

    # Number of Parallel Envs:
    num_envs = 1024
    batch_size = 256
    num_minibatches = 4

    # Initialize OSC Controller:
    taskspace_targets = jnp.zeros((num_envs, 5, 6))
    osc_controller = controller.OSCController(
        model=mj_model,
        num_contacts=4,
        num_taskspace_targets=5,
        use_motor_model=False,
        osqp_config=OSQPConfig(
            check_primal_dual_infeasability=False,
            tol=1e-3,
            maxiter=4000,
            verbose=False,
            jit=True,
        ),
    )

    # JIT Controller Functions:
    get_data_fn = jax.jit(jax.vmap(osc_utils.get_data, in_axes=(None, 0, 0, None)))
    update_fn = jax.jit(jax.vmap(osc_controller.update, in_axes=(0, 0)))
    warmstart_fn = jax.jit(osc_controller.initialize_warmstart)
    solve_fn = jax.jit(osc_controller.solve)

    # Initialize State:
    q_init = jnp.tile(q_init, (num_envs, 1))
    qd_init = jnp.tile(qd_init, (num_envs, 1))
    state = init_fn(model, q_init, qd_init)

    # Initialize Values and Warmstart:
    num_steps = 1000
    body_points = jnp.expand_dims(state.site_xpos[:, imu_id], axis=1)
    feet_points = state.site_xpos[:, feet_ids]
    points = jnp.concatenate([body_points, feet_points], axis=1)
    body_ids = jnp.concatenate([base_id, calf_ids])
    osc_data = get_data_fn(model, state, points, body_ids)
    prog_data = update_fn(taskspace_targets, osc_data)

    weight = jnp.linalg.norm(model.opt.gravity) * jnp.sum(model.body_mass)
    init_x = jnp.concatenate([
        jnp.zeros(osc_controller.dv_size),
        default_ctrl,
        jnp.array([0, 0, weight / 4] * 4),
    ])
    init_x = jnp.tile(init_x, (num_envs, 1))

    # Batched Warmstart:
    def batch_warmstart(x: jax.Array, data: OptimizationData) -> KKTSolution:
        def minibatch(carry: None, xs: Tuple[jax.Array, OptimizationData]) -> Tuple[None, KKTSolution]:
            x, data = xs
            warmstart = warmstart_fn(x, data)
            return None, warmstart
        
        # Split into Batches:
        x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
        data = jax.tree.map(lambda x: jnp.reshape(x, (num_minibatches, -1) + x.shape[1:]), data)

        _, warmstart = jax.lax.scan(
            f=minibatch,
            init=None,
            xs=(x, data),
        )

        warmstart = jax.tree.map(lambda x: jnp.concatenate(x, axis=0), warmstart)
        return warmstart
    
    warmstart = jax.jit(batch_warmstart)(init_x, prog_data)
    

    num_control_steps = 10
    def loop(carry, xs):
        def batched_solve(carry, xs):
            data, warmstart = xs
            solution = solve_fn(data, warmstart)

            primal = jnp.reshape(solution.params.primal[0], (batch_size, -1))
            dv, u, z = jnp.split(primal, [osc_controller.dv_idx, osc_controller.u_idx], axis=-1)
            warmstart = solution.params

            return None, (u, warmstart)

        def step_unroll(carry, xs):
            state, ctrl = carry
            next_state = step_fn(model, state, ctrl)
            return (next_state, ctrl), None

        def batched_step(carry, xs):
            state, warmstart = carry

            # Calculate Control:
            body_points = jnp.expand_dims(state.site_xpos[:, imu_id], axis=1)
            feet_points = state.site_xpos[:, feet_ids]
            points = jnp.concatenate([body_points, feet_points], axis=1)
            osc_data = get_data_fn(model, state, points, body_ids)
            prog_data = update_fn(taskspace_targets, osc_data)

            # Split into Batches:
            batched_data = jax.tree.map(lambda x: jnp.reshape(x, (num_minibatches, -1) + x.shape[1:]), prog_data)
            batched_warmstart = jax.tree.map(lambda x: jnp.reshape(x, (num_minibatches, -1) + x.shape[1:]), warmstart)
            
            # Solve:
            _, (u, next_warmstart) = jax.lax.scan(
                f=batched_solve,
                init=None,
                xs=(batched_data, batched_warmstart),
            )

            # Format Control and Warmstart:
            u = jnp.concatenate(u, axis=0)
            next_warmstart = jax.tree.map(lambda x: jnp.concatenate(x, axis=0), next_warmstart)

            (next_state, _), _ = jax.lax.scan(
                f=step_unroll,
                init=(state, u),
                xs=None,
                length=num_control_steps,
            )

            return (next_state, next_warmstart), (next_state, next_warmstart)
        
        # Unpack Carry:
        state, warmstart = carry

        # Run Batched Step:
        (next_state, next_warmstart), _ = jax.lax.scan(
            f=batched_step,
            init=(state, warmstart),
            xs=None,
            length=batch_size * num_minibatches // num_envs,
        )

        return (next_state, next_warmstart), _

    # Run Loop:
    initial_state = init_fn(model, q_init, qd_init)
    key = jax.random.key(0)
    start_time = time.time()
    (final_state, _), _ = jax.lax.scan(
        f=loop,
        init=(initial_state, warmstart),
        xs=None,
        length=num_steps,
    )
    final_state.q.block_until_ready()
    print(f"Time Elapsed: {time.time() - start_time}")


if __name__ == '__main__':
    app.run(main)
