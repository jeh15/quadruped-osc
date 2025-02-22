from typing import Optional, Union, Callable, Any

import jax
import jax.numpy as jnp

from flax import struct, serialization

from jaxopt import BoxOSQP
from jaxopt.base import OptStep

import mujoco
from brax.base import System
from mujoco.mjx._src.types import Model

from src.controllers.osc.utilities import OSCData


@struct.dataclass
class WeightConfig:
    # Task Space Tracking Weights:
    fl_translational_tracking: float = 10.0
    fl_rotational_tracking: float = 1.0
    fr_translational_tracking: float = 10.0
    fr_rotational_tracking: float = 1.0
    hl_translational_tracking: float = 10.0
    hl_rotational_tracking: float = 1.0
    hr_translational_tracking: float = 10.0
    hr_rotational_tracking: float = 1.0
    # Torque Minization Weight:
    torque: float = 1e-4
    # Regularization Weight:
    regularization: float = 1e-4


@struct.dataclass
class OSQPConfig:
    check_primal_dual_infeasability: str | bool = True
    sigma: float = 1e-6
    momentum: float = 1.6
    eq_qp_solve: str = 'cg'
    rho_start: float = 0.1
    rho_min: float = 1e-6
    rho_max: float = 1e6
    stepsize_updates_frequency: int = 10
    primal_infeasible_tol: float = 1e-3
    dual_infeasible_tol: float = 1e-3
    maxiter: int = 4000
    tol: float = 1e-3
    termination_check_frequency: int = 5
    verbose: Union[bool, int] = 1
    implicit_diff: bool = True
    implicit_diff_solve: Optional[Callable] = None
    jit: bool = True
    unroll: str | bool = "auto"


@struct.dataclass
class OptimizationData:
    H: jax.Array
    f: jax.Array
    A: jax.Array
    lb: jax.Array
    ub: jax.Array


class OSCController:
    def __init__(
        self,
        model: System | Model,
        num_taskspace_targets: int,
        weights_config: WeightConfig = WeightConfig(),
        osqp_config: OSQPConfig = OSQPConfig(),
        control_matrix: Optional[jax.Array] = None,
        use_motor_model: bool = False,
        foot_radius: float = 0.02,
        friction_coefficient: float = 0.8,
    ):
        """ Operational Space Control (OSC) Controller for Quadruped."""
        assert bool(use_motor_model and control_matrix) is False, (
            "If using the motor model, the control matrix must be None."
        )
        # Weight Configuration:
        self.weights_config: dict = serialization.to_state_dict(weights_config)

        # Design Variables:
        self.dv_size: int = model.nv
        self.u_size: int = model.nu

        # Design Vector:
        self.q: jax.Array = jnp.zeros(
            (self.dv_size + self.u_size,),
        )

        # Joint Acceleration Indices:
        self.dv_idx: int = self.dv_size

        # Control Input Indices:
        self.u_idx: int = self.dv_idx + self.u_size

        # Model Variables:
        # TODO(jeh15): Fix this... XML File is not accurate... 
        self.ctrl_limits: jax.Array = jnp.concatenate([
            jnp.expand_dims(
                jnp.array([-0.9472, -1.4, -2.6227] * 4), axis=1,
            ),
            jnp.expand_dims(
                jnp.array([0.9472, 2.5, -0.84776] * 4), axis=1,
            )],
            axis=1,
        )
        self.torque_limits: jax.Array = model.actuator_forcerange

        self.use_motor_model: bool = use_motor_model
        self.B: jax.Array = jnp.eye(self.u_size)
        self.default_ctrl: jax.Array = jnp.zeros((self.u_size,))
        if self.use_motor_model:
            # Actuation Model:
            actuator_mask = model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT
            trnid = model.actuator_trnid[actuator_mask, 0]
            self.actuator_q_id: jax.Array = model.jnt_qposadr[trnid]
            self.actuator_qd_id: jax.Array = model.jnt_dofadr[trnid]
            self.actuator_gear: jax.Array = model.actuator_gear
            self.actuator_gainprm: jax.Array = model.actuator_gainprm
            self.actuator_biasprm: jax.Array = model.actuator_biasprm
            self.default_ctrl: jax.Array = jnp.array(model.keyframe('home').ctrl)
        else:
            if control_matrix is not None:
                self.B: jax.Array = control_matrix

        assert self.B.shape == (model.nv, self.u_size), (
            "Control matrix shape should be size (NV, NU)."
        )

        # Utility Variables:
        self.num_targets: int = num_taskspace_targets

        # Initialize OSQP:
        self.osqp_config: OSQPConfig = osqp_config
        self.prog: BoxOSQP = BoxOSQP(
            check_primal_dual_infeasability=self.osqp_config.check_primal_dual_infeasability,
            sigma=self.osqp_config.sigma,
            momentum=self.osqp_config.momentum,
            eq_qp_solve=self.osqp_config.eq_qp_solve,
            rho_start=self.osqp_config.rho_start,
            rho_min=self.osqp_config.rho_min,
            rho_max=self.osqp_config.rho_max,
            stepsize_updates_frequency=self.osqp_config.stepsize_updates_frequency,
            primal_infeasible_tol=self.osqp_config.primal_infeasible_tol,
            dual_infeasible_tol=self.osqp_config.dual_infeasible_tol,
            maxiter=self.osqp_config.maxiter,
            tol=self.osqp_config.tol,
            termination_check_frequency=self.osqp_config.termination_check_frequency,
            verbose=self.osqp_config.verbose,
            implicit_diff=self.osqp_config.implicit_diff,
            implicit_diff_solve=self.osqp_config.implicit_diff_solve,
            jit=self.osqp_config.jit,
            unroll=self.osqp_config.unroll,
        )

        # Initialize Optimization Problem:
        self._initialize_optimization()

    def equality_constraints(self, q: jax.Array, data: OSCData) -> jax.Array:
        """Compute equality constraints for the dynamics of a system.

        Args:
            q (jax.Array): Design Variables.
            data (OSCData): Data for the OSC controller.

        Returns:
            jax.Array: Equality constraints.

            Dynamics:
            M @ dv + C - B @ u - J_contact.T @ z = 0

            # Taskspace Objective Slack Variable Formulation:
            ddx_task = J_task @ dv + task_bias

        """
        # Unpack Design Variables:
        dv, u = jnp.split(
            q, [self.dv_idx],
        )
        # Unpack Data:
        M = data.mass_matrix
        C = data.coriolis_matrix

        # Dynamics Constraint:
        if self.use_motor_model:
            u = self.motor_model(
                u,
                data.previous_q[self.actuator_q_id],
                data.previous_qd[self.actuator_qd_id],
            )

        dynamics = M @ dv + C - self.B @ u

        # Concatenate Constraints:
        equality_constraints = jnp.concatenate([dynamics])

        return equality_constraints

    def inequality_constraints(
        self, q: jax.Array,
    ) -> None:
        raise NotImplementedError

    def objective(
        self, q: jax.Array, desired_task_ddx: jax.Array, data: OSCData,
    ) -> jax.Array:
        """Compute the Task Space Tracking Objective.

        Args:
            q (jax.Array): Design Variables.
            desired_task_ddx (jax.Array): Desired Task Accelerations.
            data (OSCData): Data for the OSC controller.

        Returns:
            jax.Array: Objective Function.

        """
        # Unpack Design Variables:
        dv, u = jnp.split(
            q, [self.dv_idx],
        )

        # Task Space Objective:
        J_task = data.taskspace_jacobian
        task_bias = data.taskspace_bias
        ddx_taskspace = J_task @ dv + task_bias

        ddx_task = jnp.split(
            ddx_taskspace, self.num_targets, axis=0,
        )
        ddx_fl, ddx_fr, ddx_hl, ddx_hr = jax.tree.map(jnp.squeeze, ddx_task)

        ddx_target = jnp.split(
            desired_task_ddx, self.num_targets, axis=0,
        )
        ddx_fl_t, ddx_fr_t, ddx_hl_t, ddx_hr_t = jax.tree.map(
            jnp.squeeze, ddx_target,
        )

        # Objective Function:
        objective_terms = {
            'fl_translational_tracking': self._objective_tracking(
                ddx_fl[:3], ddx_fl_t[:3],
            ),
            'fl_rotational_tracking': self._objective_tracking(
                ddx_fl[3:], ddx_fl_t[3:],
            ),
            'fr_translational_tracking': self._objective_tracking(
                ddx_fr[:3], ddx_fr_t[:3],
                ),
            'fr_rotational_tracking': self._objective_tracking(
                ddx_fr[3:], ddx_fr_t[3:],
                ),
            'hl_translational_tracking': self._objective_tracking(
                ddx_hl[:3], ddx_hl_t[:3],
                ),
            'hl_rotational_tracking': self._objective_tracking(
                ddx_hl[3:], ddx_hl_t[3:],
                ),
            'hr_translational_tracking': self._objective_tracking(
                ddx_hr[:3], ddx_hr_t[:3],
                ),
            'hr_rotational_tracking': self._objective_tracking(
                ddx_hr[3:], ddx_hr_t[3:],
                ),
            'torque': self._objective_regularization(u),
            'regularization': self._objective_regularization(q),
        }

        objective_terms = {
            k: v * self.weights_config[k] for k, v in objective_terms.items()
        }
        objective_value = jnp.array(sum(objective_terms.values()))

        return objective_value

    def _objective_tracking(
        self, q: jax.Array, task_target: jax.Array,
    ) -> jax.Array:
        """Tracking Objective Function."""
        return jnp.sum(jnp.square(q - task_target))

    def _objective_regularization(
        self, q: jax.Array,
    ) -> jax.Array:
        """Regularization Objective Function."""
        return jnp.sum(jnp.square(q))

    def _initialize_optimization(self) -> None:
        """Initialize the Optimization Problem."""
        # Generate Optimization Functions: (jacrev -> row by row, jacfwd -> column by column)
        self.Aeq_fn = jax.jacfwd(self.equality_constraints)
        self.Aineq_fn = jax.jacfwd(self.inequality_constraints)
        self.H_fn = jax.jacfwd(jax.jacrev(self.objective))
        self.f_fn = jax.jacfwd(self.objective)

        # Constant Matricies:
        optimization_size = self.dv_size + self.u_size
        self.Abox = jnp.eye(N=optimization_size, M=optimization_size)
        # Joint Acceleration Bounds: dv = [qdd]
        dv_lb = -jnp.inf * jnp.ones((self.dv_size,))
        dv_ub = jnp.inf * jnp.ones((self.dv_size,))
        # Control Input Bounds: u = [tau_fl, tau_fr, tau_hl, tau_hr]
        if self.use_motor_model:
            u_lb = jnp.array(self.ctrl_limits[:, 0])
            u_ub = jnp.array(self.ctrl_limits[:, 1])
        else:
            u_lb = jnp.array(self.torque_limits[:, 0])
            u_ub = jnp.array(self.torque_limits[:, 1])
        self.box_lb = jnp.concatenate([dv_lb, u_lb])
        self.box_ub = jnp.concatenate([dv_ub, u_ub])

    def update(
        self, taskspace_targets: jax.Array, data: OSCData
    ) -> OptimizationData:
        """Update the Mathematical Program Matricies."""
        Aeq = self.Aeq_fn(self.q, data)
        beq = -self.equality_constraints(self.q, data)

        H = self.H_fn(self.q, taskspace_targets, data)
        f = self.f_fn(self.q, taskspace_targets, data)

        A = jnp.concatenate([Aeq, self.Abox])
        lb = jnp.concatenate([beq, self.box_lb])
        ub = jnp.concatenate([beq, self.box_ub])

        return OptimizationData(H=H, f=f, A=A, lb=lb, ub=ub)

    def solve(
        self, data: OptimizationData, warmstart: Optional[Any] = None,
    ) -> OptStep:
        """Solve using OSQP Solver."""
        solution = self.prog.run(
            init_params=warmstart,
            params_obj=(data.H, data.f),
            params_eq=data.A,
            params_ineq=(data.lb, data.ub),
        )
        return solution

    def motor_model(
        self, u: jax.Array, q: jax.Array, qd: jax.Array,
    ) -> jax.Array:
        """Brax Motor Model."""
        bias = self.actuator_gear[:, 0] * (
            self.actuator_biasprm[:, 1] * q + self.actuator_biasprm[:, 2] * qd
        )
        return self.actuator_gear[:, 0] * (self.actuator_gainprm[:, 0] * u + bias)
