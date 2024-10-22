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
    base_translational_tracking: float = 1.0
    base_rotational_tracking: float = 1.0
    fl_translational_tracking: float = 1.0
    fl_rotational_tracking: float = 1.0
    fr_translational_tracking: float = 1.0
    fr_rotational_tracking: float = 1.0
    hl_translational_tracking: float = 1.0
    hl_rotational_tracking: float = 1.0
    hr_translational_tracking: float = 1.0
    hr_rotational_tracking: float = 1.0
    # Torque Minization Weight:
    torque: float = 1e-3
    # Regularization Weight:
    regularization: float = 1e-4


@struct.dataclass
class OSQPConfig:
    check_primal_dual_infeasability: str | bool = False
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
    verbose: Union[bool, int] = False
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
        dv_size: int,
        u_size: int,
        z_size: int,
        num_taskspace_targets: int,
        weights_config: WeightConfig = WeightConfig(),
        osqp_config: OSQPConfig = OSQPConfig(),
        control_matrix: Optional[jax.Array] = None,
        use_motor_model: bool = False,
        foot_radius: float = 0.02,
        friction_coefficient: float = 0.8,
    ):
        """ Operational Space Control (OSC) Controller for Quadruped."""

        # Weight Configuration:
        self.weights_config = serialization.to_state_dict(weights_config)

        # Design Variables:
        self.dv_size = dv_size
        self.u_size = u_size
        self.z_size = z_size

        # Design Vector:
        self.q = jnp.zeros((dv_size + u_size + z_size,))

        # Joint Acceleration Indices:
        self.dv_idx = self.dv_size

        # Control Input Indices:
        self.u_idx = self.dv_idx + self.u_size

        # Contact Force Indices:
        self.z_idx = self.u_idx + self.z_size

        # Slack Variable Indices:
        # self.slack_idx = self.z_idx + self.slack_size

        # Model Variables:
        self.r = foot_radius
        self.mu = friction_coefficient
        self.torque_limits = model.actuator_forcerange

        if use_motor_model:
            # Actuation Model:
            actuator_mask = model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT
            trnid = model.actuator_trnid[actuator_mask, 0]
            self.actuator_q_id = model.jnt_qposadr[trnid]
            self.actuator_qd_id = model.jnt_dofadr[trnid]
            self.actuator_gear = model.actuator_gear
            self.actuator_gainprm = model.actuator_gainprm
            self.actuator_biasprm = model.actuator_biasprm
            self.B = jnp.eye(self.u_size)
        else:
            # Control Matrix:
            if control_matrix is None:
                self.B = jnp.eye(self.u_size)
            else:
                self.B = control_matrix

        # Utility Variables:
        self.num_contacts = self.z_size // 6
        self.num_targets = num_taskspace_targets

        # Initialize OSQP:
        self.osqp_config = osqp_config
        self.prog = BoxOSQP(
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
        dv, u, z = jnp.split(
            q, [self.dv_idx, self.u_idx, self.z_idx],
        )
        # Unpack Data:
        M = data.mass_matrix
        C = data.coriolis_matrix
        J_contact = data.contact_jacobian

        # Dynamics Constraint:
        dynamics = M @ dv + C - self.B @ u - J_contact.T @ z

        # Concatenate Constraints:
        equality_constraints = jnp.concatenate([dynamics])

        return equality_constraints

    def inequality_constraints(
        self, q: jax.Array,
    ) -> jax.Array:
        """Compute inequality constraints for the system.

        Args:
            q (jax.Array): Design Variables.
            z_previous (jax.Array): Previous Slack Variables.

        Returns:
            jax.Array: Inequality constraints.

            # Friction Cone Constraints:
            # |f_x| + |f_y| <= mu * f_z

            # Torsional Friction Constraints:
            # |tau_z| <= r x (mu * f_z)

        """
        # Unpack Design Variables:
        dv, u, z = jnp.split(
            q, [self.dv_idx, self.u_idx, self.z_idx],
        )

        # Friction Constraints:
        # Translational: |f_x| + |f_y| <= mu * f_z
        # Torsional: |tau_z| <= r x (mu * f_z)

        def translational_friction(x: jax.Array) -> jax.Array:
            constraint_1 = x[0] + x[1] - self.mu * x[2]
            constraint_2 = -x[0] + x[1] - self.mu * x[2]
            constraint_3 = x[0] - x[1] - self.mu * x[2]
            constraint_4 = -x[0] - x[1] - self.mu * x[2]
            return jnp.array(
                [constraint_1, constraint_2, constraint_3, constraint_4],
            )

        def torsional_friction(x: jax.Array) -> jax.Array:
            constraint_1 = x[5] - self.r * self.mu * x[2]
            constraint_2 = -x[5] - self.r * self.mu * x[2]
            return jnp.array([constraint_1, constraint_2])

        # Split Contact Forces:
        contact_forces = jnp.reshape(z, (self.num_contacts, -1))

        translational_constraints = jax.vmap(
            translational_friction,
        )(contact_forces).flatten()
        torsional_constraints = jax.vmap(
            torsional_friction,
        )(contact_forces).flatten()

        return jnp.concatenate(
            [translational_constraints, torsional_constraints],
        )

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
        dv, u, z = jnp.split(
            q, [self.dv_idx, self.u_idx, self.z_idx],
        )

        # Task Space Objective:
        J_task = data.taskspace_jacobian
        task_bias = data.taskspace_bias
        ddx_task = J_task @ dv + task_bias

        ddx_base, ddx_fl, ddx_fr, ddx_hl, ddx_hr = jnp.split(
            ddx_task, self.num_targets,
        )
        ddx_base_t, ddx_fl_t, ddx_fr_t, ddx_hl_t, ddx_hr_t = jnp.split(
            desired_task_ddx, self.num_targets,
        )

        # Objective Function:
        objective_terms = {
            'base_translational_tracking': self._objective_tracking(
                ddx_base[:3], ddx_base_t[:3],
            ),
            'base_rotational_tracking': self._objective_tracking(
                ddx_base[3:], ddx_base_t[3:],
            ),
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

    def initialize_optimization(self) -> None:
        """Initialize the Optimization Problem."""
        # Generate Optimization Functions: (jacrev -> row by row, jacfwd -> column by column)
        self.Aeq_fn = jax.jacfwd(self.equality_constraints)
        self.Aineq_fn = jax.jacfwd(self.inequality_constraints)
        self.H_fn = jax.jacfwd(jax.jacrev(self.objective))
        self.f_fn = jax.jacfwd(self.objective)

        # Constant Matricies:
        optimization_size = self.dv_size + self.u_size + self.z_size
        self.Abox = jnp.eye(N=optimization_size, M=optimization_size)
        # Joint Acceleration Bounds: dv = [qdd]
        dv_lb = -jnp.inf * jnp.ones((self.dv_size,))
        dv_ub = jnp.inf * jnp.ones((self.dv_size,))
        # Control Input Bounds: u = [tau_fl, tau_fr, tau_hl, tau_hr]
        u_lb = jnp.array(self.torque_limits[:, 0])
        u_ub = jnp.array(self.torque_limits[:, 1])
        # Reaction Forces: z = [f_x, f_y, f_z, tau_x, tau_y, tau_z]
        z_lb = jnp.array(
            [-jnp.inf, -jnp.inf, 0.0, -jnp.inf, -jnp.inf, -jnp.inf] * 4,
        )
        z_ub = jnp.array(
            [jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf] * 4,
        )
        self.box_lb = jnp.concatenate([dv_lb, u_lb, z_lb])
        self.box_ub = jnp.concatenate([dv_ub, u_ub, z_ub])

        # Inequality Constraints are constant:
        self.Aineq = self.Aineq_fn(self.q)
        self.bineq_ub = -self.inequality_constraints(self.q)
        self.bineq_lb = -jnp.inf * jnp.ones_like(self.bineq_ub)

    def update(
        self, taskspace_targets: jax.Array, data: OSCData
    ) -> OptimizationData:
        """Update the Mathematical Program Matricies."""
        Aeq = self.Aeq_fn(self.q, data)
        beq = -self.equality_constraints(self.q, data)

        H = self.H_fn(self.q, taskspace_targets, data)
        f = self.f_fn(self.q, taskspace_targets, data)

        A = jnp.concatenate([Aeq, self.Aineq, self.Abox])
        lb = jnp.concatenate([beq, self.bineq_lb, self.box_lb])
        ub = jnp.concatenate([beq, self.bineq_ub, self.box_ub])

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
        bias = self.actuator_gear * (
            self.actuator_biasprm[:, 1] * q + self.actuator_biasprm[:, 2] * qd
        )
        return self.actuator_gear * (self.actuator_gainprm[:, 0] * u + bias)
