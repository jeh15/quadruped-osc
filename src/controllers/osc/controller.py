import jax
import jax.numpy as jnp

from src.controllers.osc.utilities import OSCData


class OSCController:
    def __init__(self):
        # Design Variable Indices:
        self.dv_idx = self.dv_size
        self.u_idx = self.dv_idx + self.u_size
        self.z_idx = self.u_idx + self.z_size
        self.slack_idx = self.z_idx + self.slack_size
        
        # Model Variables:
        self.r = 0.02
        self.mu = 0.8

        # Utility Variables:
        self.num_contacts = self.z_size // 6

        pass


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
        dv, u, z, slack_vars = jnp.split(
            q, [self.dv_idx, self.u_idx, self.z_idx, self.slack_idx],
        )
        # Unpack Data:
        M = data.mass_matrix
        C = data.coriolis_matrix
        J_contact = data.contact_jacobian
        J_task = data.taskspace_jacobian
        task_bias = data.taskspace_bias

        # Dynamics Constraint:
        dynamics = M @ dv + C - self.B @ u - J_contact.T @ z

        # Taskspace Objective Slack Variable Formulation: (Experiment with this placement)
        ddx_task = J_task @ dv + task_bias
        ddx_task_slack = ddx_task - slack_vars

        # Concatenate Constraints:
        equality_constraints = jnp.concatenate([dynamics, ddx_task_slack])

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
        dv, u, z, slack_vars = jnp.split(
            q, [self.dv_idx, self.u_idx, self.z_idx, self.slack_idx],
        )

        # Friction Constraints:
        # Translational: |f_x| + |f_y| <= mu * f_z
        # Torsional: |tau_z| <= r x (mu * f_z)

        def translational_friction(z: jax.Array) -> jax.Array:
            constraint_1 = z[0] + z[1] - self.mu * z[2]
            constraint_2 = -z[0] + z[1] - self.mu * z[2]
            constraint_3 = z[0] - z[1] - self.mu * z[2]
            constraint_4 = -z[0] - z[1] - self.mu * z[2]
            return jnp.array([constraint_1, constraint_2, constraint_3, constraint_4])
        
        def torsional_friction(z: jax.Array) -> jax.Array:
            constraint_1 = z[5] - self.r * self.mu * z[2]
            constraint_2 = -z[5] - self.r * self.mu * z[2]
            return jnp.array([constraint_1, constraint_2])
        
        # Split Contact Forces:
        contact_forces = jnp.split(z, self.num_contacts)

        translational_constraints = jax.vmap(translational_friction)(contact_forces).flatten()
        torsional_constraints = jax.vmap(torsional_friction)(contact_forces).flatten()

        return jnp.concatenate([translational_constraints, torsional_constraints])
