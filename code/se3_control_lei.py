import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE
        self.gamma = self.k_drag/self.k_thrust
        kd1 = 5#4#5#6#5#2#3
        kd3 = 6#6#7#5#6#11.6#2
        self.kd = np.diag(np.array([kd1, kd1, kd3]))

        kp1 = 10#4#10#2.05#3
        kp3 = 10#10#5#20#33.64#
        self.kp = np.diag(np.array([kp1, kp1, kp3]))

        kr1 = 450#158#450#3364#20#100#50
        kr3 = 25#158#25#3364#50
        self.kr = np.diag(np.array([kr1, kr1, kr3]))

        kw1 = 40#25.2#10#40#116#7#6.25
        kw3 = 10# 66#10#116#6.25
        self.kw = np.diag(np.array([kw1, kw1, kw3]))


    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # Nonlinear control version
        # calculate F_des and r_ddot_des
        r_ddot_des =flat_output['x_ddot'].reshape(-1,1) - self.kd @ (state['v'] - flat_output['x_dot']).reshape(-1,1) - \
                    self.kp @ (state['x'] - flat_output['x']).reshape(-1,1)
        F_des = self.mass * r_ddot_des + np.array([0, 0, self.mass*self.g]).reshape(-1,1) # shape (3,1)

        # calculate u1
        rot = Rotation.from_quat(state['q'])
        R = rot.as_matrix()
        b3_inertia = R[:,2].reshape(-1,1) # (3,)
        u1 = b3_inertia.T @ F_des.reshape(-1, 1)

        # calculate R_des
        b3_des = F_des / np.linalg.norm(F_des) # (3,)
        b3_des = b3_des.flatten()
        a_psi = np.array([np.cos(flat_output['yaw']), np.sin(flat_output['yaw']), 0])
        b2_des = np.cross(b3_des, a_psi) / np.linalg.norm(np.cross(b3_des, a_psi))
        b1_des = np.cross(b2_des, b3_des)

        R_des = np.zeros((3,3))
        R_des[:, 0] = b1_des
        R_des[:, 1] = b2_des
        R_des[:, 2] = b3_des

        # calculate error vector
        error_matrix = 1/2 * (R_des.T @ R - R.T @ R_des)
        e1 = error_matrix[2, 1]
        e2 = error_matrix[0, 2]
        e3 = error_matrix[1, 0]
        e_r = np.array([e1, e2, e3]).reshape(-1,1) # error vector
        e_w = state['w'].reshape(-1, 1)
        u2 = self.inertia @ (-self.kr @ e_r - self.kw @ e_w)

        # calculate cmd_motor_speeds
        u = np.array([u1.item(), u2[0].item(), u2[1].item(), u2[2].item()]).reshape(-1, 1)
        parameter_matrix = np.array(
            [
                [1, 1, 1, 1],
                [0, self.arm_length, 0, -self.arm_length],
                [-self.arm_length, 0, self.arm_length, 0],
                [self.gamma, -self.gamma, self.gamma, -self.gamma]
            ]
        )

        thrust = np.linalg.solve(parameter_matrix, u) # the solution to Ax=b

        motor_speed_square = np.abs(thrust)/self.k_thrust
        cmd_motor_speeds = np.sqrt(motor_speed_square.astype(float))
        cmd_motor_speeds = cmd_motor_speeds.flatten()
        for i in range(len(thrust)):
            if thrust[i] < 0:
                cmd_motor_speeds[i] = 0

        # calculate cmd_thrust, cmd_moment, cmd_q
        cmd_thrust = self.k_thrust * cmd_motor_speeds**2
        cmd_moment = self.k_drag * cmd_motor_speeds**2
        rot1 = Rotation.from_matrix(R_des)
        cmd_q = rot1.as_quat()

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input


