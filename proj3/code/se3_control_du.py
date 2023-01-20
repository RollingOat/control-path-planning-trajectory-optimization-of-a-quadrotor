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
        # Set PD controller constants (matrices)
        self.K_d   = np.diag([4.68, 4.68, 6.8])
        self.K_p   = np.diag([6.98, 6.98, 18.98])
        self.K_rot = np.diag([2666, 2666, 166])
        self.K_w   = np.diag([136, 136, 88])

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

        # STUDENT CODE HERE
        # Extract useful data
        x = state['x']  # m
        x_dot = state['v']  # m/s
        q = state['q']  # quaternion [i,j,k,w]
        w = state['w']  # rad/s
        x_t      = flat_output['x']       # m
        x_dot_t  = flat_output['x_dot']   # m/s
        x_ddot_t = flat_output['x_ddot']  # m/s**2
        yaw_t     = flat_output['yaw']     # rad

        # <editor-fold desc="Derive the Rotation matrix (rot_mat)">
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]
        q4 = q[3]
        # First row of the rotation matrix
        r00 = 2 * (q4 ** 2 + q1 ** 2) - 1
        r01 = 2 * (q1 * q2 - q4 * q3)
        r02 = 2 * (q1 * q3 + q4 * q2)
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q4 * q3)
        r11 = 2 * (q4 ** 2 + q2 ** 2) - 1
        r12 = 2 * (q2 * q3 - q4 * q1)
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q4 * q2)
        r21 = 2 * (q2 * q3 + q4 * q1)
        r22 = 2 * (q4 ** 2 + q3 ** 2) - 1
        rot_mat = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
        # </editor-fold>

        # A Geometric Nonlinear Controller
        # Step 1 - Calculate F_des (Total commanded force)
        x_ddot_des = x_ddot_t - np.matmul(self.K_d, (x_dot - x_dot_t)) - np.matmul(self.K_p, (x - x_t))
        force_des = self.mass*x_ddot_des + np.array([0, 0, self.mass*self.g])

        # Step 2 - Compute u_1 (input)
        b_3 = rot_mat[:, 2]
        u_1 = np.matmul(b_3, force_des)

        # Step 3 - Determine rot_mat_res (Desired rotation matrix)
        b_3_des = force_des/np.linalg.norm(force_des)
        a_yaw = np.array([np.cos(yaw_t), np.sin(yaw_t), 0])
        b_2_des = np.cross(b_3_des, a_yaw)/np.linalg.norm(np.cross(b_3_des, a_yaw))
        rot_mat_res = np.transpose(np.array([np.cross(b_2_des, b_3_des), b_2_des, b_3_des]))

        # Step 4 - Find the error orientation error vector (e_rot) and error in angular velocities (e_w)
        mat_skew = np.matmul(np.transpose(rot_mat_res), rot_mat) - np.matmul(np.transpose(rot_mat), rot_mat_res)
        e_rot = 0.5*(np.array([mat_skew[2, 1],
                               mat_skew[0, 2],
                               mat_skew[1, 0]]))
        e_w = w - np.array([0, 0, 0])

        # Step 5 - Compute u_2 (control input)
        u_2 = np.matmul(self.inertia, (-np.matmul(self.K_rot, e_rot) - np.matmul(self.K_w, e_w)))

        # Get the cmd_thrust
        gamma = self.k_drag/self.k_thrust
        l = self.arm_length
        mat_mid = np.array([[    1,      1,     1,      1],
                            [    0,      l,     0,     -l],
                            [   -l,      0,     l,      0],
                            [gamma, -gamma, gamma, -gamma]])
        cmd_thrusts = np.linalg.solve(mat_mid, np.append(np.array([u_1]), u_2))
        cmd_thrust = sum(cmd_thrusts)

        # Replace the invalid values with limited maximum and minimum
        max_thrust = self.k_thrust * self.rotor_speed_max * self.rotor_speed_max
        min_thrust = self.k_thrust * self.rotor_speed_min * self.rotor_speed_min
        cmd_thrusts[cmd_thrusts < min_thrust] = min_thrust
        cmd_thrusts[cmd_thrusts > max_thrust] = max_thrust

        # Get the cmd_moter_speeds
        cmd_motor_speeds = np.sqrt(cmd_thrusts/self.k_thrust)

        # Get the cmd_moment
        cmd_moment = u_2

        # Get the cmd_q
        r = Rotation.from_matrix(rot_mat_res)
        cmd_q = r.as_quat()

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
