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

        # Mixer matrix
        self.gamma = self.k_drag/self.k_thrust
        self.A = np.array([[1, 1, 1, 1],
                           [0, self.arm_length, 0, -self.arm_length],
                           [-self.arm_length, 0, self.arm_length, 0],
                           [self.gamma, -self.gamma, self.gamma, -self.gamma]])

        # Position gains
        self.kp = np.array([[6.644938272*1.0, 0, 0],
                            [0, 6.644938272*1.0, 0],
                            [0, 0, 6.644938272*0.9]])
        self.kd = np.array([[5.155555556*1.1, 0, 0],
                            [0, 5.155555556*1.1, 0],
                            [0, 0, 5.155555556*0.9]])

        # Angle gains
        self.kp_phi = 1495.111111*1
        self.kd_phi = 77.33333333*2.5
        self.kp_theta = 1495.111111*1
        self.kd_theta = 77.33333333*2.5
        self.kp_psi = 8.41*1.2
        self.kd_psi = 5.8*1.2

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
        x_a = state['x']
        v_a = state['v']
        w_a = state['w']
        q = state['q']
        phi = np.arctan2(2*(q[3]*q[0]+q[1]*q[2]), 1-2*(q[0]**2+q[1]**2))
        theta = np.arcsin(2*(q[3]*q[1]-q[0]*q[2]))
        psi = np.arctan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2))
        x_t = flat_output['x']
        v_t = flat_output['x_dot']
        a_t = flat_output['x_ddot']
        psi_t = flat_output['yaw']
        psi_dot_t = flat_output['yaw_dot']

        a_d = a_t - np.matmul(self.kd, v_a - v_t) - np.matmul(self.kp, x_a - x_t)
        cmd_thrust = self.mass*a_d[2] + self.mass*self.g

        phi_d = (-a_d[1]*np.cos(psi_t) + a_d[0])/(self.g*np.sin(psi_t)+self.g*(np.cos(psi_t)**2))
        theta_d = a_d[0]/(self.g*np.cos(psi_t))-phi_d*np.tan(psi_t)

        eq_mat = np.array([self.kp_phi*(phi_d-phi)-self.kd_phi*w_a[0],
                          self.kp_theta*(theta_d-theta)-self.kd_theta*w_a[1],
                          self.kp_psi*(psi_t-psi)+self.kd_psi*(psi_dot_t-w_a[2])])

        u2 = np.matmul(self.inertia, eq_mat)

        u = np.array([cmd_thrust, u2[0], u2[1], u2[2]])
        speeds_2 = np.matmul(np.linalg.inv(self.A), u)*(1/self.k_thrust)

        speeds = np.ones(4)
        for i in range(4):
            if speeds_2[i] < 0:
                speeds[i] = -np.sqrt(np.absolute(speeds_2[i]))
            else:
                speeds[i] = np.sqrt(speeds_2[i])
        cmd_motor_speeds = np.clip(speeds, self.rotor_speed_min, self.rotor_speed_max)
        cmd_moment = self.k_drag*cmd_motor_speeds
        cmd_euler = [phi_d, theta_d, psi_t]
        rot2 = Rotation.from_euler('xyz', cmd_euler)
        cmd_q = rot2.as_quat()
        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
