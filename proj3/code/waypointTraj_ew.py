import numpy as np

class WaypointTraj(object):
    """

    """
    def __init__(self, points):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
        """

        # STUDENT CODE HERE
        self.points = points
        self.v = 0.5
        self.num = len(points[:, 0])

        self.p_ii = points[1:self.num, 0:3]
        self.p_i = points[0:self.num-1, 0:3]

        self.dis1 = np.linalg.norm(self.p_ii-self.p_i, axis=1)
        self.dis = np.where(self.dis1 == 0, 1, self.dis1)
        self.diff = self.p_ii-self.p_i
        self.dir = np.transpose(np.divide(np.transpose(self.diff), self.dis))
        self.vel = self.v*self.dir
        self.T = self.dis/self.v
        self.T_s = np.cumsum(self.T)
        if len(self.T_s) > 0:
            self.T_e = self.T_s[-1]
        else:
            self.T_e = 0

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        if t != np.inf:
            if t >= self.T_e:
                x = self.points[self.num-1, 0:3]
                x_dot = np.zeros((3,))
            elif t < self.T_s[0]:
                x_dot = self.v * self.dir[0, 0:3]
                x = self.p_i[0, 0:3] + x_dot * t
            else:
                check = self.T_s - t*(np.ones(self.num-1))
                find = np.where(check <= 0)
                ind = find[-1]
                print('ind', ind[-1])
                current = self.p_i[ind[-1], 0:3]
                print('current', current)
                x_dot = self.v * self.dir[ind[-1]+1, 0:3]
                x = self.p_i[ind[-1]+1, 0:3] + x_dot * (t - self.T_s[ind[-1]])
        elif t == np.inf:
            x = self.points[self.num - 1, 0:3]
            x_dot = np.zeros((3,))

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        print('Time', t)
        print('x', x)
        return flat_output
