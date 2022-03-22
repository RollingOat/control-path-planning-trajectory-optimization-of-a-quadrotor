import numpy as np

from graph_search import graph_search

import scipy

class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!

        # increase margin and resolution will prevent from finding shorter path, smaller margin and smaller resolution
        # can decrease distance
        self.resolution = np.array([0.2, 0.2, 0.2])
        self.margin = 0.1

        # assume the same average velocity between waypoints
        self.mean_velocity = 0.2

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=False)
        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        def remove_redundancy(original_path):
            '''
            Remove redundancy of path by remove intermediate points on a straight line 
            INPUT:
                original_path: ndarray of shape = (n_pts,3), the path returned from A* algorithm
            RETURN:
                sparse_path: ndarray of shape = (less_n_pts,3), the sparse path
            '''
            
            direction_vector = original_path[1:len(original_path)] - original_path[0:len(original_path)-1]
            distance = np.linalg.norm(direction_vector, axis = 1).reshape(-1, 1)
            unit_direction_vector = direction_vector / distance
            direction_difference = unit_direction_vector[1:len(unit_direction_vector)] - unit_direction_vector[0:len(unit_direction_vector)-1]

            zero_array = np.zeros_like(direction_difference)
            is_close_to_zero = []
            for i in range(len(direction_difference)):
                is_close_to_zero.append(np.allclose(direction_difference[i],zero_array[i],rtol = 1))
            is_close_to_zero = np.array(is_close_to_zero)

            index_is_not_zero = np.argwhere(is_close_to_zero == False).flatten()
            sparse_path_index = index_is_not_zero + 1

            list_sparse_path_index = list(sparse_path_index)
            list_sparse_path_index.insert(0,0)
            list_sparse_path_index.append(-1)
            sparse_path = original_path[list_sparse_path_index]

            return sparse_path
        
        self.points = remove_redundancy(self.path) # shape=(n_pts,3)
        self.points = self.path
        self.num_wayPoints = len(self.points)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE
        # min jerk trajectory
        def compute_coeff(waypoints,time):
            '''
            compute the coefficients of the trajectory polynomials
            INPUTS: 
                waypoints: the x or y or z coordinate of simplified waypoints path, ndarray of shape (npts, )
            Return:
                coeff: coefficients of each segment, ndarray of shape (num of segments, 6) 
            '''
            num_of_points = len(waypoints)
            num_of_segment = num_of_points - 1

            # create A and b matrix to solve Ax=b
            start_points = waypoints[0]
            end_points = waypoints[-1]
            b = [start_points, 0, 0]

            # The start points constraints
            A_start_points = np.array(
                [
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 2, 0, 0]
                ]
            )

            # endpoints constrants
            t = time[-1].item()
            A_end_points = np.array(
                [
                    [t**5, t**4, t**3, t**2, t, 1],
                    [5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0],
                    [20*t**3, 12*t**2, 6*t, 2, 0, 0]
                ]
            )
            A = np.zeros((6*num_of_segment, 6*num_of_segment)) 
            A[0:3,0:6] = A_start_points
            A[-3:,-6:] = A_end_points
            row_index = 3
            col_index = 0
            for i in range(num_of_segment-1):
                # create block vector of b
                b_middle = [waypoints[i+1], 0, 0, 0, 0, waypoints[i+1]]
                b = b + b_middle

                # create the block matrix of A
                t = time[i].item() # the time at p1,p2,...,pt
                time_block = np.array(
                    [
                        [t**5, t**4, t**3, t**2, t, 1],# x1(t1) = x1
                        [5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0], # x1_dot(t1) = x2_dot(t1)
                        [20*t**3, 12*t**2, 6*t, 2, 0, 0], 
                        [60*t**2, 24*t, 6, 0, 0, 0],
                        [120*t, 24, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]
                    ]
                )
                A[row_index:row_index+6,col_index:col_index+6] = time_block
                constant_block = np.array(
                    [
                        [0, 0, 0, 0, 0, 0],# x1(t1) = x1
                        [0, 0, 0, 0, -1, 0], # x1_dot(t1) = x2_dot(t1)
                        [0, 0, 0, -2, 0, 0], 
                        [0, 0, -6, 0, 0, 0],
                        [0, -24, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1] 
                    ]
                )
                A[row_index:row_index+6,col_index+6:col_index+12] = constant_block
                row_index = row_index+6
                col_index = col_index+6


            b = b + [end_points, 0, 0]
            b = np.array(b).reshape(-1,1)
            
            # solve equation
            coeff = scipy.linalg.solve(A,b)
            coeff = coeff.reshape(num_of_segment,6)

            return coeff

        # compute the polynomial coefficients of x y z
        if self.num_wayPoints > 1:
                distance = self.points[1:self.num_wayPoints] - self.points[0:self.num_wayPoints- 1]
                distance = np.linalg.norm(distance, axis = 1).reshape(-1, 1)
                # print(distance.shape)
                # calculate the time between two points
                time = (distance / self.mean_velocity).flatten()
                # time = np.ones_like(distance)

                self.coeff_x = compute_coeff(self.points[:,0],time)
                self.coeff_y = compute_coeff(self.points[:,1],time)
                self.coeff_z = compute_coeff(self.points[:,2],time)
                    
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
         # initialize n-1 segment unit directional vector, n-1*1 ndarray
        # calculate distance of each segment, n-1*1 ndarray
        if self.num_wayPoints > 1:
            distance = self.points[1:self.num_wayPoints] - self.points[0:self.num_wayPoints - 1]
            distance = np.linalg.norm(distance, axis = 1).reshape(-1, 1)

            direction_vector = self.points[1:self.num_wayPoints] - self.points[0:self.num_wayPoints - 1]
            direction_vector = direction_vector / distance

            # calculate the time between two points
            time_of_segment = (distance / self.mean_velocity).flatten()

            # start_time_of each segment, if the start time of the first segment is 0
            start_time_of_segment = (np.cumsum(time_of_segment) - time_of_segment).flatten()

            if t > (start_time_of_segment[-1] + time_of_segment[-1]):
                x = self.points[-1]
            else:
                # determine which segment the drone is in given present time, return the index of segment
                segment_location_map_1 = (start_time_of_segment + time_of_segment > t)
                segment_location_map_2 = (t >= start_time_of_segment)
                segment_location_map_3 = segment_location_map_1 & segment_location_map_2
                segment_location = np.argwhere(segment_location_map_3 == True)[0].item()
                
                # calculate the time expreseed in a single segment
                t = t - start_time_of_segment[segment_location]

                # calculate x
                x_t = np.array([t**5,t**4,t**3,t**2,t,1])
                x_x = self.coeff_x[segment_location] @ x_t.T
                x_y = self.coeff_y[segment_location] @ x_t.T
                x_z = self.coeff_z[segment_location] @ x_t.T
                x= np.array([x_x,x_y,x_z])

                x_dot_t = np.array([5*t**4,4*t**3,3*t**2,2*t,1,0])
                x_dot_x = self.coeff_x[segment_location]@x_dot_t.T
                x_dot_y = self.coeff_y[segment_location]@x_dot_t.T
                x_dot_z = self.coeff_z[segment_location]@x_dot_t.T
                x_dot= np.array([x_dot_x,x_dot_y,x_dot_z])

                x_ddot_t = np.array([20*t**3,12*t**2,6*t,2,0,0])
                x_ddot_x = self.coeff_x[segment_location]@x_ddot_t.T
                x_ddot_y = self.coeff_y[segment_location]@x_ddot_t.T
                x_ddot_z = self.coeff_z[segment_location]@x_ddot_t.T
                x_ddot= np.array([x_ddot_x,x_ddot_y,x_ddot_z])

                x_dddot_t = np.array([60*t**2,24*t,6,0,0,0])
                x_dddot_x = self.coeff_x[segment_location]@x_dddot_t.T
                x_dddot_y = self.coeff_y[segment_location]@x_dddot_t.T
                x_dddot_z = self.coeff_z[segment_location]@x_dddot_t.T
                x_dddot= np.array([x_dddot_x,x_dddot_y,x_dddot_z])

                x_ddddot_t = np.array([120*t,24,0,0,0,0])
                x_ddddot_x = self.coeff_x[segment_location]@x_ddddot_t.T
                x_ddddot_y = self.coeff_y[segment_location]@x_ddddot_t.T
                x_ddddot_z = self.coeff_z[segment_location]@x_ddddot_t.T
                x_ddddot= np.array([x_ddddot_x,x_ddddot_y,x_ddddot_z])

        else:
            x = self.points[0]


        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
                        
        return flat_output
