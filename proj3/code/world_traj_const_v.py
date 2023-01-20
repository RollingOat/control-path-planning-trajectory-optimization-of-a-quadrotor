import numpy as np
from .graph_search import graph_search

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
        self.resolution = np.array([0.35, 0.35, 0.25])
        self.margin = 0.5

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
            # find the difference between unit directional vector
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
            if list_sparse_path_index[0] != 1:
                list_sparse_path_index.insert(0,1)
            list_sparse_path_index.insert(0,0)
            list_sparse_path_index.append(-1)

            sparse_path = original_path[list_sparse_path_index]
            return sparse_path

        sparse_path_index = np.arange(1,len(self.path)-1,4)
        list_sparse_path_index = list(sparse_path_index)
        list_sparse_path_index.insert(0,0)
        list_sparse_path_index.append(-1)
        self.points = self.path[list_sparse_path_index]

        # self.points = remove_redundancy(self.path) # shape=(n_pts,3)
        # self.points = self.path

        self.num_wayPoints = len(self.points)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE
        # constant acceleration trajectory
        self.time = 0.6

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
        x = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        # initialize n-1 segment unit directional vector, n-1*1 ndarray
        # calculate distance of each segment, n-1*1 ndarray
        if self.num_wayPoints > 1:
            # distance = self.points[1:self.num_wayPoints] - self.points[0:self.num_wayPoints - 1]
            # distance = np.linalg.norm(distance, axis = 1).reshape(-1,1)

            direction_vector = self.points[1:self.num_wayPoints] - self.points[0:self.num_wayPoints - 1]
            unit_direction_vector = np.zeros_like(direction_vector) # (n,3)
            distance = np.zeros((len(direction_vector),1)) #(n,1)
            for i in range(len(direction_vector)):
                distance[i] = np.linalg.norm(direction_vector[i])
                unit_direction_vector[i] = direction_vector[i]/distance[i]
                
            # velocity vector will be, n-1*1 ndarray
            # velocity = self.v * unit_direction_vector

            # calculate the duration of each time segment, n-1*1 ndarray
            # time_of_segment = (distance / self.v).flatten()
            fake_time_of_segment = np.full_like(distance, self.time).flatten()
            fake_time_of_segment[0] = 1.5
            
            speed = distance.flatten()/fake_time_of_segment
            speed_max = 4
            speed = np.clip(speed,0, speed_max)
            velocity = speed.reshape(-1,1) * unit_direction_vector

            time_of_segment = (distance.flatten() / speed)
            # start_time_of each segment, if the start time of the first segment is 0
            start_time_of_segment = (np.cumsum(time_of_segment)-time_of_segment).flatten()

            if t >= (start_time_of_segment[-1] + time_of_segment[-1]):
                x = self.points[-1]
                x_dot = np.zeros((3,))
            elif t<0:
                x = self.points[0]
            else:
                # determine which segment the drone is in given present time, return the index of segment
                segment_location_map_1 = (start_time_of_segment + time_of_segment > t)
                segment_location_map_2 = (t >= start_time_of_segment)
                segment_location_map_3 = segment_location_map_1 & segment_location_map_2
                segment_location = np.argwhere(segment_location_map_3 == True)[0].item()

                # calculate position r(t) = point_prev + (t-start_time)*velocity
                x = self.points[segment_location] + (t - start_time_of_segment[segment_location]) * velocity[segment_location]

                # calculate x_dot
                x_dot = velocity[segment_location] # np.zeros((3,))

        else:
            x = self.points[0]

        # print('t',t)
        # print('x',x)
        # print('x_dot',x_dot)

        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                       'yaw': yaw, 'yaw_dot': yaw_dot}
        
        return flat_output