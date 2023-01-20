from heapq import heapify,heappush, heappop  # Recommended.
import numpy as np

from flightsim.world import World

from .occupancy_map import OccupancyMap # Recommended.

class Node:
    '''
    A customized class Node
    attributes:
        total_cost: total_cost = heuristic cost + cost_to_come
        heuristic_cost: the L2 norm distance between the node and goal node
        cost_to_come: the best cost to visit the node
        parent_index: the index of parent node, i.e [x, y, z]
        is_visited: True if visited
        index: the index of current node, i.e [x, y, z]
    '''

    def __init__(self,heuristic_cost, cost_to_come, index, parent_index, astar):
        self.heuristic_cost = heuristic_cost
        self.cost_to_come = cost_to_come
        self.index = index
        self.parent_index = parent_index
        self.is_visited = False
        self.total_cost = self.heuristic_cost + self.cost_to_come
        self.astar = astar

    def __lt__(self,other):
        if self.astar:
            return self.total_cost < other.total_cost
        else:
            return self.cost_to_come < other.cost_to_come


def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    #print(occ_map.map.shape)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    # print('start',start)
    # print('goal',goal)
    start_index = tuple(occ_map.metric_to_index(start)) # tuple (2,2,2)
    goal_index = tuple(occ_map.metric_to_index(goal)) # tuple (5, 15, 4)
    # print('start_node', start_index)
    # print('goal_index',goal_index)
    # Return a tuple (path, nodes_expanded)

    # the following code is to initialize a priority queue and a dictionary with all nodes stored inside
    heap = []
    dic = dict()
    unvisited_dic = dict()
    for x in range(occ_map.map.shape[0]):
        for y in range(occ_map.map.shape[1]):
            for z in range(occ_map.map.shape[2]):
                index = (x, y, z)
                parent_index = None
                heuristic_cost = np.linalg.norm(np.array(index)-np.array(goal_index)) # calculate the distance(L2 norm) to goal in index
                if x == start_index[0] and y == start_index[1] and z == start_index[2]: # set the cost of the start node to be zero
                    cost_to_come = 0
                    node = Node(heuristic_cost, cost_to_come, index, parent_index, astar)
                    heappush(heap,node)
                else:
                    cost_to_come = np.inf
                    node = Node(heuristic_cost, cost_to_come, index, parent_index, astar)
                dic[index] = node # add the node to a dictionary referenced by its index
                unvisited_dic[index] = node
                # help_node = Node(np.inf,np.inf,(-1,-1,-1),None,astar)
                # heappush(heap,help_node)
    
    initial_num_node = len(dic)
    
    
    # the following code is excuting breath-first search
    if astar:
        comp = heap[0].total_cost
    else:
        comp = heap[0].cost_to_come

    while (unvisited_dic.get(goal_index) != None) and (comp < np.inf) and len(heap) !=0:
        visit_node = heappop(heap)
        visit_node.is_visited = True
        # print('visit node',visit_node.index,' ', 'visit node cost',visit_node.cost_to_come)
        # print(visit_node.index)
        del unvisited_dic[visit_node.index]
        # print('is goal visited', not goal_index in unvisited_dic)
        # print('visit_node',visit_node.index,'visit_node parent',visit_node.parent_index)
        neighboring_node = find_neighbors(visit_node, dic, occ_map)
        for node in neighboring_node:
            edge_cost = np.linalg.norm(np.array(node.index)-np.array(visit_node.index))
            d = visit_node.cost_to_come + edge_cost
            if node.cost_to_come == np.inf and d < node.cost_to_come:
                heappush(heap, node)
            if d < node.cost_to_come:
                node.cost_to_come = d
                node.total_cost = node.cost_to_come + node.heuristic_cost
                node.parent_index = visit_node.index
                
                # print('node:',node.index,' ','parent:',visit_node.index,' ','node cost',node.cost_to_come)
        heapify(heap)

    # retrive the shortest path 
    num_node_expanded = initial_num_node - len(unvisited_dic)
    current_node = dic[goal_index]
    index_path = [current_node.index]

    while current_node.parent_index != None:
        parent_node = dic[current_node.parent_index]
        index_path.insert(0, parent_node.index)
        current_node = parent_node
    #print(index_path)
    if index_path[0] != start_index:
        path = None
    else:
        path = np.zeros((len(index_path),3))
        for i in range(len(index_path)):
            node_in_meter = occ_map.index_to_metric_center(index_path[i])
            path[i] = node_in_meter
        path[0] = start
        path[-1] = goal
    #print(path)
    return path, num_node_expanded

def find_neighbors(node, dic, occ_map):
    '''
    find the neighboring nodes of a given nodes in a map. Qualifeid neighbors should not exceed the boundary and not occupied
    Input:
        node: the neighbors of which to be found
        dic: the dictionary which stores all the node indexed by their coordinate
        map: an occupancy_map object
    Return:
        a list of neighboring nodes
    '''
    index = node.index
    neighbor_list = []
    for x in range(index[0]-1, index[0]+2):
        for y in range(index[1]-1, index[1]+2):
            for z in range(index[2]-1, index[2]+2):
                if not occ_map.is_occupied_index((x,y,z)):
                    # print((x,y,z),' ',occ_map.is_occupied_index((x,y,z)),' ',dic[(x,y,z)].is_visited)
                    if not dic[(x,y,z)].is_visited:
                        # print('nb node',(x,y,z))
                        neighbor_list.append(dic[(x,y,z)])
                            
    return neighbor_list
