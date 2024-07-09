import heapq
import math

# A* algorithm is a popular pathfinding algorithm that is both complete and optimal.
# It uses a heuristic function to estimate the cost of the cheapest path from the current node to the goal node.
# The heuristic function is admissible if it never overestimates the cost to reach the goal node.
# The algorithm maintains two lists: open list and closed list.
# The open list contains nodes that are yet to be explored, while the closed list contains nodes that have been explored.
# The algorithm starts by adding the start node to the open list.
# While the open list is not empty, the algorithm pops the node with the lowest f-value (cost + heuristic) from the open list.
# If the current node is the goal node, the algorithm reconstructs the path from the start node to the goal node.
# Otherwise, the algorithm expands the current node by generating its neighbors and adding them to the open list.
# The algorithm continues until the goal node is found or the open list is empty.
def euclidean_distance(start, goal):
    return math.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)

def manhattan_distance(start, goal):
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

class Node:
    def __init__(self, x, y, cost, heuristic, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.heuristic = heuristic
        self.parent = parent

    def __lt__(self, other):
        return self.cost + self.heuristic < other.cost + other.heuristic

def a_star_search(start, goal):
    open_list = []
    closed_list = set()

    start_node = Node(start[0], start[1], 0, manhattan_distance(start, goal))
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        if (current_node.x, current_node.y) == goal:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        closed_list.add((current_node.x, current_node.y))

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            x, y = current_node.x + dx, current_node.y + dy
            if (x, y) not in closed_list:
                cost = current_node.cost + 1
                heuristic = manhattan_distance((x, y), goal)
                node = Node(x, y, cost, heuristic, current_node)
                heapq.heappush(open_list, node)

    return None

start_point = (0, 0)
goal_point = (-4, 5)

path = a_star_search(start_point, goal_point)
if path:
    print("Path found:", path)
else:
    print("No path found")