import numpy as np
from math import pi
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import Polygon2
from sklearn.neighbors import NearestNeighbors

DUBIN_FACTOR = 5
DUBIN_SEGMENT_END = 5
USE_EUC_KNN = True


def generate_qrandom_free(map: rtb.PolygonMap, vehicle: rtb.VehicleBase):
    while True:
        q = np.random.uniform(
            low=(map.workspace[0], map.workspace[2], -np.pi),
            high=(map.workspace[1], map.workspace[3], np.pi),
            size=(3,),
        )
        if not map.iscollision(vehicle.polygon(q)):
            return q


def path_in_collision(path: np.ndarray, map: rtb.PolygonMap, vehicle: rtb.VehicleBase):
    for pose in path:
        if map.iscollision(vehicle.polygon(pose)):
            return True
    return False


def calculate_q_path_dubins(
    dubins: rtb.DubinsPlanner,
    q_nearest: np.ndarray,
    q_target: np.ndarray,
):
    dpath, dstat = dubins.query(start=q_nearest, goal=q_target)

    if dpath.shape[0] < DUBIN_SEGMENT_END:
        return dpath, dpath, dstat
    return dpath[:DUBIN_SEGMENT_END, :], dpath, dstat


def nearest_neighbor_search(dubins: rtb.DubinsPlanner, vertices: np.ndarray, q_target: np.ndarray):
    if USE_EUC_KNN:
        nbrs = NearestNeighbors(n_neighbors=1).fit(vertices)
        indx_qnear_all = nbrs.kneighbors([q_target], 1, return_distance=False)
        return indx_qnear_all

    dists = np.zeros(vertices.shape[0])

    for i in range(vertices.shape[0]):
        dpath, dstat = dubins.query(start=vertices[i], goal=q_target)
        dists[i] = dstat.length

    closest = np.argmin(dists)
    return [[closest]]


def trace_path(
    indx_last_vertex: int,
    vertices: np.ndarray,
    parent_indices: np.ndarray,
    npoints: int,
):
    path = np.zeros((npoints, 3))
    next = indx_last_vertex

    for istep in range(npoints):
        path[istep, :] = vertices[next, :]
        next = parent_indices[next]

        if next == 0:
            return True, np.flip(path[0 : (istep + 1), :], axis=0)

    return False, np.flip(path, axis=0)


def path_indices(
    indx_last_vertex: int,
    parent_indices: np.ndarray,
    npoints: int,
    indx_first_vertex: int = 0,
):
    path = [indx_last_vertex]
    next = indx_last_vertex
    while True:
        next = parent_indices[next]
        path.append(next)
        if next == indx_first_vertex:
            break
    return np.flip(path)


def plot_rrt_tree(tree: dict):
    start = tree["vertices"][0]
    arrow_size = 0.28
    length = arrow_size
    dx = np.cos(start[2]) * length
    dy = np.sin(start[2]) * length
    plt.arrow(
        start[0],
        start[1],
        dx,
        dy,
        head_width=arrow_size,
        head_length=length,
        length_includes_head=True,
        alpha=0.85,
        edgecolor=None,
        linewidth=None,
        facecolor="blue",
    )
    plt.plot(
        start[0],
        start[1],
        marker="o",
        markersize=10,
        markeredgecolor="green",
        markerfacecolor="green",
    )

    final_path = tree["final_path"]
    plt.plot(
        final_path[:, 0],
        final_path[:, 1],
        linewidth=3,
        alpha=0.7,
        color="black",
    )

    for indx_node in range(1, tree["vertices"].shape[0]):
        node = tree["vertices"][indx_node]
        arrow_size = 0.2
        plt.plot(
            node[0],
            node[1],
            marker="o",
            markersize=5,
            markeredgecolor="darkblue",
            markerfacecolor="darkblue",
            alpha=0.2,
        )
        dx = np.cos(node[2]) * length
        dy = np.sin(node[2]) * length
        plt.arrow(
            node[0],
            node[1],
            dx,
            dy,
            head_width=arrow_size,
            head_length=length,
            length_includes_head=True,
            alpha=0.5,
            edgecolor=None,
            linewidth=None,
            facecolor="darkblue",
        )

        path = tree["edges"][indx_node]
        plt.plot(path[:, 0], path[:, 1], linewidth=1, alpha=0.7, color="black")

    if "overwritten_edges" in tree.keys():
        overwritten_edges = tree["overwritten_edges"]
        for iter_edge in range(len(overwritten_edges)):
            ov_edge = overwritten_edges[iter_edge]
            plt.plot(ov_edge[:, 0], ov_edge[:, 1], linewidth=2, color="red")

    if "new_edges" in tree.keys():
        new_edges = tree["new_edges"]
        for iter_edge in range(len(new_edges)):
            new_edge = new_edges[iter_edge]
            plt.plot(new_edge[:, 0], new_edge[:, 1], linewidth=2, color="yellow")


def pose_dist(a: np.ndarray, b: np.ndarray):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return np.sqrt(dx * dx + dy * dy)


def flatten(l):
    return [item for sublist in l for item in sublist]


# FIRST RRT IMPLEMENTATION (here for reference)
# Generally not used in final results - turn off flags in rrt_star.py to use RRT only
def rrt_path_plan(
    pmap: rtb.PolygonMap,
    vehicle: rtb.VehicleBase,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    npoints: int = 200,
    rtt_stepsize: float = 4,
    bias: float = 0.01,
):
    if pmap.iscollision(vehicle.polygon(q_start)) or pmap.iscollision(vehicle.polygon(q_goal)):
        return False, []

    parent_indices = np.zeros((npoints,), dtype=np.uint8)
    vertices = np.zeros((npoints, 3))
    edges = [[]]
    vertices[0, :] = q_start
    indx_vertex_to_fill = 1

    dubins = rtb.DubinsPlanner(curvature=2, stepsize=rtt_stepsize / DUBIN_FACTOR)
    dubins.plan()

    while indx_vertex_to_fill < npoints:
        if np.random.random() < bias:
            q_target = q_goal
        else:
            q_target = generate_qrandom_free(pmap, vehicle)
        plt.plot(
            q_target[0],
            q_target[1],
            marker="o",
            markersize=5,
            markeredgecolor="pink",
            markerfacecolor="pink",
        )
        # retrieve k-nearest neighbors by index in the filled part of vertices
        indx_qnear_all = nearest_neighbor_search(dubins, vertices[:indx_vertex_to_fill, :], q_target)[0]

        # retrieve index and value of nearest neighbor
        indx_qnearest = indx_qnear_all[0]
        q_nearest = vertices[indx_qnearest, :]

        # calculate new pose
        q_path, q_path_full, q_stat = calculate_q_path_dubins(dubins, q_nearest, q_target)

        # collision check before adding to tree
        if not path_in_collision(q_path, pmap, vehicle):
            q_new = q_path[-1, :]
            vertices[indx_vertex_to_fill, :] = q_new
            parent_indices[indx_vertex_to_fill] = indx_qnearest
            edges.append(q_path)

            if pose_dist(q_new, q_goal) < 0.5:
                print("FOUND THE GOAL")
                break

            if indx_vertex_to_fill == npoints - 1:
                print("did not find goal...")
                break

            indx_vertex_to_fill += 1

    path_found, path = trace_path(indx_vertex_to_fill, vertices, parent_indices, npoints)

    final_path_nodes = path_indices(indx_vertex_to_fill, parent_indices, npoints)
    path_segments = list(map(lambda i: edges[i], final_path_nodes))
    final_path = np.array(flatten(path_segments))

    rrtTree = {}
    rrtTree["goal"] = q_goal
    rrtTree["vertices"] = vertices[: indx_vertex_to_fill + 1]
    rrtTree["parent_indices"] = parent_indices[: indx_vertex_to_fill + 1]
    rrtTree["edges"] = edges
    rrtTree["final_path"] = final_path

    plot_rrt_tree(rrtTree)

    return path_found, path
