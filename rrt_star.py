from rrt import *
import numpy as np

NODE_IMPROVEMENT = True
NEIGHBOR_IMPROVEMENT = False


def compute_path_dist(path: np.ndarray):
    deltas = np.diff(path, axis=0)
    dists = np.sqrt(deltas[:, 0] * deltas[:, 0] + deltas[:, 1] * deltas[:, 1])
    return np.sum(dists)


def knnsearch_modified_rttstar(
    dubins: rtb.DubinsPlanner,
    vertices: np.ndarray,
    q_target: np.ndarray,
    rrt_star_radius: float = 3,
):
    all_neighbors = []

    for iV in range(vertices.shape[0]):
        dpath, stat = dubins.query(start=vertices[iV], goal=q_target)
        if stat.length <= rrt_star_radius:
            all_neighbors.append(iV)

    return all_neighbors


def rrt_star_path_plan(
    pmap: rtb.PolygonMap,
    vehicle: rtb.VehicleBase,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    rrt_star_radius: float = 3,
    npoints: int = 230,
    rtt_stepsize: float = 4,
    bias: float = 0.02,
):
    if pmap.iscollision(vehicle.polygon(q_start)) or pmap.iscollision(vehicle.polygon(q_goal)):
        return False, [], 0

    parent_indices = np.zeros((npoints,), dtype=np.uint8)
    vertices = np.zeros((npoints, 3))
    edges = [[]]
    edges_cost = [0]
    vertices[0, :] = q_start
    indx_vertex_to_fill = 1

    overwritten_edges = []
    new_edges = []

    dubins = rtb.DubinsPlanner(curvature=2, stepsize=rtt_stepsize / DUBIN_FACTOR)
    dubins.plan()

    def total_path_cost(vertex_index):
        indices = path_indices(vertex_index, parent_indices[: indx_vertex_to_fill + 1], None)
        return sum(map(lambda i: edges_cost[i], indices))

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
            alpha=0.8,
        )

        # retrieve k-nearest neighbors by index in the filled part of vertices
        indx_qnear_all = nearest_neighbor_search(dubins, vertices[:indx_vertex_to_fill, :], q_target)[0]

        # retrieve index and value of nearest neighbor
        indx_qnearest = indx_qnear_all[0]
        q_nearest = vertices[indx_qnearest, :]

        # calculate new pose
        q_path, q_path_full, q_stat = calculate_q_path_dubins(dubins, q_nearest, q_target)

        # collision check before adding to tree
        if len(q_path) != 0 and not path_in_collision(q_path, pmap, vehicle):
            q_new = q_path[-1, :]
            vertices[indx_vertex_to_fill, :] = q_new

            q_new_cost = total_path_cost(indx_qnearest) + compute_path_dist(q_path)

            all_neighbors = knnsearch_modified_rttstar(
                dubins,
                vertices[:indx_vertex_to_fill, :],
                q_new,
                rrt_star_radius=8,
            )

            q_min_path = q_path
            q_min_cost = q_new_cost
            indx_qmin = indx_qnearest
            best_edge_cost = compute_path_dist(q_path)

            if NODE_IMPROVEMENT:
                for iNbr in range(len(all_neighbors)):
                    indx_qneighbor = all_neighbors[iNbr]
                    q_neighbor = vertices[indx_qneighbor, :]

                    (
                        qneighbor_to_qnew_path,
                        qneighbor_to_qnew_stat,
                    ) = dubins.query(start=q_neighbor, goal=q_new)

                    if (not path_in_collision(qneighbor_to_qnew_path, pmap, vehicle)) and (
                        # edges_cost[indx_qneighbor]
                        # + qneighbor_to_qnew_stat.length
                        # < q_min_cost
                        total_path_cost(indx_qneighbor) + qneighbor_to_qnew_stat.length
                        < q_min_cost
                    ):
                        indx_qmin = indx_qneighbor
                        q_min_cost = total_path_cost(indx_qneighbor) + qneighbor_to_qnew_stat.length
                        q_min_path = qneighbor_to_qnew_path
                        best_edge_cost = qneighbor_to_qnew_stat.length
                        # print("IMPROVE1")

            edges.append(q_min_path)
            edges_cost.append(best_edge_cost)
            parent_indices[indx_vertex_to_fill] = indx_qmin

            if NEIGHBOR_IMPROVEMENT:
                for iNbr in range(len(all_neighbors)):
                    indx_qneighbor = all_neighbors[iNbr]
                    q_neighbor = vertices[indx_qneighbor, :]

                    (
                        qnew_to_qneighbor_path,
                        qnew_to_qneighbor_stat,
                    ) = dubins.query(start=q_new, goal=q_neighbor)

                    if (not path_in_collision(qnew_to_qneighbor_path, pmap, vehicle)) and (
                        q_min_cost + qnew_to_qneighbor_stat.length
                        # < edges_cost[indx_qneighbor]
                        < total_path_cost(indx_qneighbor)
                    ):
                        # Save copies of the new & overwritten edges to visualize the tree diff
                        overwritten_edges.append(edges[indx_qneighbor])
                        new_edges.append(qnew_to_qneighbor_path)

                        # q_parent = parent_indices[indx_qneighbor]
                        parent_indices[indx_qneighbor] = indx_vertex_to_fill
                        edges[indx_qneighbor] = qnew_to_qneighbor_path
                        # edges_cost[indx_qneighbor] = (
                        #     q_min_cost + qnew_to_qneighbor_stat.length
                        # )
                        edges_cost[indx_qneighbor] = qnew_to_qneighbor_stat.length

                        print("IMPROVE2")

            # Print statement helps keep visual progress through npoints
            # print("indx_vertex_to_fill", indx_vertex_to_fill)

            indx_vertex_to_fill += 1

    path_end_vertex_index = 1

    total_cost_to_goal = np.Inf

    # Find the first node which is effectively the goal
    path_found = False
    for i in range(vertices.shape[0]):
        if pose_dist(vertices[i], q_goal) < 0.5 and np.abs(vertices[i, 2] - q_goal[2]) < pi / 2:
            print("FOUND THE GOAL! At node", i)
            cost_to_goal = total_path_cost(i)
            if cost_to_goal < total_cost_to_goal:
                path_end_vertex_index = i
                path_found = True
                total_cost_to_goal = cost_to_goal

    if not path_found:
        indx_vertex_to_fill = len(vertices) - 1

    path_traced, path = trace_path(path_end_vertex_index, vertices, parent_indices, npoints)

    final_path_nodes = path_indices(path_end_vertex_index, parent_indices, npoints)
    path_segments = list(map(lambda i: edges[i], final_path_nodes))
    final_path = np.array(flatten(path_segments))

    rrtTree = {}
    rrtTree["goal"] = q_goal
    rrtTree["vertices"] = vertices[: indx_vertex_to_fill + 1]
    rrtTree["parent_indices"] = parent_indices[: indx_vertex_to_fill + 1]
    rrtTree["edges"] = edges
    rrtTree["final_path"] = final_path

    # New tree visualization for RRT* changes
    # rrtTree["overwritten_edges"] = overwritten_edges
    # rrtTree["new_edges"] = new_edges

    # rrtTree["overwritten_edges"] = overwritten_edges[-1:]
    # rrtTree["new_edges"] = new_edges[-1:]

    plot_rrt_tree(rrtTree)

    return path_found, path, total_cost_to_goal
