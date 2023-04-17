from rrt import *

USE_EUC_KNN_MODIFIED = False


def knnsearch_modified_rttstar(
    dubins: rtb.DubinsPlanner,
    vertices: np.ndarray,
    q_target: np.ndarray,
    rrt_star_radius: float = 8,
):

    if USE_EUC_KNN_MODIFIED:
        nbrs = NearestNeighbors(radius=rrt_star_radius).fit(vertices)
        q_near_dists, i_qnear_all = nbrs.radius_neighbors(
            [q_target], radius=rrt_star_radius, sort_results=True
        )
        return i_qnear_all[0], q_near_dists[0]

    dists = np.zeros(vertices.shape[0])

    for iV in range(vertices.shape[0]):
        dpath, stat = dubins.query(start=vertices[iV], goal=q_target)
        dists[iV] = stat.length

    # closest = np.argmin(dists)
    all_neighbors = []
    for iV in range(vertices.shape[0]):
        if dists[iV] <= rrt_star_radius:
            all_neighbors.append(iV)

    # print("closest", closest, vertices[closest])
    return all_neighbors, dists
    # return [[vertices[closest]]]

    # dists[i] =


def rrt_star_path_plan(
    pmap: rtb.PolygonMap,
    vehicle: rtb.VehicleBase,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    rrt_star_radius: float = 8,  # MODIFIED FROM RRT PLANNER
    npoints: int = 200,
    rtt_stepsize: float = 4,
    bias: float = 0.01,
):
    if pmap.iscollision(vehicle.polygon(q_start)) or pmap.iscollision(
        vehicle.polygon(q_goal)
    ):
        return False, []

    parent_indices = np.zeros((npoints,), dtype=np.uint8)
    vertices = np.zeros((npoints, 3))
    edges = [[]]
    edges_cost = [0]
    vertices[0, :] = q_start
    i_vertex_to_fill = 1

    dubins = rtb.DubinsPlanner(
        curvature=2, stepsize=rtt_stepsize / DUBIN_FACTOR
    )
    dubins.plan()

    while i_vertex_to_fill < npoints:
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
        i_qnear_all = knnsearch(
            dubins, vertices[:i_vertex_to_fill, :], q_target, k_nearest=1
        )[0]

        # retrieve index and value of nearest neighbor
        i_qnearest = i_qnear_all[0]
        q_nearest = vertices[i_qnearest, :]

        # calculate new pose
        q_path, q_path_full, q_stat = calculate_q_path_dubins(
            dubins, q_nearest, q_target
        )

        # plt.plot(
        #     q_path_full[:, 0],
        #     q_path_full[:, 1],
        #     linewidth=1,
        #     alpha=0.7,
        #     color="grey",
        # )

        # --------- BELOW MODIFIED FROM RTT PLANNER ---------

        # collision check before adding to tree
        if not path_in_collision(q_path, pmap, vehicle):
            # if not pmap.iscollision(vehicle.polygon(q_new)):
            q_new = q_path[-1, :]
            vertices[i_vertex_to_fill, :] = q_new
            # parent_indices[i_vertex_to_fill] = i_qnearest

            q_new_cost = edges_cost[i_qnearest] + sum(
                q_stat.seglengths[:DUBIN_SEGMENT_END]
            )  # NEW!
            # edges.append(q_path)
            # edges_cost.append(q_new_cost)  # NEW!

            # print("q_path_full shape:", q_path_full.shape)
            # print("q_path shape:", q_path.shape)
            # print("q_stat.length:", q_stat.length)
            # print("q_stat.seglengths:", q_stat.seglengths)
            # print("q_stat.seglengths shape:", len(q_stat.seglengths))
            # print(
            #     "q_stat.seglengths CLIPPED:",
            #     q_stat.seglengths[:DUBIN_SEGMENT_END],
            # )

            # print(
            #     "q_stat.seglengths CLIPPED shape",
            #     len(q_stat.seglengths[:DUBIN_SEGMENT_END]),
            # )

            all_neighbors, dists = knnsearch_modified_rttstar(
                dubins,
                vertices[:i_vertex_to_fill, :],
                q_new,
                rrt_star_radius=8,
            )

            q_min_cost = q_new_cost
            i_qmin = i_qnearest

            for iNbr in range(len(all_neighbors)):
                iq_neighbor = all_neighbors[iNbr]
                q_neighbor = vertices[iq_neighbor, :]

                qneighbor_to_qnew_path, qneighbor_to_qnew_stat = dubins.query(
                    start=q_neighbor, goal=q_new
                )

                if (
                    not path_in_collision(
                        qneighbor_to_qnew_path, pmap, vehicle
                    )
                ) and (
                    edges_cost[iq_neighbor] + qneighbor_to_qnew_stat.length
                    < q_min_cost
                ):
                    i_qmin = iq_neighbor
                    q_min_cost = (
                        edges_cost[iq_neighbor]
                        + qneighbor_to_qnew_stat.length
                    )
                    q_min_path = qneighbor_to_qnew_path

            edges.append(q_min_path)
            edges_cost.append(q_min_cost)
            parent_indices[i_vertex_to_fill] = i_qmin

            for iNbr in range(len(all_neighbors)):
                iq_neighbor = all_neighbors[iNbr]
                q_neighbor = vertices[iq_neighbor, :]

                qnew_to_qneighbor_path, qnew_to_qneighbor_stat = dubins.query(
                    start=q_new, goal=q_neighbor
                )

                if (
                    not path_in_collision(
                        qnew_to_qneighbor_path, pmap, vehicle
                    )
                ) and (
                    q_new_cost + qnew_to_qneighbor_stat.length
                    < edges_cost[iq_neighbor]
                ):
                    q_parent = parent_indices[iq_neighbor]
                    parent_indices[iq_neighbor] = i_vertex_to_fill
                    edges[iq_neighbor] = qnew_to_qneighbor_path

                # qneigbor_to_qnew_cost = dists[iNbr]

                # if (
                #     q_new_cost + qneigbor_to_qnew_cost
                #     < edges_cost[iq_neighbor]
                # ):
                #     edges_cost[iq_neighbor] = (
                #         q_new_cost + qneigbor_to_qnew_cost
                #     )
                #     edges[iq_neighbor] = np.concatenate(
                #         edges[i_qnearest], q_path
                #     )
                #     parent_indices[iq_neighbor] = i_vertex_to_fill

            # if np.array_equal(q_new, q_goal):
            #     break

            if pose_dist(q_new, q_goal) < 0.5:
                print("FOUND THE GOAL")
                break

            if i_vertex_to_fill == npoints - 1:
                print("did not find goal...")
                break

            i_vertex_to_fill += 1

    path_found, path = trace_path(
        i_vertex_to_fill, vertices, parent_indices, npoints
    )

    final_path_nodes = path_indices(i_vertex_to_fill, parent_indices, npoints)
    path_segments = list(map(lambda i: edges[i], final_path_nodes))
    final_path = np.array(flatten(path_segments))
    print("final_path", final_path)

    # print(parent_indices)
    # print(parent_indices[: i_vertex_to_fill + 1])

    # print(vertices)
    # print(vertices[: i_vertex_to_fill + 1])

    # To visualize...
    # - start
    # - nodes in the graph
    # - edges in the graph

    rrtTree = {}
    rrtTree["goal"] = q_goal
    rrtTree["vertices"] = vertices[: i_vertex_to_fill + 1]
    rrtTree["parent_indices"] = parent_indices[: i_vertex_to_fill + 1]
    rrtTree["edges"] = edges
    rrtTree["final_path"] = final_path

    plot_rrt_tree(rrtTree)

    return path_found, path
