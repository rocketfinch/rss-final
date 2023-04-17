import numpy as np
from math import pi
import sys
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import Polygon2

from rrt import *


def create_map_with_obstacles(
    range_start: float, range_end: float, obst_list: list
):
    map = rtb.PolygonMap(workspace=[range_start, range_end])
    for obst in obst_list:
        map.add(obst)
    return map


def plot_pose(pose: np.ndarray, name: str):
    length = 0.4
    dx = np.cos(pose[2]) * length
    dy = np.sin(pose[2]) * length
    plt.arrow(pose[0], pose[1], dx, dy, width=0.05)


def plot_car(car: rtb.Bicycle, pose: np.ndarray):
    car_poly = car.polygon(pose).vertices()
    plt.fill(
        car_poly[0, :],
        car_poly[1, :],
        facecolor="lightsalmon",
        edgecolor="orangered",
        linewidth=3,
        alpha=0.5,
    )


def plot_path(path: np.ndarray, plot_arrows: bool = True, subsample=4):
    plt.plot(path[:, 0], path[:, 1], linewidth=4, alpha=0.7)

    if plot_arrows:
        arrow_size = 0.2
        arrows = path[::subsample, :]
        for i in range(arrows.shape[0]):
            pose = arrows[i, :]
            # print(pose)
            length = arrow_size
            dx = np.cos(pose[2]) * length
            dy = np.sin(pose[2]) * length
            plt.arrow(
                pose[0],
                pose[1],
                dx,
                dy,
                head_width=arrow_size,
                head_length=length,
                length_includes_head=True,
                alpha=0.35,
                edgecolor=None,
                linewidth=None,
                facecolor="green",
            )


def plot_map(map: rtb.PolygonMap):
    for polygon in map.polygons:
        poly = polygon.vertices()

        plt.fill(
            poly[0, :],
            poly[1, :],
            facecolor="grey",
            edgecolor="black",
            linewidth=3,
            alpha=0.85,
        )

    plt.xlim([map.workspace[0], map.workspace[1]])
    plt.ylim([map.workspace[2], map.workspace[3]])


def main():

    obstacles = list()
    # obstacles.append(np.transpose(np.array([[5, 4], [5, 6], [6, 6], [6, 4]])))
    # obstacles.append(
    #     np.transpose(np.array([[5, 3], [5, -4], [6, -4], [6, 3]]))
    # )

    polygon_map = create_map_with_obstacles(
        range_start=-1, range_end=10, obst_list=obstacles
    )

    q_start = np.array([8, 2, pi / 2])
    q_goal = np.array([3, 7, pi / 2])

    l, w = 1, 0.5
    car_shape_array = np.array(
        [
            [-l / 2, -w / 2],
            [-l / 2, w / 2],
            [l / 2, w / 2],
            [l / 2, -w / 2],
        ]
    )

    car_shape = Polygon2(np.transpose(car_shape_array))
    car = rtb.Bicycle(polygon=car_shape)

    # plot(polygon_map, q_start, q_goal, car)
    # print(car)

    dubins = rtb.DubinsPlanner(curvature=1, stepsize=0.1)
    dubins.plan()
    path, status = dubins.query(start=q_start, goal=q_goal)
    # dubins.plot(path=path, block=True)
    # print(path)
    # print("dubin path length:", status.length)

    # plot_pose(q_start, "start")
    # plot_pose(q_goal, "end")
    plot_car(car, q_goal)
    # plot_path(path)

    rtt_status, rtt_path = rrt_path_plan(polygon_map, car, q_start, q_goal)
    # print(rtt_path)
    # plot_path(rtt_path)

    # print("dubins path shape:", path.shape)
    # print("rtt path shape:", rtt_path.shape)

    plot_map(polygon_map)

    plt.show()

    # q_target = generate_qrandom_free(polygon_map, car)
    # print("q_start shape:", q_start.shape)
    # print("q_goal shape:", q_goal.shape)
    # print("q_target:", q_target)
    # print("q_target shape:", q_target.shape)

    # print(
    #     "qpath from start to random target:",
    #     calculate_q_path_dubins(dubins, q_start, q_target),
    # )

    # nbrs = knnsearch(path, q_target, 2)

    # print("knnsearch around target", nbrs)
    # print(nbrs[0][0])

    # print(
    #     "is path in collison (should say True):",
    #     path_in_collision(path, polygon_map, car),
    # )

    # print(
    #     "is (first ten steps of) path in collison (should say False):",
    #     path_in_collision(path[:10, :], polygon_map, car),
    # )

    # print(
    #     "is collison:",
    #     polygon_map.iscollision(car.polygon(q_goal)),
    # )

    # rrt = rtb.RRTPlanner(map=map, vehicle=car, npoints=50)
    # print("flag1")
    # rrt.plan(goal=q_goal, animate=False)
    # print("flag2")
    # path, status = rrt.query(start=q_start)
    # print("flag3")
    # rrt.plot(path=path, background=True, block=True)
    # print("Path:", path)
    # print("Status:", status)

    # car.run(animate=True)


if __name__ == "__main__":
    main()

    # try:
    #     main()
    #     plt.ion()

    #     while True:
    #         plt.show()
    #         break
    # except KeyboardInterrupt:
    #     print("")
    #     sys.exit(0)
