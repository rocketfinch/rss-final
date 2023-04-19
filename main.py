import numpy as np
from math import pi
import sys
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import Polygon2

from rrt import *
from rrt_star import *


def create_map_with_obstacles(range_start: float, range_end: float, obst_list: list):
    map = rtb.PolygonMap(workspace=[range_start, range_end])
    for obst in obst_list:
        map.add(obst)
    return map


def plot_pose(pose: np.ndarray, name: str, color: str):
    length = 0.4
    dx = np.cos(pose[2]) * length
    dy = np.sin(pose[2]) * length
    plt.arrow(pose[0], pose[1], dx, dy, width=0.05, color=color)


def plot_car(car: rtb.Bicycle, pose: np.ndarray, facecolor: str, edgecolor: str):
    car_poly = car.polygon(pose).vertices()
    plt.fill(
        car_poly[0, :],
        car_poly[1, :],
        facecolor=facecolor,
        edgecolor=edgecolor,
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
                facecolor="darkgreen",
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


def main(seed=None):

    obstacles = list()
    # Leave obstacles commented below for unconstrained (no obstacle) scenario

    # Obstacles for the simple obstacle scenario
    # obstacles.append(np.transpose(np.array([[5, 4], [5, 6], [6, 6], [6, 4]])))
    # obstacles.append(np.transpose(np.array([[5, 2.5], [5, -4], [6, -4], [6, 2.5]])))

    # Obstacles for the parking-lot scenario
    # obstacle_centers = np.array([[1, 7], [5, 7], [7, 7], [0, 3], [2, 3], [4, 3], [6, 3]])
    # for i in range(obstacle_centers.shape[0]):
    #     size = 0.75
    #     obstacles.append(np.transpose(size * np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) + obstacle_centers[i]))

    polygon_map = create_map_with_obstacles(range_start=-1, range_end=10, obst_list=obstacles)

    # Starting position for simple obstacle scenario and unconstrained
    q_start = np.array([8, 2, pi / 2])

    # Starting position for parking lot scenario
    # q_start = np.array([5, 1, 0])

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

    plot_pose(q_start, "start", "green")
    plot_pose(q_goal, "end", "orangered")
    plot_car(car, q_start, facecolor="green", edgecolor="darkgreen")
    plot_car(car, q_goal, facecolor="lightsalmon", edgecolor="orangered")

    # RRT can also be run by turning off the flags in rrt_star.py
    # rtt_status, rtt_path = rrt_path_plan(polygon_map, car, q_start, q_goal)

    if seed:
        np.random.seed(seed)

    rtt_star_status, rtt_star_path, rtt_star_path_length = rrt_star_path_plan(polygon_map, car, q_start, q_goal)

    plot_map(polygon_map)

    plt.show()

    return rtt_star_status, rtt_star_path, rtt_star_path_length


if __name__ == "__main__":
    main()

    # file = open("asdf.txt", "w")

    # for seed in range(1, 11):
    #     runstatus, runpath, runpathlength = main(seed)
    #     file.write("STATUS: ")
    #     file.write(str(runstatus))
    #     file.write("\n")
    #     file.write("LENGTH: ")
    #     file.write(str(runpathlength))
    #     file.write("\n")

    # file.close()
