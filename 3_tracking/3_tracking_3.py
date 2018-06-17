# Blatt 3 "3_tracking"
# Aufgabe 3: RANSAC Outlier Filtering
# Steffen Burlefinger (859077)

import numpy as np
import math
import sys
from random import randint
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class RansacPointGenerator:
    """generates a set points - linear distributed + a set of outliers"""
    def __init__(self, numpointsInlier, numpointsOutlier):
        self.numpointsInlier = numpointsInlier
        self.numpointsOutlier = numpointsOutlier
        self.points = []

        pure_x = np.linspace(0, 1, numpointsInlier)
        pure_y = np.linspace(0, 1, numpointsInlier)

        noise_x = np.random.normal(0, 0.025, numpointsInlier)
        noise_y = np.random.normal(0, 0.025, numpointsInlier)

        outlier_x = np.random.random_sample((numpointsOutlier,))
        outlier_y = np.random.random_sample((numpointsOutlier,))

        points_x = pure_x + noise_x
        points_y = pure_y + noise_y

        points_x = np.append(points_x, outlier_x)
        points_y = np.append(points_y, outlier_y)

        self.points = np.array([points_x, points_y])


class Line:
    """helper class"""
    def __init__(self, a, b):
        # y = mx + b
        self.m = a
        self.b = b


class Ransac:
    """RANSAC class. """
    def __init__(self, points, threshold):
        self.points = points
        self.threshold = threshold
        self.best_model = Line(1, 0)
        self.best_inliers = []
        self.best_score = 1000000000
        self.current_inliers = []
        self.current_model = Line(1, 0)
        self.num_iterations = int(self.estimate_num_iterations(0.99, 0.5, 2))
        self.iteration_counter = 0

    def estimate_num_iterations(self, ransacProbability, outlierRatio, sampleSize):
        """
        Helper function to generate a number of generations that depends on the probability
        to pick a certain set of inliers vs. outliers.
        See https://de.wikipedia.org/wiki/RANSAC-Algorithmus for more information
        :param ransacProbability: std value would be 0.99 [0..1]
        :param outlierRatio: how many outliers are allowed, 0.3-0.5 [0..1]
        :param sampleSize: 2 points for a line
        :return:
        """
        return math.ceil(math.log(1-ransacProbability) / math.log(1-math.pow(1-outlierRatio, sampleSize)))

    def estimate_error(self, p, line):
        """
        Compute the distance of a point p to a line y=mx+b
        :param p: Point
        :param line: Line y=mx+b
        :return:
        """
        return math.fabs(line.m * p[0] - p[1] + line.b) / math.sqrt(1 + line.m * line.m)

    def find_line_model(self, point_a, point_b):
        """
        https://salzis.wordpress.com/2014/06/10/robust-linear-model-estimation-using-ransac-python-implementation/
        find a line model for the given points
        :param points selected points for model fitting
        :return line model
        """
        # add some noise to avoid division by zero

        # m = (y2 -y1) / (x2 - x1)
        m = (point_a[0] - point_b[0]) / (point_a[1] - point_b[1] + sys.float_info.epsilon)
        # y = xm + b -> b = y -xm
        b = point_a[1] - m * point_a[0]

        return m, b

    def get_point_at_index(self, index):
        return self.points[0][index], self.points[1][index]

    def step(self, iter):
        """
        Run the ith step in the algorithm. Collects self.currentInlier for each step.
        Sets if score < self.bestScore
        self.bestModel = line
        self.bestInliers = self.currentInlier
        self.bestScore = score
        :param iter: i-th number of iteration
        :return:
        """
        self.current_inliers = []
        score = 0
        idx = 0

        # sample two random points from point set
        length = len(self.points[0])
        rnd_idx = np.random.random_integers(0, length-1, 2)
        # print("rnd idx's: ", rnd_idx)

        # print("\n", index_point_a, index_point_b)
        m, b = self.find_line_model(self.get_point_at_index(rnd_idx[0]), self.get_point_at_index(rnd_idx[1]))
        # print("\nmy model m: ", m, "b: ", b)

        line = Line(m, b)
        # print("testing line: y = x"+str(m)+" + "+str(b))

        # loop over all points
        # compute error of all points and
        # add to inliers if err smaller than threshold update score,
        # otherwise add error/threshold to score
        for i in range(0, len(self.points[0])):
            point = [self.points[0][i], self.points[1][i]]
            # print(point)
            err = self.estimate_error(point, line)
            # print("err: ", err)
            if err < self.threshold:
                self.current_inliers.append(i)
                # score = err
            else:
                score = score + err / self.threshold

        # print(iter, "  :::::::::: score:     ", score, " model: ", m, b)

        # if score < self.bestScore: update the best model/inliers/score
        # please do look at resources in the internet :)
        if score < self.best_score:
            self.best_model = line
            self.best_inliers = self.current_inliers
            self.best_score = score

        # print(iter, "  :::::::::: bestscore: ", self.best_score, " bestModel: ", self.best_model.m, self.best_model.b)

    def run(self):
        """
        run RANSAC for a number of iterations
        :return:
        """
        for i in range(0, self.num_iterations):
            self.step(i)


configs = [(231, 100, 15, 0.05, '100+15@0.05'), (232, 100, 45, 0.05, '100+45@0.05'),
           (233, 100, 75, 0.05, '100+75@0.05'), (234, 100, 45, 0.0005, '100+45@0.0005'),
           (235, 100, 45, 0.01, '100+45@0.01'), (236, 100, 45, 0.5, '100+45@0.5')]


plt.figure(1, figsize=[14, 8])

for conf in configs:
    plt.subplot(conf[0])
    plt.title(conf[4])
    rpg = RansacPointGenerator(conf[1], conf[2])
    ransac = Ransac(rpg.points, conf[3])
    ransac.run()
    plt.plot(rpg.points[0, :], rpg.points[1, :], 'ro')
    m = ransac.best_model.m
    b = ransac.best_model.b
    plt.plot([0, 1], [m * 0 + b, m * 1 + b], color='k', linestyle='-', linewidth=2)
    plt.axis([0, 1, 0, 1])

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.show()

# csv_data = np.loadtxt('resources/80m.csv', delimiter=';', skiprows=1)
# raw_data = csv_data[:, [0, 3]]  # only z values
# rpg.points = np.array([raw_data[:, [0]], raw_data[:, [1]]])
# plt.plot([0, max(raw_data[:, [0]])], [m*0 + b, m*1+b], color='k', linestyle='-', linewidth=2)
# plt.axis([0, max(raw_data[:, [0]]), min(raw_data[:, [1]]), max(raw_data[:, [1]])])
# plt.show()