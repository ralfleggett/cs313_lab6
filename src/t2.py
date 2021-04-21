#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import threading
from copy import copy

import rospy
import tf

from controller import Controller

from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose2D

from map import Map


class Turtlebot3():
    def __init__(self):
        rospy.init_node("turtlebot3_move_square")
        rospy.loginfo("Press Ctrl + C to terminate")

        # Particle set [(x,y,th)]
        self.N = 1000
        self.particles = np.zeros((self.N, 3))
        self.particles[:,0] = -0.7
        self.weights = np.repeat((1.0 / self.N), self.N)

        # Mean & covariance
        self.mean = np.zeros(3, dtype=np.float)
        self.mean[0] = -0.7
        self.covar = np.zeros((3,3), dtype=np.float)

        # History of particle values for visualisation
        self.particle_history = []

        # waypoints
        self.waypoints = np.array([
            [-0.7, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -2.0, -np.pi/2],
            [2.0, -2.0, 0.0],
            [2.0, 0.0, np.pi/2],
            [1.0, 0.0, -np.pi],
            [1.0, 1.0, np.pi/2],
            [2.0, 1.0, 0.0]
        ])

        # initialise controllers for theta and distance
        self.th_controller = Controller(P=1, D=0.8, rate=10)
        self.d_controller = Controller(P=1, D=0.8, rate=10)

        # set up publisher
        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.rate = rospy.Rate(10)

        # subscribe to odometry
        self.pose = Pose2D()
        self.logging_counter = 99 # Make subscriber log first point
        self.trajectory = list()
        self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback)

        # LIDAR scan variables
        self.scan_indices = np.arange(0, 360, 10)
        self.scan_covar = np.diag([0.15] * len(self.scan_indices))

        # create map object
        self.map = Map("src/lab6/src/map.pgm", (np.pi / 180) * self.scan_indices)

        self.run()

    def run(self):
        """ Main method for tracking self.waypoints """
        # Wait for the first pose measurement to come in
        time.sleep(1)
        for i, waypoint in enumerate(self.waypoints[1:]):

            # Turn until robot yaw is correct
            turn_angle = self.turn(waypoint[2])
            if turn_angle != 0:
                self.update_particles_turn(turn_angle)
                
            # Check whether we're moving in x or y direction
            x_dir = waypoint[0] - self.waypoints[i][0] != 0
            dir_sign = 1 if sum(waypoint[:2] - self.waypoints[i][:2]) > 0 else -1
            # Find distance we need to travel
            if x_dir:
                total_dist = abs(waypoint[0] - self.waypoints[i][0])
            else:
                total_dist = abs(waypoint[1] - self.waypoints[i][1])

            dist_moved = 0
            # Threshold used for reliability
            while abs(dist_moved - total_dist) > 0.01:
                # Calculate the destination for this loop
                step = min(0.5, total_dist - dist_moved)
                if x_dir:
                    change = np.array([dist_moved + step, 0]) 
                else:
                    change = np.array([0, dist_moved + step])
                destination = self.waypoints[i][:2] + dir_sign * change
                # Move to the destination, record how far we actually moved
                dist = self.move_forward(destination, x_dir)
                self.update_particles_forward_movement(dist)
                # dist_moved uses step rather than dist to ensure we satisfy
                # the loop condition
                dist_moved += step

        # Visualise particles
        self.visualise()
        
        rospy.loginfo("Finished execution!")

    def update_particles_forward_movement(self, d):
        """ Update mean, covar, particles after moving d metres """
        # Sample e and f
        e = np.random.normal(scale=2e-3, size=self.N)
        f = np.random.normal(scale=2e-3, size=self.N)

        # Apply process model
        self.particles[:,0] += (d + e) * np.cos(self.particles[:,2])
        self.particles[:,1] += (d + e) * np.sin(self.particles[:,2])
        self.particles[:,2] += f

        # Apply measurement model
        self.update_particles_measurement()

        # Save only the first 100 or we'll run out of memory
        self.particle_history.append(np.copy(self.particles))
        self.update_mean_covar(dist=d)

    def update_particles_turn(self, a):
        """ Update mean, covar, particles after turning a radians """
        # Sample g
        g = np.random.normal(scale=2e-3, size=self.N)

        # Apply process model
        self.particles[:,2] += a + g

        # Apply measurement model
        self.update_particles_measurement()

        # Save only the first 100 or we'll run out of memory
        self.particle_history.append(np.copy(self.particles))
        self.update_mean_covar(turn=a)

    def update_particles_measurement(self):
        """ Performs a measurement update using a laserscan """
        # Get a LaserScan message
        msg = rospy.wait_for_message("scan", LaserScan)
        measurement = np.array(msg.ranges)[self.scan_indices]
        # Ensure measurements in range
        measurement[measurement > 3.5] = 3.5
        measurement[measurement < 0.12] = 0.12

        # Generate expected measurements
        expected_measurements = []
        for particle in self.particles:
            z_hat = self.map.compute_measurement(*particle)
            expected_measurements.append(z_hat)
        
        # Compute likelihoods
        weights = []
        d = len(self.scan_indices)
        denom = 1 / ((2 * np.pi)**(d/2) * np.sqrt(np.linalg.det(self.scan_covar)))
        if denom <= 0:
            rospy.logwarn("Weight calculation denominator less than zero!")
        for m_hat in expected_measurements:
            diff = np.zeros((d, 1))
            diff[:, 0] = m_hat - measurement
            num = np.exp(-0.5 * (diff.T.dot(np.linalg.pinv(self.scan_covar)).dot(diff)))
            p = (num * denom)[0][0]
            if p < 0:
                rospy.logwarn("Probability < 0!")
            weights.append(p)

        # Normalise weights
        weights = np.array(weights)
        if np.sum(weights) == 0:
            rospy.logwarn("No particle was of any use!")
        weights = weights / np.sum(weights)

        self.resample(weights)

    def resample(self, weights):
        """ Performs importance resampling on self.particles """
        indicies = np.random.choice(
            len(self.particles),
            size=len(self.particles),
            replace=True,
            p=weights
        )
        self.particles = self.particles[indicies]
        # Since self.weights was never changed, there is no need to set the 
        # weights to be the same again

    def update_mean_covar(self, dist=0.0, turn=0.0):
        """ Update and report mean, covariance """
        self.mean = np.sum(self.weights.reshape((self.N, 1)) * self.particles, axis=0)
        # Normalise yaw angle
        self.mean[2] = self.normalise_angle(self.mean[2])
        self.covar = np.cov(self.particles, aweights=self.weights, rowvar=False)
        rospy.loginfo("After moving " + str(dist) + "m and turning " + str(turn) \
            + "rad: \nMean: "+str(self.mean) + "\nCovariance: \n"+str(self.covar))

        # Get ground truth state
        msg = rospy.wait_for_message("ground_truth/state", Odometry)
        pos = msg.pose.pose.position
        yaw = self.quat2yaw(msg.pose.pose.orientation)
        gt_state = np.array([pos.x, pos.y, yaw])

        # Compare to ground truth
        pos_diff = np.linalg.norm(self.mean[:2] - gt_state[:2])
        yaw_diff = min(abs(self.mean[2] - gt_state[2]), abs(self.mean[2] + gt_state[2]))
        rospy.loginfo("\nGround truth: " + str(gt_state) + "\n Position error: " + \
            str(pos_diff) + "; Yaw error: "+str(yaw_diff))

    def visualise(self):
        """ Visualise self.particle_history """
        _, ax = plt.subplots(1)
        ax.set_aspect('equal')

        # Plot the trajectory
        plt.plot(np.array(self.trajectory)[:, 0], np.array(self.trajectory)[:, 1])

        # Plot the particles
        colours = "bgrcmykw"
        for i, particles in enumerate(self.particle_history):
            ax.scatter(particles[:,0], particles[:,1], c=colours[i % 8], zorder=(i+1))
        # self.particle_history = []
        plt.xlim(-2, 3)
        plt.ylim(-3, 2)
        plt.show()

    def turn(self, th_star):
        """ Turns the robot so its yaw is equal to th_star """
        self.th_controller.setPoint(th_star)

        # Store initial value to calc turn_angle
        theta_0 = copy(self.pose.theta)

        # Use a tolerance of 0.001 radians as cannot get it exactly aligned
        while abs(th_star - self.pose.theta) > 0.001:
            theta = self.pose.theta
            # Deal with discontinuity at -pi, pi
            if th_star - theta < -np.pi:
                theta = theta - 2 * np.pi 
            elif th_star - theta >= np.pi:
                theta = theta + 2 * np.pi
            w = self.th_controller.update(theta)
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = w 
            self.vel_pub.publish(msg)
            self.rate.sleep()
        # Tell the robot to stop
        self.stop()

        # Calculate turn_angle
        turn_angle = self.normalise_angle(self.pose.theta - theta_0)
        return turn_angle

    def find_desired_yaw(self, waypoint):
        """ Finds the desired yaw based off of self.pose given a waypoint """
        th_star = np.arctan2(waypoint[1] - self.pose.y, waypoint[0] - self.pose.x)
        # Normalise to [-pi, pi)
        th_star = self.normalise_angle(th_star)
        return th_star

    def move_forward(self, waypoint, x_dir):
        """ Move to the waypoint then stop """
        # Find desired x / y position
        d_star = waypoint[0] if x_dir else waypoint[1]
        self.d_controller.setPoint(d_star)

        # Store initial position
        p_0 = np.array([copy(self.pose.x), copy(self.pose.y)])

        while abs(d_star - (self.pose.x if x_dir else self.pose.y)) > 0.001:
            # Find desired yaw
            th_star = self.find_desired_yaw(waypoint)
            self.th_controller.setPoint(th_star)
            # Find w control input
            theta = self.pose.theta
            # Deal with discontinuity at -pi, pi
            if th_star - theta < -np.pi:
                theta = theta - 2 * np.pi 
            elif th_star - theta >= np.pi:
                theta = theta + 2 * np.pi
            w = self.th_controller.update(theta)
            v = abs(self.d_controller.update((self.pose.x if x_dir else self.pose.y)))
            msg = Twist()
            msg.linear.x = v
            msg.angular.z = w
            self.vel_pub.publish(msg)
            self.rate.sleep()
        # Tell the robot to stop
        self.stop()

        # Calculate distance moved
        p_final = np.array([self.pose.x, self.pose.y])
        return np.linalg.norm(p_final - p_0)

    def stop(self):
        """ Publishes a stop message """
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.vel_pub.publish(msg)
        self.rate.sleep()

    def odom_callback(self, msg):
        # get pose = (x, y, theta) from odometry topic
        quaternion = [msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,\
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        (_, _, yaw) = tf.transformations.euler_from_quaternion(quaternion)
        self.pose.theta = yaw
        self.pose.x = msg.pose.pose.position.x
        self.pose.y = msg.pose.pose.position.y

        # logging once every 100 times (Gazebo runs at 1000Hz; we save it at 10Hz)
        self.logging_counter += 1
        if self.logging_counter == 100:
            self.logging_counter = 0
            # save trajectory
            self.trajectory.append([self.pose.x, self.pose.y])

    def quat2yaw(self, quat):
        """ Converts Quarternion to yaw angle in [-pi, pi) range """
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        return self.normalise_angle(yaw)

    def normalise_angle(self, th):
        """ Returns an angle in range [-pi, pi) """
        while th >= np.pi:
            th = th - 2 * np.pi
        while th < -np.pi:
            th = th + 2 * np.pi
        return th
    
if __name__ == '__main__':
    try:
        robot = Turtlebot3()
    except rospy.ROSInterruptException:
        rospy.loginfo("Action terminated.")
