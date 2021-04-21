#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import copy

import rospy
import tf

from controller import Controller

from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D


class Turtlebot3():
    def __init__(self):
        rospy.init_node("turtlebot3_move_square")
        rospy.loginfo("Press Ctrl + C to terminate")

        # Particle set [(x,y,th)]
        self.N = 100
        self.particles = np.zeros((self.N, 3))
        self.weights = np.repeat((1.0 / self.N), self.N)

        # Mean & covariance
        self.mean = np.zeros(3, dtype=np.float)
        self.covar = np.zeros((3,3), dtype=np.float)

        # History of particle values for visualisation
        self.particle_history = []
        self.particle_history_start_idx = 0

        # waypoints
        self.waypoints = np.array([[0,0], [4, 0], [4, 4], [0, 4], [0, 0]])

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

        self.run()

    def run(self):
        """ Main method for tracking self.waypoints """
        # Wait for the first pose measurement to come in
        time.sleep(1)
        for i, waypoint in enumerate(self.waypoints[1:]):

            # Turn until robot yaw is correct
            turn_angle = self.turn(waypoint)
            if turn_angle != 0:
                self.update_particles_turn(turn_angle)

            # Check whether we're moving in x or y direction
            x_dir = waypoint[0] - self.waypoints[i][0] != 0
            dir_sign = 1 if sum(waypoint - self.waypoints[i]) > 0 else -1

            # Move forward 4m
            for j in range(8):
                change = np.array([0.5, 0]) if x_dir else np.array([0, 0.5])
                destination = self.waypoints[i] + (j+1) * dir_sign * change
                dist = self.move_forward(destination, x_dir)
                self.update_particles_forward_movement(dist)

            # Visualise particles along the side of the square we just traversed
            self.visualise()
        
        # Finally turn back to theta = 0. Use the waypoint [4, 0] to make it turn
        turn_angle = self.turn([4, 0])
        self.update_particles_turn(turn_angle)
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

        self.particle_history.append(np.copy(self.particles))
        self.update_mean_covar(dist=d)

    def update_particles_turn(self, a):
        """ Update mean, covar, particles after turning a radians """
        # Sample g
        g = np.random.normal(scale=2e-3, size=self.N)

        # Apply process model
        self.particles[:,2] += a + g

        self.particle_history.append(self.particles)
        self.update_mean_covar(turn=a)

    def update_mean_covar(self, dist=0.0, turn=0.0):
        """ Update and report mean, covariance """
        self.mean = np.sum(self.weights.reshape((self.N, 1)) * self.particles, axis=0)
        # Normalise yaw angle
        self.mean[2] = self.normalise_angle(self.mean[2])
        self.covar = np.cov(self.particles, aweights=self.weights, rowvar=False)
        rospy.loginfo("After moving " + str(dist) + "m and turning " + str(turn) \
            + "rad: \nMean: \n"+str(self.mean) + "\nCovariance: \n"+str(self.covar))

    def visualise(self):
        """ Visualise self.particle_history """
        _, ax = plt.subplots(1)
        ax.set_aspect('equal')

        # Plot the trajectory
        plt.plot(np.array(self.trajectory)[:, 0], np.array(self.trajectory)[:, 1])

        # Plot the particles
        colours = "bgrcmykw"
        for i, particles in enumerate(self.particle_history[self.particle_history_start_idx:]):
            ax.scatter(particles[:,0], particles[:,1], c=colours[i % 8], zorder=(i+1))
        # This variable ensures we only plot the particles from the last few movements, rather
        # than all movements recorded
        self.particle_history_start_idx = len(self.particle_history)
        plt.xlim(-2, 6)
        plt.ylim(-2, 6)
        plt.show()

    def find_desired_yaw(self, waypoint):
        """ Finds the desired yaw based off of self.pose given a waypoint """
        th_star = np.arctan2(waypoint[1] - self.pose.y, waypoint[0] - self.pose.x)
        # Normalise to [-pi, pi)
        th_star = self.normalise_angle(th_star)
        return th_star

    def turn(self, waypoint):
        """ Turns the robot so its yaw is equal to th_star """
        # Find desired yaw
        th_star = self.find_desired_yaw(waypoint)
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