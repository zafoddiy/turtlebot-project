#!/usr/bin/env python3

"""
@author: Elias Sepp
@date: 04.11.24
"""

import math
import rclpy
import time
from rclpy.node import Node
from rclpy.time import Duration, Time
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped, Point
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion


class PDController(Node):
    def __init__(self):
        
        super().__init__('controller')

        # Wait for run other nodes
        time.sleep(5)

        # variables for error change rate calculation
        self.start_time = self.get_clock().now()
        self.last_odom_time = self.get_clock().now()

        self.sub_odom = self.create_subscription(
            Odometry, '/diff_cont/odom', self.onOdom, 10)
        self.sub_goal = self.create_subscription(
            PoseStamped, '/goal_pose', self.onGoal, 10)

        self.vel_cmd_pub = self.create_publisher(
            Twist, '/diff_cont/cmd_vel', 10)
        self.pub_viz = self.create_publisher(Marker, "waypoints", 10)

        self.marker_frame = "odom"

        self.vel_cmd_msg = Twist()
        self.vel_cmd = np.array([0.0, 0.0])

        self.pos = np.array([0.0, 0.0])
        self.pos_diff = np.array([0.0, 0.0])
        self.theta = 0.0
        self.th_diff = 0.0

        self.error = np.array([0.0, 0.0])

        self.error_change_rate = np.array([0.0, 0.0])

        self.error_integral = np.array([0.0, 0.0])

        # Load params from parameter server
        self.waypoints = []
        self.waypoints_x = []
        self.waypoints_y = []
        read_waypoints_x = self.declare_parameter('waypoints_x', [0.0]).value
        read_waypoints_y = self.declare_parameter('waypoints_y', [0.0]).value
        self.Kp = np.array(self.declare_parameter('Kp', [0.0, 0.0]).value)
        self.Kd = np.array(self.declare_parameter('Kd', [0.0, 0.0]).value)
        self.Ki = np.array(self.declare_parameter('Ki', [0.0, 0.0]).value)

        # Identify the waypoints parameter is correcly load or not
        if (read_waypoints_x == [0.0]) and (read_waypoints_y == [0.0]):
            self.get_logger().error("!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!")
            self.get_logger().error("Parameters not loaded correctly")
            self.get_logger().error("Please check your launch file to load yaml file correctly")
            self.get_logger().error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        if (read_waypoints_x) and (read_waypoints_y):
            for i in range(len(read_waypoints_x)):
                self.waypoints.append(
                    [read_waypoints_x[i], read_waypoints_y[i]])
        self.waypoints = np.array(self.waypoints)

        self.distance_margin = self.declare_parameter(
            'distance_margin', 0.05).value
        self.target_angle = self.wrapAngle(self.declare_parameter(
            'target_angle', 0.0).value)

        # Check Params
        self.get_logger().info(f'Waypoints: \n {self.waypoints}')
        self.get_logger().info(f'Kp: {self.Kp}')
        self.get_logger().info(f'Ki: {self.Ki}')
        self.get_logger().info(f'Kd: {self.Kd}')
        self.get_logger().info(f'Start Time: {self.start_time}')

    def wrapAngle(self, angle):
        """
        Helper function that returns angle wrapped between +- Pi.
        @param: self
        @param: angle - angle to be wrapped in [rad]
        @result: returns wrapped angle -Pi <= angle <= Pi
        """

        if angle > math.pi:
            corr_angle = angle - 2 * math.pi
        elif angle < -math.pi:
            corr_angle = angle + 2 * math.pi
        else:
            corr_angle = angle
        # return the angle between -PI and PI
        return corr_angle

    def control(self):
        """
        Takes the errors and calculates velocities from it, according to
        PD control algorithm.
        @param: self (errors got using "calculateError" function)
        @result: sets the values in self.vel_cmd
        """
        self.vel_cmd[0] = self.Kp[0] * self.error[0] + self.Kd[0] * \
            self.error_change_rate[0] + self.Ki[0] * self.error_integral[0]
        self.vel_cmd[1] = self.Kp[1] * self.error[1] + self.Kd[1] * \
            self.error_change_rate[1] + self.Ki[1] * self.error_integral[1]

    def publishWaypoints(self):
        """
        Publishes the list of waypoints, so RViz can see them.
        @param: self
        @result: publish message
        """
        marker = Marker()
        marker.header.frame_id = self.marker_frame

        marker.type = marker.SPHERE_LIST
        marker.action = marker.ADD

        marker.scale.x = self.distance_margin
        marker.scale.y = self.distance_margin
        marker.scale.z = self.distance_margin
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        marker.pose.orientation.w = 1.0

        marker.points = [Point(x=waypoint[0], y=waypoint[1])
                         for waypoint in self.waypoints]

        self.pub_viz.publish(marker)

    def calculateError(self):
        """
        Calculate lateral error to first waypoint and heading error between
        line to waypoint and robot heading.
        @param: self
        @result: updates self.error, self.error_change_rate, self.th_diff and
                 self.pos_diff
        """
        
        try:
            self.pos_diff = [self.waypoints[0][0] -
                             self.pos[0], self.waypoints[0][1] - self.pos[1]]
            theta_goal = np.arctan2(self.waypoints[0][1] -
                                    self.pos[1], self.waypoints[0][0] - self.pos[0])
            self.th_diff = theta_goal - self.wrapAngle(self.theta)
        except:
            self.pos_diff = [0.0, 0.0]
            self.th_diff = 0.0
        dist_error = math.sqrt(
            math.pow(self.pos_diff[0], 2) + math.pow(self.pos_diff[1], 2))
        self.error = [dist_error, self.th_diff]
        self.error_change_rate = [dist_error / self.dt, self.th_diff / self.dt]
        self.error_integral = [dist_error * self.dt, self.th_diff * self.dt]

    def isWaypointReached(self):
        """
        check if waypoint is reached based on user define threshold
        @param: self
        @result: if waypoint is reached that waypoint is popped from
                 waypoint list and True is returned,
                 otherwise False is returned.
        """

        if self.waypoints.size != 0:
            if (self.error[0] < self.distance_margin):
                self.waypoints = np.delete(self.waypoints, 0, axis=0)
                return True

        return False

    def onOdom(self, odom_msg):
        """
        Handles incoming odometry updates (callback function).
        @param: self
        @param odom_msg - odometry geometry message
        @result: update of relevant vehicle state variables
        """

        self.pos[0] = odom_msg.pose.pose.position.x
        self.pos[1] = odom_msg.pose.pose.position.y

        orientation_q = odom_msg.pose.pose.orientation
        orientation_list = [orientation_q.x,
                            orientation_q.y, orientation_q.z, orientation_q.w]
        self.theta = euler_from_quaternion(orientation_list)[2]

        now_odom_time = rclpy.time.Time.from_msg(odom_msg.header.stamp)

        dt_tmp = (now_odom_time - self.last_odom_time).to_msg()
        self.dt = float(dt_tmp.sec + dt_tmp.nanosec/1e9)
        self.last_odom_time = now_odom_time

        # Calculate error between current pose and next waypoint position
        self.calculateError()

        # Check reaching waypoint or not
        if self.isWaypointReached():
            self.get_logger().info(
                "Reached waypoint!\nFuture waypoint list: "
                + str(self.waypoints))
            self.calculateError()  # Update error with new target waypoint

        # Calculate velocity command using PID control
        self.control()

        # publish velocity commands
        self.vel_cmd_msg.linear.x = self.vel_cmd[0]
        self.vel_cmd_msg.angular.z = self.vel_cmd[1]
        self.vel_cmd_pub.publish(self.vel_cmd_msg)

        # Publish waypoints visualization
        self.publishWaypoints()

    def onGoal(self, goal_msg):
        """
        Handles incoming goal additions (callback function).
        @param: self
        @param goal_msg - goal message
        @result: update of goal waypoint based on user input
        """
        new_waypoint = np.array(
            [goal_msg.pose.position.x, goal_msg.pose.position.y])
        self.waypoints = np.vstack((self.waypoints, new_waypoint))
        self.get_logger().info(
            f"New waypoint received: {self.waypoints[0]}")


def main(args=None):
    rclpy.init(args=args)
    controller = PDController()
    rclpy.spin(controller)


if __name__ == "__main__":
    main()
