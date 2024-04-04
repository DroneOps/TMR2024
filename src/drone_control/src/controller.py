#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from drone_control.msg import Custom


class Controller():
    def __init__(self):
        rospy.init_node("controller_node", anonymous=False)
        self.drone_state =  State()
        self.aruco_sub = rospy.Subscriber("/arucoId", Custom, self.aruco_state_callback)
        self.drone_state_sub = rospy.Subscriber("mavros/state", State, self.drone_state_callback)

    def aruco_callback(self, msg):
        print("Aruco id: " + str(msg.id) + " ~ Distance: " + str(msg.distance))

    def drone_state_callback(self, msg):
        self.drone_state = msg


    def run(self):
        print("Running...")
        while not rospy.is_shutdown() and self.drone_state.armed == True:
            print("...")

if __name__ == '__main__':
    controller = Controller()
    controller.run()
    