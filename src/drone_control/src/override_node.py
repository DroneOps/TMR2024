import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
import sys
import rosservice

class OverrideNode(object):
    def __init__(self):
        rospy.init_node('override_node', anonymous=False)
        self.armed_state = False
        self.drone_status = State()
        self.state_sub = rospy.Subscriber("mavros/state", State, callback = self.refresh_state)

    def refresh_state(self, msg):
        self.drone_status = msg
    
    def keyboard_checker(self):
        while not rospy.is_shutdown():
            input()
            rospy.loginfo("Toggle arm or dissarm")
            self.armed_state = not self.armed_state
            rosservice.call_service("/mavros/cmd/arming", self.armed_state)
        
if __name__ == '__main__':
    print("--OVERRIDER CONSOLE--")
    print(" ")
    print("Press any key to toggle arm drone")
    security_node = OverrideNode()
    security_node.keyboard_checker()