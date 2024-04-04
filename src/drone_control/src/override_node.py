import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
import sys
import rosservice
from std_msgs.msg import Bool

class OverrideNode(object):
    def __init__(self):
        rospy.init_node('override_node', anonymous=False)
        self.drone_state = State()
        self.state_sub = rospy.Subscriber("mavros/state", State, callback = self.refresh_state)

    def refresh_state(self, msg):
        self.drone_status = msg
    
    def keyboard_checker(self):
        while not rospy.is_shutdown():
            input()
            rospy.loginfo("Toggle arm or dissarm")
            self.drone_state.armed = not self.drone_state.armed
            rosservice.call_service("/mavros/cmd/arming", self.drone_state)
        
if __name__ == '__main__':
    print("--OVERRIDER CONSOLE--")
    print(" ")
    print("Press any key to toggle arm drone")
    security_node = OverrideNode()
    security_node.keyboard_checker()