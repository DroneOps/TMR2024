#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from drone_control.msg import Custom
from std_msgs.msg import Float32

class ArucoNode():
	def __init__(self):
		rospy.init_node('arucoDetect_node', anonymous=False)
		self.publisher = rospy.Publisher("arucoId", Custom, queue_size=10)
		self.aruco_type = "DICT_5X5_1000"
		self.dictionary = {
			"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
			"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
			"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
			"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
			"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
			"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
			"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
			"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
			"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
			"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
			"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
			"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
			"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
			"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
			"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
			"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
			"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
			"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
			"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
			"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
			"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
		}
		self.arucoDict = cv2.aruco.getPredefinedDictionary(self.dictionary[self.aruco_type])
		self.arucoParams = cv2.aruco.DetectorParameters_create()
		self.markerLength = 8
		self.cameraMatrix = np.array([[1000, 0, 640],
                        			  [0, 1000, 360],
                                      [0, 0, 1]], dtype="double")
		self.distCoeffs = np.array([0.1, -0.1, 0, 0, 0])

	def detect(self):
		print("Reading aruco codes...")
		cap = cv2.VideoCapture('/dev/video4')
		while not rospy.is_shutdown():
			if cap.isOpened():
				cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
				cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
				ret, img = cap.read()
				corners, ids, rejected = cv2.aruco.detectMarkers(img, self.arucoDict, parameters=self.arucoParams)
				if ids is not None:
					rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, self.markerLength, self.cameraMatrix, self.distCoeffs)
					for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
						distancia_a_camara = tvec[0][2]
						msg = Custom()
						msg.id = ids[0][0]
						msg.distance = distancia_a_camara
						self.publisher.publish(msg)


if __name__ == '__main__':
	vision_node = ArucoNode()
	vision_node.detect()