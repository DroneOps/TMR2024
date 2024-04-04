import numpy as np
import time
import itertools
import cv2


ARUCO_DICT = {
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


def aruco_display(corners, ids, rejected, image):
	if len(corners) > 0:
		
		ids = ids.flatten()
		
		for (markerCorner, markerID) in zip(corners, ids):
			
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			print("[Inference] ArUco marker ID: {}".format(markerID))
			
	return image

def medir_distancia_entre_marcadores(corners, ids):
    if ids is not None and len(ids) > 1:  # Asegúrate de que hay al menos 2 marcadores detectados
        # Calcula los centros de los dos primeros marcadores
        centro_1 = np.mean(corners[0], axis=1).flatten()
        centro_2 = np.mean(corners[1], axis=1).flatten()

        # Calcula la distancia Euclidiana entre los centros
        distancia = np.linalg.norm(centro_1 - centro_2)

        print(f"Distancia entre marcador {ids[0][0]} y {ids[1][0]}: {distancia} píxeles")
        return distancia
    else:
        print("No se detectaron suficientes marcadores para medir distancia.")
        return None


aruco_type = "DICT_5X5_1000"

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters_create()


# Matriz de calibración de la cámara y coeficientes de distorsión
# Reemplaza estos valores con los resultados de tu calibración
cameraMatrix = np.array([[1000, 0, 640],
                         [0, 1000, 360],
                         [0, 0, 1]], dtype="double")
distCoeffs = np.array([0.1, -0.1, 0, 0, 0])

# Tamaño del lado del marcador ArUco en centímetros
markerLength = 8  # Ajusta este valor según el tamaño real de tu marcador

# Inicia la captura de video
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecta los marcadores ArUco en la imagen
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    # Si se detectan marcadores, estima la pose
    if ids is not None:
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs)

        # Dibuja los marcadores detectados y sus IDs
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Para cada marcador, muestra la distancia de la cámara al marcador
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            # La distancia a la cámara es la componente Z del vector de traslación
            distancia_a_camara = tvec[0][2]  # La distancia está en las mismas unidades que el tamaño del marcador

            # Calcula la posición para mostrar el texto (usando el centro del marcador)
            centro_marcador = np.mean(corners[i], axis=1).flatten().astype(int)
            texto_posicion = (centro_marcador[0], centro_marcador[1])

            # Muestra la distancia en el vídeo
            cv2.putText(frame, f"{distancia_a_camara:.2f} cm", texto_posicion, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Dibuja las poses de los marcadores
            #cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 10)

    # Muestra el resultado
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()


























#!/usr/bin/env python3
import rospy
import cv2

class ArucoNode():
	def __init__(self):
		rospy.init_node('arucoDetect_node', anonymous=False)
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
		self.arucoDict = cv2.aruco.getPredefinedDictionary()


	def detect(self):
		cap = cv2.VideoCapture('/dev/video4')
		while not rospy.is_shutdown() and cap.isOpened() :
			arucoDict = cv2.aruco.getPredefinedDictionary(self.dictionary[self.aruco_type])
			arucoParams = cv2.aruco.DetectorParameters_create()
			cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
			cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
			ret, img = cap.read()
			corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
			print('Marker id:{}'.format(ids))


if __name__ == '__main__':
	vision_node = ArucoNode()
	vision_node.detect()