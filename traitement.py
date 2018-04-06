import subprocess
import time
import cv2
import numpy as np

from nao_brain_class import EDCNN

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
prog = 'C:\\Python27\\python.exe'
cmd = 'C:\\3A\\projet_mos\\NAO_FINAL_BOX\controle_nao.py'

# initialize the cascade
cascade_classifier = cv2.CascadeClassifier('haarcascade_classifiers/haarcascade_frontalface_default.xml')


def format_image(image):
    """
    Function to format frame
    """
    if len(image.shape) > 2 and image.shape[2] == 3:
        # determine whether the image is color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Image read from buffer
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
    if not len(faces) > 0:
        return None

    # initialize the first face as having maximum area, then find the one with max_area
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    face = max_area_face

    # extract ROI of face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    try:
        # resize the image so that it can be passed to the neural network
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.

    except Exception:
        print("----->Problem during resize")
        return None

    return image


def softMaxCouche(result):
    if result[0][6] < 0.6:
        result[0][6] = result[0][6] - 0.12
        result[0][:3] += 0.01
        result[0][4:5] += 0.04
    return result


# Initialize object of EDCNN class
print("ETAPE1: INITIALISATION DU PROGRAMME")
network = EDCNN()
print("ETAPE2: INITIALISATION DU RESEAUX NEURONES")
network.build_network()
print("-----------------------------------")

font = cv2.FONT_HERSHEY_SIMPLEX;
iteration = 0;
frameTMP = np.array([])

while True:

    frame = cv2.imread('C:\\3A\\projet_mos\\images\\image.jpg')
    if frame is not None and not np.array_equal(frame, frameTMP):
        time.sleep(2)
        frame = cv2.imread('C:\\3A\\projet_mos\\images\\image.jpg')
        frameTMP = np.array(frame);
        iteration += 1;
        print("-> Visage tester numero: ", iteration)

        frame_final = format_image(frame)

        result = network.predict(frame_final)
        #print(str(result[0]))
        if result is not None:
            # compute softmax probabilities
            result = softMaxCouche(result)
            # find the emotion with maximum probability and display it
            maxindex = np.argmax(result[0])
            print("-> Resultat: ", EMOTIONS[maxindex], ": ", round(result[0][maxindex] * 100, 3), "%")
            cv2.putText(frame, EMOTIONS[maxindex], (10, 360), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
            # controle NAO
            subprocess.call([prog, cmd, EMOTIONS[maxindex]])
        cv2.imshow('Frame', cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))
    else:
        # print("Pas de nouveau visage detecte! \n")
        # time.sleep(5)
        1 == 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(5)

cv2.destroyAllWindows()
