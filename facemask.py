import sys
import getopt
import time
import keras
import numpy as np
import cv2


def evaluate_camera():
    class_indices = {'fully_covered': 0, 'not_covered': 1, 'partially_covered': 2}

    model = keras.models.load_model("model_dataset1_v2.h5")
    haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    vid = cv2.VideoCapture(0)

    while (True):

        ret, frame = vid.read()
        time.sleep(0.1)

        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rects = haar_cascade_face.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=12)

        for face in faces_rects:
            startY = face[1]
            endY = startY + face[3]
            startX = face[0]
            endX = startX + face[2]
            face_roi = frame[startY:endY, startX:endX]
            face_roi_resized = cv2.resize(face_roi, (64, 64))
            face_roi_resized = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2GRAY)

            input_arr = keras.preprocessing.image.img_to_array(face_roi_resized) / 255.0
            input_arr = np.array([input_arr])

            prediction = model.predict(input_arr)
            result = make_decision(prediction, class_indices)

            if result == 'fully_covered':
                color = (255, 0, 0)
            elif result == 'partially_covered':
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.putText(frame, result, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow('frameCaptured', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


def make_decision(arr, class_indices):
    max_index = np.argmax(arr)
    values = list(class_indices.keys())

    return values[max_index]


def detect_and_classification(img_path):
    haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    model = keras.models.load_model("model_dataset1.h5")
    class_indices = {'fully_covered': 0, 'not_covered': 1, 'partially_covered': 2}

    image = cv2.imread(img_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_rects = haar_cascade_face.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=12)

    for face in faces_rects:
        startY = face[1]
        endY = startY + face[3]
        startX = face[0]
        endX = startX + face[2]
        face_roi = image[startY:endY, startX:endX]
        face_roi_resized = cv2.resize(face_roi, (64, 64))
        face_roi_resized = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2GRAY)

        input_arr = keras.preprocessing.image.img_to_array(face_roi_resized) / 255.0
        input_arr = np.array([input_arr])
        prediction = model.predict(input_arr)
        result = make_decision(prediction, class_indices)

        if result == 'fully_covered':
            color = (255, 0, 0)
        elif result == 'partially_covered':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.putText(image, result, (startX, startY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 12)

    cv2.imwrite("processed.jpg", image)
    cv2.imshow("Processed image", cv2.resize(image, (960, 540)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def evaluate_face(path):
    model = keras.models.load_model("model_dataset1_v2.h5")
    class_indices = {'fully_covered': 0, 'not_covered': 1, 'partially_covered': 2}

    image_display = cv2.imread(path)
    image = keras.preprocessing.image.load_img(path, target_size=(64, 64), color_mode="grayscale")
    input_arr = keras.preprocessing.image.img_to_array(image) / 255.0
    input_arr = np.array([input_arr])

    result = model.predict(input_arr)
    result = np.argmax(result)
    values = list(class_indices.keys())
    print(values[result])

    x_size = image_display.shape[0]
    y_size = image_display.shape[1]
    if image_display.shape[0] < 450 or image_display.shape[0] > 800:
        x_size = 450
        y_size = 450

    if values[result] == 'fully_covered':
        color = (255, 0, 0)
    elif values[result] == 'partially_covered':
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    image_display = cv2.resize(image_display, (x_size, y_size))
    cv2.putText(image_display, values[result], (30, y_size - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Processed image", image_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    argv = sys.argv[1:]
    usage = "-c --camera \n" \
            "-p --photo <path_to_photography> \n" \
            "-f --face <path_to_face_photography> \n"
    try:
        opts, args = getopt.getopt(argv, "hcp:f:",
                                   ["help",
                                    "camera",
                                    "photo=",
                                    "face="])
    except getopt.GetoptError:
        print(usage)

    for opt, arg in opts:
        if opt in ['-c', '--camera']:
            evaluate_camera()
        elif opt in ['-p', '--photo']:
            path = arg
            print("PHOTO " + path)
            detect_and_classification(path)
        elif opt in ['-f', '--face']:
            path = arg
            evaluate_face(path)
        elif opt in ['-h', '--help']:
            print(usage)
        else:
            print(usage)


if __name__ == "__main__":
    main()
