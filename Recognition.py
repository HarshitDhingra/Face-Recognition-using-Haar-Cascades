import numpy as np
import cv2

cam = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

f_01 = np.load('dataset/face_101.npy').reshape((20, 50 * 50 * 3))
f_02 = np.load('dataset/face_102.npy').reshape((20, 50 * 50 * 3))

names_dict = {
    0: ' Harshit',
    1: ' Ansh',
}

Y = np.zeros((40, 1))
Y[:20, :] = 0.0
Y[20:40, :] = 1.0

data = np.concatenate([f_01, f_02])


def euclidean_distance(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).sum())


def knn(x1, train, k=5):
    n = train.shape[0]
    distance = []
    for i in range(n):
        distance.append(euclidean_distance(x1, train[i]))
    distance = np.asarray(distance)
    z = np.argsort(distance)
    sorted_dist = Y[z][:k]
    count = np.unique(sorted_dist, return_counts=True)
    return count[0][np.argmax(count[1])]


while True:
    ret, frame = cam.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        multi_faces = face_cas.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in multi_faces:
            face_each = frame[y:y + h, x:x + w, :]
            fc = cv2.resize(face_each, (50, 50))

            lab = knn(fc.flatten(), data)
            name_text = names_dict[int(lab)]

            cv2.putText(frame, name_text, (x, y), font, 1, (255, 255, 0), 2)

        cv2.imshow('face recognition', frame)

        if cv2.waitKey(1) == 27:
            break
    else:
        print("Face not Detected")

cv2.destroyAllWindows()
