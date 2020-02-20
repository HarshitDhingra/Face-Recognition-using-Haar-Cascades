import numpy as np
import cv2

cam = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

data = []
i = 0

while True:
    ret, frame = cam.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_multi = face_cas.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in face_multi:

            face_each = frame[y:y + h, x:x + w, :]

            fc = cv2.resize(face_each, (50, 50))

            if i % 10 == 0 and len(data) < 20:
                data.append(fc)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        i += 1
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27 or len(data) >= 20:
            break
    else:
        print("error")

cv2.destroyAllWindows()

data = np.asarray(data)

np.save('dataset/face_101', data)
