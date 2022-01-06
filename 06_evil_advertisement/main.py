'''
It is program that stop movie when you don't look at screen
Authors:
Maciej Rybacki
Łukasz Ćwikliński

## Zbuduj złą platformę do wyświetlania reklam
    - detekcja oczu (sprawdź czy oglądający nie zamknął oczu)
    - zatrzymuje wysłanie reklamy, gdy oglądający nie patrzy
'''

import os

import cv2

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


# getting video from webcam
cap = cv2.VideoCapture(0)

# pre trained classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# define a webdriver
browser = webdriver.Chrome(service=Service(f"{os.path.abspath(os.getcwd())}/chromedriver"))
browser.get("https://www.youtube.com/watch?v=YyzmLJVYtqk")
WebDriverWait(browser, 3).until(EC.presence_of_element_located((By.CLASS_NAME, 'video-stream')))

# play a video
browser.execute_script('document.getElementsByTagName("video")[0].play()')

# define statements
is_playing = True
is_paused = False

# loop for opened eyes
while cap.isOpened():
    # Reading from video
    ret, frame = cap.read()

    # Converting the recorded image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Applying filter to remove impurities
    gray = cv2.bilateralFilter(gray, 5, 1, 1)

    # Detecting the face for region of image to be fed to eye classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Preview from camera
    cv2.imshow("kamerka", frame)

    for (x, y, w, h) in faces:
        # Draw a rectangle for face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

        # roi_gray is face which is input to eyes classifier
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

        # Examining the length of eyes object for eyes
        if len(eyes) >= 2:
            if not is_playing:
                # play a video
                browser.execute_script('document.getElementsByTagName("video")[0].play()')
                is_playing = True
                is_paused = False
        else:
            if not is_paused:
                # stop a video
                browser.execute_script('document.getElementsByTagName("video")[0].pause()')
                is_playing = False
                is_paused = True

    key = cv2.waitKey(1)

    # safety key
    if key == ord('q'):
        browser.close()
        break


browser.close()
cap.release()
cv2.destroyAllWindows()
