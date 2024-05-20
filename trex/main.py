import cv2 as cv
from mss import mss
import numpy as np
from time import sleep, time
import pyautogui as inp
from skimage.morphology import closing,disk

dino_img = cv.imread("t-rex.png")
dino_img = cv.cvtColor(dino_img, cv.COLOR_RGB2GRAY)


monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

frame_rate = 120
frame_duration = 1.0 / frame_rate

with mss() as sct:
    inp.press("space")
    sleep(2)

    # Захват изображения
    img = np.array(sct.grab(monitor))
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Ищем динозаврика
    _, thrash, t_min_loc, t_max_loc = cv.minMaxLoc(cv.matchTemplate(img, dino_img, cv.TM_SQDIFF_NORMED))

    monitor = {
        "top": t_min_loc[1] - 94,
        "left": t_min_loc[0] - 4,
        "width": 300,
        "height": 150,
    }

    img = np.array(sct.grab(monitor))
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    _, thrash, t_min_loc, t_max_loc = cv.minMaxLoc(cv.matchTemplate(img, dino_img, cv.TM_SQDIFF_NORMED))

    start = time()
    timer = 0

    while True:
        frame_start_time = time()

        img = np.array(sct.grab(monitor))
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)

        enemy = img[:, int(img.shape[1] * 0.3) :]
        enemy = enemy[: int(enemy.shape[0] * 0.855), :]
        enemy = closing(enemy, disk(3))

        contours, _ = cv.findContours(enemy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Получаем координаты ограничивающего прямоугольника для текущего контура
            x, y, w, h = cv.boundingRect(contour)

            if time() - start < 50: agr = 20
            else: agr = 45

            timer = min(time() - start, 330)

            if x <= agr:
                if y + h - t_min_loc[1] - 20 > 0:
                    sleeper = (w) * 20 / (1000 + timer * 40)

                    if y - t_min_loc[1] >= -10:
                        sleeper += 0.1

                    inp.press("up")
                    sleep(sleeper / 4)
                    inp.keyDown("down")
                    sleep(0.02)
                    inp.keyUp("down")
                else:
                    inp.keyDown("down")
                    sleep(abs(sleeper-0.04))
                    inp.keyUp("down")


        frame_end_time = time()
        elapsed_time = frame_end_time - frame_start_time

        if elapsed_time < frame_duration:
            sleep(frame_duration - elapsed_time)

        if cv.waitKey(1) == ord("q"):
            break

        cv.imshow('Image', enemy)

cv.destroyAllWindows()
