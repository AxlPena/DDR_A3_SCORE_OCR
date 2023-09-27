import cv2
import pytesseract
import numpy as np
from datetime import date
import os
import platform
from wslpath import wslpath as wp


def remove_outline(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    temp_img = np.copy(img)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    return cv2.bitwise_and(cv2.bitwise_not(temp_img), mask, cv2.bitwise_not(temp_img))


tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
image_path = r"D:\Downloads\image2.png"

if platform.system().lower() != "windows":
    tesseract_path = "/usr/bin/tesseract"
    image_path = wp(image_path)


pytesseract.pytesseract.tesseract_cmd = tesseract_path

today = date.today()
today = today.strftime("%m/%d/%Y")

cwd = os.getcwd()
tessdata_dir_config = r"--tessdata-dir " + os.path.relpath(cwd)

fcRank = {"MFC": 4, "PFC": 3, "GrFC": 2, "GFC": 1, "NoFC": 0}


img = cv2.imread(image_path)


mNokEX = 3
pEX = 2
gEX = 1

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_threshold = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)[1]
gray_not = cv2.bitwise_not(gray_threshold)

cv2.imshow("Frame", gray_not)
# Waits for a keystroke

screenOut = pytesseract.image_to_string(
    gray_not[10:58, 764:1153], lang="eng", config="--psm 6  " + tessdata_dir_config
)
print(screenOut)
if "results" in screenOut.lower():
    tabOut = pytesseract.image_to_string(
        gray_not[552:591, 330:591],
        lang="eng+jpn",
        config="--psm 6  " + tessdata_dir_config,
    )
    print(tabOut)
    if "max combo" in tabOut.lower():
        # [slice(550,590),slice(630,734)] [slice(550,880),slice(600,734)]

        song_threshold = cv2.threshold(
            gray[415:458, 750:1170], 220, 255, cv2.THRESH_BINARY
        )[1]

        maxOut = pytesseract.image_to_string(
            gray_not[550:600, 630:734],
            lang="eng",
            config="--psm 6 digits",
        )

        comboOut = pytesseract.image_to_string(
            cv2.GaussianBlur(gray_not[600:880, 600:734], (7, 7), 0),
            lang="eng",
            config="--psm 6 digits",
        )

        fastOut = pytesseract.image_to_string(
            remove_outline(gray_not[734:772, 782:922]),
            lang="eng",
            config="--psm 6 digits",
        )

        slowOut = pytesseract.image_to_string(
            remove_outline(gray_not[817:858, 782:922]),
            lang="eng",
            config="--psm 6 digits -c page_separator=",
        )

        songOut = pytesseract.image_to_string(
            song_threshold, lang="eng+jpn", config="--psm 6  " + tessdata_dir_config
        )

        diffOut = pytesseract.image_to_string(
            gray_not[152:202, 585:655],
            lang="eng",
            config="--psm 6  digits" + tessdata_dir_config,
        )

        maxOut = int(maxOut.split()[0])
        marvOut = int(comboOut.split()[0])
        perfOut = int(comboOut.split()[1])
        gretOut = int(comboOut.split()[2])
        goodOut = int(comboOut.split()[3])
        okOut = int(comboOut.split()[4])
        missOut = int(comboOut.split()[5])

        if missOut == goodOut == gretOut == perfOut == 0:
            fullCombo = "MFC"

        elif missOut == goodOut == gretOut == 0:
            fullCombo = "PFC"

        elif missOut == 0 and goodOut == 0:
            fullCombo = "GrFC"

        elif missOut == 0 and goodOut != 0:
            fullCombo = "GFC"

        else:
            fullCombo = "NoFC"

        exOut = mNokEX * (marvOut + okOut) + pEX * perfOut + gEX * gretOut

        sc = marvOut + perfOut + gretOut + goodOut + missOut + okOut
        marvS = 1000000 / sc
        perfS = marvS - 10
        greatS = 0.6 * marvS - 10
        goodS = 0.2 * marvS - 10
        score = (
            (marvOut + okOut) * marvS
            + perfOut * perfS
            + gretOut * greatS
            + goodOut * goodS
        )
        score = np.floor(score / 10) * 10

        print("Song: {}".format(songOut))
        print("Diff: " + diffOut.split()[0])
        print("Max Combo: {}".format(maxOut))
        print("Full Combo: " + fullCombo)
        print("Marvelous: {}".format(marvOut))
        print("Perfect: {}".format(perfOut))
        print("Great: {}".format(gretOut))
        print("Good: {}".format(goodOut))
        print("O.K.: {}".format(okOut))
        print("Miss: {}".format(missOut))
        print(" ")
        print("Fast: " + fastOut.split()[0])
        print("Slow: " + slowOut.split()[0])
        print("EX: {}".format(exOut))
        print("Money Score: {}".format(score))
        cv2.waitKey(0)
