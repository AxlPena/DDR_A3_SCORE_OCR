import cv2
import pytesseract
import time
import numpy as np
import pandas as pd
from datetime import date
import os
import platform
from wslpath import wslpath as wp

#Process Start Time
start_time = time.time()

#Paths and Files
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
test_video_path = r"D:\Downloads\videoplayback.mp4"
cwd = os.getcwd()
tessdata_dir_config = r"--tessdata-dir " + os.path.relpath(cwd)
fileName = "Scores.csv"

#Displays Current Working Diredtory
print(cwd)

if platform.system().lower() != "windows":
    tesseract_path = "/usr/bin/tesseract"
    test_video_path = wp(test_video_path)

#Tesseract OCR Initiallization
pytesseract.pytesseract.tesseract_cmd = tesseract_path

#Current Date
today = date.today()
today = today.strftime("%m/%d/%Y")

#Full Combo Rank List
fcRank = {"MFC": 4, "PFC": 3, "GrFC": 2, "GFC": 1, "NoFC": 0}

#Removes the Text Outlines in image
def remove_outline(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    temp_img = np.copy(img)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    return cv2.bitwise_and(cv2.bitwise_not(temp_img), mask, cv2.bitwise_not(temp_img))

#Webcam Input
# cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)

#Test Video Input
cap = cv2.VideoCapture(test_video_path)

#Configures Webcam to 1920x1080 Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)



# Get video metadata
video_fps = (cap.get(cv2.CAP_PROP_FPS),)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

#Displays Video Metadata
print("FPS: {} | Resolution: {}x{}".format(video_fps[0],int(width),int(height)))

frame_count = 0

csv = pd.read_csv(fileName, header="infer")

print("Running")

while cap.isOpened():
    ret, frame = cap.read()

    mNokEX = 3
    pEX = 2
    gEX = 1
    frame_count += 1

    if frame_count % 30 == 0:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_threshold = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)[1]
        
        gray_not = cv2.bitwise_not(gray_threshold)

        cv2.imshow("Frame", gray_not)
        if cv2.waitKey(1) == ord("q"):
            break

        screenOut = pytesseract.image_to_string(
            gray_not[10:58, 764:1153],
            lang="eng",
            config="--psm 6  " + tessdata_dir_config,
        )

        if "results" in screenOut.lower():
            tabOut = pytesseract.image_to_string(
                gray_not[552:591, 330:591],
                lang="eng+jpn",
                config="--psm 6  " + tessdata_dir_config,
            )

            if "max combo" in tabOut.lower():
                song_threshold = cv2.threshold(
                    gray[415:458, 750:1170], 220, 255, cv2.THRESH_BINARY
                )[1]

                maxOut = pytesseract.image_to_string(
                    cv2.GaussianBlur(gray_not[550:590, 630:734], (5, 5), 0),
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
                    config="--psm 6 digits",
                )

                songOut = pytesseract.image_to_string(
                    song_threshold,
                    lang="eng+jpn",
                    config="--psm 6  " + tessdata_dir_config,
                ).strip()

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

                print("Song: {}".format(songOut.strip()))
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

                if (csv["Song"].eq(songOut)).any():
                    row_index = csv.index[csv["Song"].isin([songOut])][0]

                    # dateOfLastPlay
                    csv.iloc[row_index, 6] = today

                    # Money Score
                    if csv.iloc[row_index, 2] <= score:
                        csv.iloc[row_index, 2] = score

                    # EX
                    if csv.iloc[row_index, 3] <= exOut:
                        csv.iloc[row_index, 3] = exOut

                    # FC
                    if fcRank[csv.iloc[row_index, 4]] <= fcRank[fullCombo]:
                        csv.iloc[row_index, 4] = fullCombo
                        # dateOfFC
                        csv.iloc[row_index, 5] = today

                    elif csv.iloc[row_index, 4] == fullCombo:
                        # dateOfFC
                        csv.iloc[row_index, 5] = today

                    # MaxCombo
                    if csv.iloc[row_index, 7] < maxOut:
                        csv.iloc[row_index, 7] = maxOut

                    # Combo Scores
                    if csv.iloc[row_index, 8] <= marvOut:
                        csv.iloc[row_index, 8] = marvOut
                        csv.iloc[row_index, 9] = perfOut
                        csv.iloc[row_index, 10] = gretOut
                        csv.iloc[row_index, 11] = goodOut
                        csv.iloc[row_index, 12] = okOut
                        csv.iloc[row_index, 13] = missOut
                        csv.iloc[row_index, 14] = int(slowOut.split()[0])
                        csv.iloc[row_index, 15] = int(fastOut.split()[0])

                    csv.to_csv(fileName, mode="w", index=False, header=True)
                    print("Updated DDR data for song: " + songOut)

                else:
                    data = {
                        "Diff": [diffOut.split()[0]],
                        "Song": [songOut],
                        "MoneyScore": [score],
                        "EX": [exOut],
                        "FC": [fullCombo],
                        "dateOfFC": [today],
                        "dateOfLastPlay": [today],
                        "MaxCombo": [maxOut],
                        "Marvelous": [marvOut],
                        "Perfect": [perfOut],
                        "Great": [gretOut],
                        "Good": [goodOut],
                        "OK": [okOut],
                        "Miss": [missOut],
                        "Slow": [slowOut.split()[0]],
                        "Fast": [fastOut.split()[0]],
                    }
                    df = pd.DataFrame(data)
                    df.to_csv(fileName, mode="a", index=False, header=False)
                    print("First time play data added for new song: " + songOut)
                print("--- %s seconds ---" % (time.time() - start_time))
                time.sleep(120 - (time.time() - start_time))
                start_time = time.time()
