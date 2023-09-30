import cv2
import pytesseract
import time
import numpy as np
from datetime import date
import os
import platform
import pickle


try:
    import cv2
    import pytesseract
    import pandas as pd
    import numpy as np
    from tkinter import *
    from tkinter import filedialog as fd

except ImportError:
    print("\n TLDR: You are missing some import packages. \n Going to Fetch them.")
    pip.main(["install", "-r", "requirements.txt", "--user"])
    print("All Packages have been installed!")
    time.sleep(2)
    os.system("cls")

    import cv2
    import pytesseract
    import pandas as pd
    import numpy as np
    from tkinter import *
    from tkinter import filedialog as fd


def remove_outline(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    temp_img = np.copy(img)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    return cv2.bitwise_and(cv2.bitwise_not(temp_img), mask, cv2.bitwise_not(temp_img))


tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


window = Tk()
window.withdraw()

window.title("Select the Screenshot to Scan.")
os.system("cls")
print("Select the Screenshot to Scan.")

image_path = fd.askopenfilename(initialdir="/", parent=window)

#Grand Prix Image Flag
gp_flag = 0

ans = input("Which will you be scanning? \nEnter: A3 or Grand Prix(GP)\n")

while True:
    if ans.lower() == "grand prix" or ans.lower() == "gp":
        os.system("cls")
        os.system("The image is of Grand Prix Results.")
        gp_flag = 1
        break

    elif ans.lower() == "a3" or ans.lower() == "":
        os.system("cls")
        os.system("The image is of A3 Results.")
        break

    else:
        os.system("cls")
        ans = input(
            "Invalid Input! \nWhich will you be scanning? \nEnter: A3 or Grand Prix(GP)\n"
        )



if platform.system().lower() != "windows":
    from wslpath import wslpath as wp

    tesseract_path = "/usr/bin/tesseract"
    image_path = wp(image_path)


pytesseract.pytesseract.tesseract_cmd = tesseract_path

player_loc = "Not Player"

if os.path.isfile("userData.p"):
    mainPlayer = pickle.load(open("userData.p", "rb"))
    ans = input(
        "Last Player Data was grabbed for was: {} ?\nDo you wish to grab data for this user? \nEnter: Y/N \n".format(
            mainPlayer
        )
    )

    while True:
        if ans.lower() == "y" or ans.lower() == "":
            os.system("cls")
            break

        elif ans.lower() == "n":
            mainPlayer = input("Enter your DDR Username: ")
            print("Will cache username for future use.")
            pickle.dump(mainPlayer, open("userData.p", "wb"))
            time.sleep(2)
            os.system("cls")
            break

        else:
            os.system("cls")
            ans = input(
                "Invalid Input! \nDo you want to grab data for Player: {} ? \nEnter: Y/N \n".format(
                    mainPlayer
                )
            )

else:
    mainPlayer = input("Enter your DDR Username:")
    print("Caching username for future use.")
    pickle.dump(mainPlayer, open("userData.p", "wb"))
    time.sleep(2)
    os.system("cls")

print("Will be grabbing Score data for user: {}".format(mainPlayer))

today = date.today()
today = today.strftime("%m/%d/%Y")

cwd = os.getcwd()
tessdata_dir_config = r"--tessdata-dir " + os.path.relpath(cwd)

fcRank = {"MFC": 4, "PFC": 3, "GrFC": 2, "GFC": 1, "NoFC": 0}


img = cv2.imread(image_path)

height, width, _ = img.shape

# Used to deteremine if upscaling is needed.

if height != 1080 and width != 1920:
    print(
        "Image resolution is {}x{}. \nWill force 1920x1080 by upscaling resolution. \n***Results will Vary!***\n".format(
            width, height
        )
    )
    img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)


mNokEX = 3
pEX = 2
gEX = 1

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_threshold = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)[1]
gray_not = cv2.bitwise_not(gray_threshold)


cv2.imshow("Frame", gray_not)
# Waits for a keystroke
cv2.waitKey(0)

screenOut = pytesseract.image_to_string(
    gray_not[10:58, 764:1153], lang="eng", config="--psm 10  " + tessdata_dir_config
)
print(screenOut)

p1_loc = [
    (slice(552, 591), slice(330, 591)),
    (slice(415, 458), slice(750, 1170)),
    (slice(550, 590), slice(630, 734)),
    (slice(600, 880), slice(600, 734)),
    (slice(734, 772), slice(782, 922)),
    (slice(817, 858), slice(782, 922)),
    (slice(152, 202), slice(585, 655)),
]

p2_loc = [
    (slice(552, 591), slice(330, 591)),
    (slice(415, 458), slice(750, 1170)),
    (slice(554, 589), slice(1389, 1487)),
    (slice(600, 866), slice(1375, 1493)),
    (slice(818, 858), slice(1539, 1664)),
    (slice(817, 858), slice(782, 922)),
    (slice(152, 202), slice(1267, 1342)),
]

gp_loc = [
    (slice(552, 591), slice(330, 591)),
    (slice(415, 458), slice(750, 1170)),
    (slice(550, 590), slice(630, 734)),
    (slice(600, 880), slice(600, 734)),
    (slice(734, 772), slice(782, 922)),
    (slice(817, 858), slice(782, 922)),
    (slice(152, 202), slice(1267, 1342)),
]

if "results" in screenOut.lower():
    player1Out = pytesseract.image_to_string(
        gray_not[115:158, 39:231],
        lang="eng+jpn",
        config="--psm 10  " + tessdata_dir_config,
    ).strip()

    player2Out = pytesseract.image_to_string(
        gray_not[115:158, 1677:1902],
        lang="eng+jpn",
        config="--psm 10  " + tessdata_dir_config,
    ).strip()

    if gp_flag == 1:
        player_loc = gp_loc

    elif player1Out.lower() == mainPlayer.lower():
        player_loc = p1_loc

    elif player2Out.lower() == mainPlayer.lower():
        player_loc = p2_loc

    
    else:
        player_loc = "Not Player"
        print("{} is not playing this set :C.".format(mainPlayer))

    if player_loc != "Not Player":
        tabOut = pytesseract.image_to_string(
            gray_not[player_loc[0]],
            lang="eng+jpn",
            config="--psm 10  " + tessdata_dir_config,
        )

        if "max combo" in tabOut.lower():
            # [slice(550,590),slice(630,734)] [slice(550,880),slice(600,734)]

            song_threshold = cv2.threshold(
                gray[player_loc[1]], 220, 255, cv2.THRESH_BINARY
            )[1]

            maxOut = pytesseract.image_to_string(
                gray_not[player_loc[2]],
                lang="eng",
                config="--psm 10 -c tessedit_char_whitelist=0123456789",
            )

            comboOut = pytesseract.image_to_string(
                cv2.GaussianBlur(gray_not[player_loc[3]], (5, 5), 0),
                lang="eng",
                config="--psm 6 -c tessedit_char_whitelist=0123456789",
            )

            fastOut = pytesseract.image_to_string(
                remove_outline(gray_not[player_loc[4]]),
                lang="eng",
                config="--psm 10 -c tessedit_char_whitelist=0123456789",
            )

            slowOut = pytesseract.image_to_string(
                remove_outline(gray_not[player_loc[5]]),
                lang="eng",
                config="--psm 10 -c tessedit_char_whitelist=0123456789",
            )

            songOut = pytesseract.image_to_string(
                song_threshold,
                lang="eng+jpn",
                config="--psm 10  " + tessdata_dir_config,
            )

            diffOut = pytesseract.image_to_string(
                cv2.dilate(gray_not[player_loc[6]], np.ones((3, 3), np.uint8)),
                lang="eng",
                config="--psm 10  -c tessedit_char_whitelist=0123456789"
                + tessdata_dir_config,
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
            score = int(np.floor(score / 10) * 10)

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
            print("Fast: " + fastOut.split()[0])
            print("Slow: " + slowOut.split()[0])
            print("EX: {}".format(exOut))
            print("Money Score: {}".format(score))
            cv2.waitKey(0)
