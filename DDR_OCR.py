import time
from datetime import date
import pickle
import os
import platform
import pip._internal as pip


try:
    import cv2
    import pytesseract
    import pandas as pd
    import numpy as np
    from pygrabber.dshow_graph import FilterGraph
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
    from pygrabber.dshow_graph import FilterGraph
    from tkinter import *
    from tkinter import filedialog as fd


# Removes the Text Outlines in image
def remove_outline(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    temp_img = np.copy(img)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    return cv2.bitwise_and(cv2.bitwise_not(temp_img), mask, cv2.bitwise_not(temp_img))


# Paths and Files
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
video_path = r"D:\Downloads\videoplayback.mp4"
cwd = os.getcwd()
tessdata_dir_config = r"--tessdata-dir " + os.path.relpath(cwd)
fileName = "Scores.csv"

ans = input("Which will you be scanning? \nEnter: OBS or Video(V)\n")

while True:
    if ans.lower() == "obs" or ans.lower() == "":
        os.system("cls")

        # Verfies that OBS Virtual camera is On
        entered = 0
        while "OBS Virtual Camera" not in FilterGraph().get_input_devices():
            if entered != 1:
                print(
                    "Turn on OBS Virtual Camera! \nTo stop script hit Ctrl+C within terminal."
                )
                entered = 1

        print(
            "OBS Virtual Camera has been detected. \nPlease verify that the proper gameplay source is being captured. \nThen verify that the Virtual Camera is running.\n"
        )

        # Grabs camera index
        cam_idx = FilterGraph().get_input_devices().index("OBS Virtual Camera")

        # Webcam Input
        cap = cv2.VideoCapture(cam_idx)

        break

    elif ans.lower() == "video" or ans.lower() == "v":
        window = Tk()
        window.withdraw()
        window.title("Select the video to Scan.")
        image_path = fd.askopenfilename(initialdir="/", parent=window)

        # Video Input
        os.system("cls")
        print("Using Video Input.")
        cap = cv2.VideoCapture(video_path)

        break

    else:
        os.system("cls")
        ans = input(
            "Invalid Input! \nWhich will you be scanning? \nEnter: OBS or Video(V)\n"
        )


# Grand Prix Image Flag
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


# Change paths based on OS
if platform.system().lower() != "windows":
    from wslpath import wslpath as wp

    tesseract_path = "/usr/bin/tesseract"
    video_path = wp(video_path)


# Username Input Section
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


# Player 1 & 2 Search tile Locations
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
    (slice(554, 590), slice(1389, 1487)),
    (slice(600, 880), slice(1375, 1493)),
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

# Full Combo Rank List
fcRank = {"MFC": 4, "PFC": 3, "GrFC": 2, "GFC": 1, "NoFC": 0}

# Tesseract OCR Initiallization
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Current Date
today = date.today()
today = today.strftime("%m/%d/%Y")


# Configures Webcam to 1920x1080 Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


# Get video metadata
video_fps = int((cap.get(cv2.CAP_PROP_FPS)))
if video_fps == 0:
    print(
        "Make sure OBS Virtual Camera FPS is set to 60 FPS in settings. \nWill force 60 FPS. \nResults may Vary.\n"
    )
    cap.set(cv2.CAP_PROP_FPS, 60)
    video_fps = int((cap.get(cv2.CAP_PROP_FPS)))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Used to deteremine if upscaling is needed.
upscale_flag = 0
if height != 1080 and width != 1920:
    print(
        "Make sure OBS Virtual Camera Resolution is set to 1920x1080 in settings. \nWill force resolution, this will take more processing power and may slow down your system. \n***Results will Vary!***\n"
    )
    upscale_flag = 1

# Frame Count Initialization
frame_count = 0

# Frame Capture Flag
frame_captured = False

# Displays Video Metadata
print("FPS: {} | Input Resolution: {}x{}".format(video_fps, int(width), int(height)))

# Loading CSV File
csv = pd.read_csv(fileName, header="infer")

# Displaying User name
print("Will be grabbing Score data for user: {}".format(mainPlayer))
# Lets User know script is running
print("Running")
print("Hit Ctrl+C to exit the script. \nType cls to clear terminal view, if needed.")
#  Process Start Time
start_time = time.time()

# Video Processing Loop
while cap.isOpened():
    ret, frame = cap.read()

    mNokEX = 3
    pEX = 2
    gEX = 1
    frame_count += 1

    if (frame_count % 30 == 0) and (frame_captured == False):
        # Resizes image
        if upscale_flag == 1:
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)

        # Converts Frame to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Creates Binary Thresholded Image
        gray_threshold = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)[1]

        # Inverts Thresholded Image
        gray_not = cv2.bitwise_not(gray_threshold)

        # Scans for results in Results Page
        screenOut = pytesseract.image_to_string(
            gray_not[10:58, 764:1153],
            lang="eng",
            config="--psm 10  " + tessdata_dir_config,
        )

        if "results" in screenOut.lower():
            # Scans for Player 1 Name
            player1Out = pytesseract.image_to_string(
                gray_not[115:153, 39:231],
                lang="eng+jpn",
                config="--psm 10  " + tessdata_dir_config,
            ).strip()

            # Scans for Player 2 Name
            player2Out = pytesseract.image_to_string(
                gray_not[115:153, 1677:1902],
                lang="eng+jpn",
                config="--psm 10  " + tessdata_dir_config,
            ).strip()

            # Selects Image Slices based on user Player position
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
                # Scans for max combo text in Results Page
                tabOut = pytesseract.image_to_string(
                    gray_not[player_loc[0]],
                    lang="eng+jpn",
                    config="--psm 10  " + tessdata_dir_config,
                )

                if "max combo" in tabOut.lower():
                    # Displays Frame to be processed
                    # cv2.imshow("Frame", gray_not)
                    # if cv2.waitKey(1) == ord("q"):
                    #     break

                    # Creates seperate threshold for Song title ROI
                    song_threshold = cv2.threshold(
                        gray[player_loc[1]], 220, 255, cv2.THRESH_BINARY
                    )[1]

                    # Scans text in Max Combo ROI
                    maxOut = pytesseract.image_to_string(
                        gray_not[player_loc[2]],
                        lang="eng",
                        config="--psm 10 -c tessedit_char_whitelist=0123456789",
                    )

                    # Scans text in Combo ROI
                    comboOut = pytesseract.image_to_string(
                        cv2.GaussianBlur(
                            remove_outline(gray_not[player_loc[3]]), (7, 7), 0
                        ),
                        lang="eng",
                        config="--psm 6 -c tessedit_char_whitelist=0123456789",
                    )

                    # Scans text in Fast steps ROI
                    fastOut = pytesseract.image_to_string(
                        remove_outline(gray_not[player_loc[4]]),
                        lang="eng",
                        config="--psm 10 -c tessedit_char_whitelist=0123456789",
                    )

                    # Scans text in Slow steps ROI
                    slowOut = pytesseract.image_to_string(
                        remove_outline(gray_not[player_loc[5]]),
                        lang="eng",
                        config="--psm 10 -c tessedit_char_whitelist=0123456789",
                    )

                    # Scans text in Song title ROI
                    songOut = pytesseract.image_to_string(
                        song_threshold,
                        lang="eng+jpn",
                        config="--psm 10  " + tessdata_dir_config,
                    )

                    # Scans text in Difficulty ROI
                    diffOut = pytesseract.image_to_string(
                        cv2.dilate(gray_not[player_loc[6]], np.ones((3, 3), np.uint8)),
                        lang="eng",
                        config="--psm 10  -c tessedit_char_whitelist=0123456789"
                        + tessdata_dir_config,
                    )

                    # Splits Combo step text
                    maxOut = int(maxOut.split()[0])
                    marvOut = int(comboOut.split()[0])
                    perfOut = int(comboOut.split()[1])
                    gretOut = int(comboOut.split()[2])
                    goodOut = int(comboOut.split()[3])
                    okOut = int(comboOut.split()[4])
                    missOut = int(comboOut.split()[5])

                    # Calculates Full combo status based on combo steps
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

                    # Calculates the EX score
                    exOut = mNokEX * (marvOut + okOut) + pEX * perfOut + gEX * gretOut

                    # Calclates the Money Score
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

                    # Displays the text in the console
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

                # Updates or Stores data into CSV File
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
                frame_captured = True
                print("--- %s seconds ---" % (time.time() - start_time))
                start_time = time.time()

    elif (frame_captured == True) and ((frame_count % (140 * video_fps)) == 0):
        frame_captured = False
