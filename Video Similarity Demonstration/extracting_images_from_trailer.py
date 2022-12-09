import cv2
import os
import numpy as np
from scenedetect import detect, ContentDetector, AdaptiveDetector, ThresholdDetector


def extract_frames(video_directory, output_directory_name, file_name):
    scene_list = detect(
        video_directory,

        ContentDetector(threshold=15, min_scene_len=5))
    # Read the video from specified path
    cam = cv2.VideoCapture(video_directory)

    try:

        # creating a folder named data
        if not os.path.exists(output_directory_name):
            os.makedirs(output_directory_name)

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame

    name_list = []
    previous_frame = None
    for i, scene in enumerate(scene_list):
        print('Current scene is %d' % (i))

        for j in range(scene[0].get_frames(), scene[1].get_frames(), 2):
            currentframe = j
            frame_count = j
            cam.set(1, currentframe)
            # reading from frame
            ret, frame = cam.read()

            name = './' + output_directory_name + './' + file_name + 'frame' + str(frame_count) + '.jpg'
            print('Creating...' + name)
            ## Removing all black images from trailer and single color images
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # print(gray)
            if j > scene[0].get_frames() and previous_frame is not None:

                hist_1 = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8],
                                      [0, 256, 0, 256, 0, 256])
                hist_1 = cv2.normalize(hist_1, hist_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()

                hist_2 = cv2.calcHist([previous_frame], [0, 1, 2], None, [8, 8, 8],
                                      [0, 256, 0, 256, 0, 256])
                hist_2 = cv2.normalize(hist_2, hist_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()

                similarity = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_CORREL)

                if similarity > 0.85:
                    print('Histogram similarity has occured, not counted ====================================')
                    continue

            if j > scene[0].get_frames() and previous_frame is not None:
                previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
                hist_1 = cv2.calcHist([gray], [0], None, [256],
                                      [0, 256])
                hist_1 = cv2.normalize(hist_1, hist_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()

                hist_2 = cv2.calcHist([previous_gray], [0], None, [256],
                                      [0, 256])
                hist_2 = cv2.normalize(hist_2, hist_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()

                similarity = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_CORREL)

                if similarity > 0.85:
                    print('Histogram similarity for RED has occured, not counted ++++++++++++++')
                    continue

            if (len(gray[(gray <= 26)]) / (gray.shape[0] * gray.shape[1]) >= 0.8) or (
                    len(gray[(gray >= 228)]) / (gray.shape[0] * gray.shape[1]) >= 0.8):
                print('Frame: %d is meeting the single color criteria' % (currentframe))
                continue

            elif cv2.Laplacian(gray, cv2.CV_64F).var() < 4:
                print('Frame: %d is meeting the blur criteria' % (currentframe))
                continue

            else:
                # writing the extracted images

                # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
                # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # cnt = contours[0]
                # x, y, w, h = cv2.boundingRect(cnt)
                # crop = frame[y:y + h, x:x + w]

                y_nonzero, x_nonzero, _ = np.nonzero(frame)

                cv2.imwrite(name, frame[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)])
                # cv2.imwrite(name, frame)
                name_list.append(name)
            # increasing counter so that it will
            # show how many frames are created

            previous_frame = frame

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    image_directory = os.path.join(os.getcwd(), output_directory_name)
    return name_list, image_directory
