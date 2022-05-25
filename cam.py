import cv2
import numpy as np
import math

cam = cv2.VideoCapture(0)


# def preprocess(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.adaptiveThreshold(
#         img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_TRUNC, 13, 1.2)

#     # blur = cv2.GaussianBlur(img, (5, 5), 0)
#     # ret3, img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     return img


while(cam.isOpened()):
    # read image
    ret, img = cam.read()
    img = cv2.flip(img, 1)

    if not ret:
        continue

    # get hand data from the rectangle sub window on the screen
    cv2.rectangle(img, (350, 350), (50, 50), (0, 255, 0), 0)
    crop_img = img[50:350, 50:350]

    # crop_img = preprocess(crop_img)

    # convert to grayscale
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # applying gaussian blur
    value = (5, 5)
    blurred = cv2.GaussianBlur(grey, value, 0)

    # thresholdin: Otsu's Binarization method
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # get contours
    contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)

    # find contour with max area
    cnt = max(contours, key=lambda x: cv2.contourArea(x))

    # create bounding rectangle around the contour (can skip below two lines)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt)

    # drawing contours
    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt, returnPoints=False)

    # finding convexity defects
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    # show contours
    cv2.imshow('Contours', drawing)
    cv2.imshow("Frame", crop_img)

    # cv2.imwrite("some.jpg", drawing)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()

cv2.destroyAllWindows()
