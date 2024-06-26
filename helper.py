#############################################################################
#                               Libraries                                   #
#############################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt

def imread(img_path):
    """
    ------------------------------------------------------------------------------------------------
    Input:       img_path: string representing the path to the image file.
    Output:      img: numpy array representing the image in RGB color space.
    Description: Read an image from a given path and convert it from BGR to RGB color space.
    ------------------------------------------------------------------------------------------------
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def imshow(img, figsize=(5, 5)):
    """
    ------------------------------------------------------------------------------------------------
    Inputs:      img: numpy array representing the image.
                 figsize: tuple specifying the size of the figure (default is (5, 5)).
    Description: Display an image using Matplotlib.
    ------------------------------------------------------------------------------------------------
    """
    plt.figure(figsize=figsize)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='gray')
    plt.show()
    
def rgb2bin(img_rgb):    
    """
    ------------------------------------------------------------------------------------------------
    Input:       img_rgb: numpy array representing the input RGB image.
    Output:      img_bin: numpy array representing the binary image.
    Description: Convert an RGB image to a binary image using edge detection(Canny) 
                 after applying GaussianBlur and finally apply thresholding.
    ------------------------------------------------------------------------------------------------
    """
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blured_image = cv2.GaussianBlur(img_gray,(5,5), 1)
    canny_image = cv2.Canny(blured_image, 10, 50)
    t, img_bin = cv2.threshold(
        canny_image, 150, 255,cv2.THRESH_BINARY
    )
    return img_bin

def best_contours(img_rgb):    
    """
    ------------------------------------------------------------------------------------------------
    Input:       img_rgb: numpy array representing the input RGB image.
    Output:      best_contours: sorted numpy arrays representing the best contours based on High area.
    Description: Find the best contours in an RGB image based on contour area.
    ------------------------------------------------------------------------------------------------
    """
    img_bin = rgb2bin(img_rgb)
    contours, h = cv2.findContours(
        img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    best_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return best_contours

def find_contours(original_image, binary_image, index=-1, color=(255, 0, 0), thickness=10):
    """
    ------------------------------------------------------------------------------------------------
    Inputs:
    - original_image: numpy array representing the original image to show image on it.
    - binary_image: numpy array representing the binary image to find Contours.
    - index: index of the contour to draw (default is -1 to draw all contours).
    - color: tuple specifying the color of the contour (default is (255, 0, 0)).
    - thickness: integer specifying the thickness of the contour (default is 10).
    Description: Find and draw contours on an original image based on a binary image.
    ------------------------------------------------------------------------------------------------
    """
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_copy = original_image.copy()
    cv2.drawContours(img_copy, contours, index, color, thickness)
    # imshow(img_copy)

def get_corner_points(contour):    
    """
    ------------------------------------------------------------------------------------------------
    Input:       contour: numpy array representing the contour.
    Output:      approx: numpy array representing the corner points of the contour.
    Description: Approximate the corner points of a contour.
    ------------------------------------------------------------------------------------------------
    """
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02* peri, True)
    return approx

def reorderPoints(points):
    """
    ------------------------------------------------------------------------------------------------
    Input:
    - points: numpy array representing the corner points of the quadrilateral.
    Output: new_points: 
                        numpy array representing the reordered corner points
                        (4, 1, 2) which means that there is `4 rows` each row has `2 columns`, we don't need to 1.
                        we must reshape this to (4, 2)
    Description: This function used for reordering each point to know which point is orgin and which is top left and so on.
    ------------------------------------------------------------------------------------------------
    """
    points = points.reshape((4,2))
    new_points = np.zeros((4,1,2), np.int32)
    summation = points.sum(1)
    # print(points)
    # print(summation)
    new_points[0] = points[np.argmin(summation)]    # [0, 0]
    new_points[3] = points[np.argmax(summation)]    # [w, h]
    different = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(different)]    # [w, 0]
    new_points[2] = points[np.argmax(different)]    # [0, h]
    # print(different)
    return new_points

def show_answers(image, answer_index, grading, answers, questions, choices):    
    """
    ------------------------------------------------------------------------------------------------
    Inputs:
    - image: numpy array representing the input image.
    - answer_index: list containing the index of the selected answer for each question.
    - grading: list containing the grading (1 for correct, 0 for incorrect) for each question.
    - answers: list containing the correct answers for each question.
    - questions: integer specifying the number of questions.
    - choices: integer specifying the number of choices for each question.
    Output:     image: numpy array representing the image with answers and grading displayed.
    Descrition: Display answers and grading on an image.
    ------------------------------------------------------------------------------------------------
    """
    width_sec = int(image.shape[1] / choices)  # Adjusted for 4 choices
    height_sec = int(image.shape[0] / questions)  # Adjusted for 6 questions

    for x in range(0, questions):
        my_answer = answer_index[x]
        cx = (my_answer * width_sec) + width_sec // 2
        cy = (x * height_sec) + height_sec // 2

        if grading[x] == 1:
            my_color = (0, 255, 0)
        else:
            my_color = (255, 0, 0)
            correct_answer = answers[x]
            cv2.circle(image, ((correct_answer * width_sec) + width_sec // 2, (x * height_sec) + height_sec // 2), 25,
                       (255, 255, 0), cv2.FILLED)
        cv2.circle(image, (cx, cy), 50, my_color, cv2.FILLED)
    return image
