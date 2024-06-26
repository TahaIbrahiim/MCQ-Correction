#############################################################################
#                               Libraries                                   #
#############################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
import helper

#############################################################################
#                               Read image                                  #
#############################################################################

images = [
    ['images/img5_1.jpg', 5, 5, [0,0,0,0,0]],
    ['images/img5_2.jpg', 5, 5, [0,0,0,0,0]],
    ['images/img6_1.jpg', 6,4,[0,0,0,0,0,0]],
    ['images/img6_2.jpg', 6,4,[0,0,0,0,0,0]],    
]

image_num = 1

image_path =  images[image_num][0]              # Path of image
width_image = 700                       # Width of image
height_image = 700                      # Height of image
questions = images[image_num][1]
choices = images[image_num][2]
camFeed = True
answers = images[image_num][3]

cap = cv2.VideoCapture(0)
# cap.set(10, 150)


while True: 
    # ret, original_image = cap.read()

    # if not ret:
    #     print("Error: Failed to capture frame from the webcam")
    #     break

    # else: 
    original_image = helper.imread(image_path)          # Read image

    #############################################################################
    #                               Preprocessing                               #
    #############################################################################
    # - Convert image to binary.
    # - Find contours

    bin_image = helper.rgb2bin(original_image)

    helper.find_contours(original_image, binary_image=bin_image, color=(0, 255, 0))
    biggest_contours = helper.best_contours(original_image)
    # print(f"First contour: {len(biggest_contours[0])}, Second contour: {len(biggest_contours[1])}")

    answer_contour= helper.get_corner_points(biggest_contours[0])
    grade_contour = helper.get_corner_points(biggest_contours[1])

    copy_image_for_answer = original_image.copy() 
    # helper.imshow(cv2.drawContours(copy_image_for_answer, answer_contour, -1, (0, 255,0), 50))
    # print('*'*50)
    copy_image_for_grade = original_image.copy() 
    # helper.imshow(cv2.drawContours(copy_image_for_grade, grade_contour, -1, (0, 255,0), 50))

    answer_contour = helper.reorderPoints(answer_contour)
    grade_contour = helper.reorderPoints(grade_contour)
    # from output `[1166  570 1014 1644]` smallest value '570' is orgin 

    #############################################################################
    #                               Extract answer part                         #
    #############################################################################

    point_answer1 = np.float32(answer_contour)
    point_answer2 = np.float32([[0, 0], [width_image, 0], [0, height_image], [width_image, height_image]])

    matrix_answer = cv2.getPerspectiveTransform(point_answer1, point_answer2)
    image_answer_display = cv2.warpPerspective(original_image, matrix_answer,(width_image, height_image))
    # helper.imshow(image_answer_display)

    ## Now, find grade part from an image by using coordinates 
    point_grade1 = np.float32(grade_contour)
    point_grade2 = np.float32([[0, 0], [300, 0], [0, 180], [300, 180]])

    matrix_grade = cv2.getPerspectiveTransform(point_grade1, point_grade2)
    image_grade_display = cv2.warpPerspective(original_image, matrix_grade,(300, 180))
    # helper.imshow(image_grade_display)

    #############################################################################
    #                               Check answer                                #
    #############################################################################
    # - At first, we must get threshold of an image 

    image_answer_bin = helper.rgb2bin(image_answer_display)
    # helper.imshow(image_answer_bin)

    # + Now, i need to crop each letter from this image
    def split_letters(image_answer_bin):
        letters = []
        rows = image_answer_bin.shape[0]
        multiple = (rows + questions - 1) // questions * questions

        # Resize the image to have a number of rows that is divisible by questions
        image_answer_resized = cv2.resize(image_answer_bin, (image_answer_bin.shape[1], multiple))

        # Split the resized image into rows
        rows = np.vsplit(image_answer_resized, questions)
        # imshow(rows[0])     # To show Row

        for row in rows:
            cols = np.hsplit(row, choices)
            for col in cols:
                letters.append(col)
        return letters


    letters = split_letters(image_answer_bin)
    # helper.imshow(letters[15])

    # How i now marked image 
    # 1. from binary image i count number of non zero values.
    # 2. if this answer is marked so it has a high of non zero values, because it white.

    # Show this example
    # cv2.countNonZero(letters[0]), cv2.countNonZero(letters[1]) output (2871, 6681)
    # From these result 
    # - letter[0] `A` is not marked.
    # - letter[1] `B` is marked.

    # Now i create array with the same size of an image
    new_image_value = np.zeros((questions, choices))
    row_counter = 0
    column_counter = 0

    for letter in letters:
        value_pixel = cv2.countNonZero(letter)

        new_image_value[row_counter, column_counter] = value_pixel
        column_counter += 1
        if column_counter == choices : row_counter +=1 ; column_counter = 0

    # print(new_image_value)

    # Now, i need to find index of each maximum value
    answer_index = []

    for x in range(0, questions):
        temp = new_image_value[x]
        index_value = np.where(temp == np.amax(temp))
        # print(index_value[0])
        answer_index.append(index_value[0][0])

    #############################################################################
    #                               Grading                                     #
    #############################################################################
    grading = []

    for x in range(0, questions):
        if answers[x] == answer_index[x]:
            grading.append(1)
        else: 
            grading.append(0)

    final_score = (sum(grading) / questions) * 100
    # print(final_score)

    # Show answers in part of image
    image_answer_display_copy = image_answer_display.copy()
    image_result = helper.show_answers(image_answer_display_copy, answer_index, grading, answers, questions, choices)

    # Now i will show these answer in original image
    image_row_drawing = np.zeros_like(image_answer_display)
    image_row_drawing = helper.show_answers(image_row_drawing, answer_index, grading, answers, questions, choices)
    # helper.imshow(image_row_drawing)

    # Invert the perspective transformation matrix
    inverse_matrix_answer = np.linalg.inv(matrix_answer)

    # Apply the inverse perspective transformation to bring the image back to its original position
    original_image_restored = cv2.warpPerspective(image_row_drawing, inverse_matrix_answer, (original_image.shape[1], original_image.shape[0]))
    # helper.imshow(original_image_restored)

    image_row_grading = np.zeros_like(image_grade_display)
    cv2.putText(image_row_grading, str(int(final_score)) + '%', (50, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)

    # helper.imshow(image_row_grading)

    # Invert the perspective transformation matrix
    inverse_matrix_grade = np.linalg.inv(matrix_grade)

    # Apply the inverse perspective transformation to bring the image back to its original position
    original_grade_restored = cv2.warpPerspective(image_row_grading, inverse_matrix_grade, (original_image.shape[1], original_image.shape[0]))
    # helper.imshow(original_grade_restored)

    final_image = original_image.copy()

    final_image = cv2.addWeighted(final_image,1, original_image_restored, 1, 0)
    final_image = cv2.addWeighted(final_image,1, original_grade_restored, 1, 0)
    impath = f"output/Final Result of image {image_num + 1}.jpg"
    cv2.imwrite(impath, final_image)
    helper.imshow(final_image)

    if cv2.waitKey(1) or 0xFF == ord('q'):
        break
    # helper.imshow(final_image, figsize=(7,7))

cap.release()
cv2.destroyAllWindows()



