# HOW TO USE
# Use this with a printed SUDOKU Grid
# Press ESC key to Exit
import cv2
import numpy as np
from operator import itemgetter

# https://caphuuquan.blogspot.com/2017/04/building-simple-sudoku-solver-from.html

ratio2 = 3
kernel_size = 3
lowThreshold = 30

cv2.namedWindow("SUDOKU Solver")
sudoku1 = cv2.imread('test.jpg', 0)
sudoku1 = cv2.blur(sudoku1, (3, 3))
# Apply Canny edge detection
edges = cv2.Canny(sudoku1, lowThreshold, lowThreshold * ratio2, kernel_size)
# Apply Hough Line Transform, return a list of rho and theta
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200, 0, 0)
# print(lines[1][0])
# print(type(lines[1]))
if (lines is not None):
    # print("hallo")
    lines = lines[0]
    # lines = sorted(lines, key=lambda line: (line[0], line[1]))
    lines.sort(axis = 0) # = sorted(lines[0], key=itemgetter(1))
    print(lines)
    # Define the position of horizontal and vertical line
    pos_hori = 0
    pos_vert = 0
    # Create a list to store new bundle of lines
    New_lines = []
    # Store intersection points
    Points = []
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        # If b > 0.5, the angle must be greater than 45 degree
        # so we consider that line as a vertical line
        print(b)
        if (b > 0.5):
            # Check the position
            if (rho - pos_hori > 10):
                # Update the position
                pos_hori = rho
                # Saving new line, 0 is horizontal line, 1 is vertical line
                New_lines.append([rho, theta, 0])
        else:
            if (rho - pos_vert > 10):
                pos_vert = rho
                New_lines.append([rho, theta, 1])
                print("New_Lines: " , rho, theta )

    for i in range(len(New_lines)):
        if (New_lines[i][2] == 0):
            for j in range(len(New_lines)):
                if (New_lines[j][2] == 1):
                    theta1 = New_lines[i][1]
                    theta2 = New_lines[j][1]
                    p1 = New_lines[i][0]
                    p2 = New_lines[j][0]
                    xy = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
                    p = np.array([p1, p2])
                    res = np.linalg.solve(xy, p)
                    Points.append(res)
                    print("Points: " + len(Points))

    if (len(Points) == 100):
        print("hallo2")
        sudoku1 = cv2.adaptiveThreshold(sudoku1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 1)
        for i in range(0, 9):
            for j in range(0, 9):
                y1 = int(Points[j + i * 10][1] + 5)
                y2 = int(Points[j + i * 10 + 11][1] - 5)
                x1 = int(Points[j + i * 10][0] + 5)
                x2 = int(Points[j + i * 10 + 11][0] - 5)
                # Saving extracted block for training, uncomment for saving digit blocks
                cv2.imwrite(str((i + 1) * (j + 1)) + ".jpg", sudoku1[y1: y2,
                                                             x1: x2])
                cv2.rectangle(sudoku1, (x1, y1),
                              (x2, y2), (0, 255, 0), 2)
# Show the result
cv2.imshow("SUDOKU Solver", sudoku1)
cv2.waitKey(0)
cv2.destroyAllWindows()
