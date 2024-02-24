import cv2
import numpy as np
import time

# Release any existing capture if it exists
try:
    cap.release()
except NameError:
    pass

# Open the webcam
cap = cv2.VideoCapture(0)

# Define a kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Create a named window for displaying the canvas
cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)

# Initialize variables for drawing
canvas = None
prev_pt = None
clear = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Initialize canvas if not yet initialized
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of red color in HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Perform morphological operations to remove noise
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process contours
    if contours:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        # Draw on canvas
        if prev_pt is not None:
            canvas = cv2.line(canvas, prev_pt, (x, y), [0, 255, 0], 2)
        prev_pt = (x, y)
        
        # Clear canvas if area exceeds threshold
        if area > 50000:
            cv2.putText(canvas, 'Clearing Canvas', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5,
                        cv2.LINE_AA)
            clear = True
    else:
        prev_pt = None

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Add canvas to the frame
    frame = cv2.add(frame, canvas)

    # Display the resulting frame
    cv2.imshow('Canvas', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', res)

    # Clear canvas if needed
    if clear:
        time.sleep(0.8)
        canvas = np.zeros_like(frame)
        clear = False

    # Exit if 'ESC' is pressed or close button is clicked
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
