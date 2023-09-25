import cv2
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture('walking.avi')
while True:
    # Read the frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass the frame to the body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        # Draw a rectangle around the detected area
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Display the frame
    cv2.imshow('Pedestrians', frame)

    # Exit if the Space key is pressed
    if cv2.waitKey(1) == 32:
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
