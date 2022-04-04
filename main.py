import cv2

cap = cv2.VideoCapture('Car - 16849.mp4')


object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)




class Box:
    def __init__(self, start_point, width_height):
        self.start_point = start_point
        self.end_point = (start_point[0] + width_height[0], start_point[1] + width_height[1])
        self.counter = 0
        self.frame_countdown = 0

    def overlap(self, start_point, end_point):
        if self.start_point[0] >= end_point[0] or self.end_point[0] <= start_point[0] or self.start_point[1] >= end_point[1] or self.end_point[1] <= start_point[1]:
            return False
        else:
            return True

boxes = []
boxes.append(Box((300, 200),(10, 80)))

while cap.isOpened():
    ret, frame = cap.read()
    height, width, _ = frame.shape

    roi = frame[250: 650, 250: 1000]

    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 1000:
            #cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)


    cv2.imshow("Roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if ret:
        key = cv2.waitKey(30)
    else:
        break

cap.release()
cv2.destroyAllWindows()
