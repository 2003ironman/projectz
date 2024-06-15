import cv2

video_path = r"C:\Users\Aditya Singh\OneDrive\Desktop\SPEED DECTETION\2165-155327596_tiny.mp4"
cap = cv2.VideoCapture(video_path)
X = cv2.createBackgroundSubtractorMOG2()  # X is the background subtractor

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # background subtraction to detect motions of the objects 
    Y = X.apply(frame)  # Y is the input binary image containing the foreground mass (motion mass = background subtractions)
    
    # Find contours to form boxes around objects for proper detection
    # cv2.RETR_EXTERNAL: Ignores any contours nested within others, retrieval mode for contours
    # cv2.CHAIN_APPROX_SIMPLE: Contours approximation method, stores only the end points
    Z,_ = cv2.findContours(Y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    for contour in Z:
        a, b, w, h = cv2.boundingRect(contour)  
        
        # The variable min_area filters out minimally integrated areas for small contours.
        min_area = 1000
        if cv2.contourArea(contour) > min_area:
            cv2.rectangle(frame, (a, b), (a+w, b+h), (0, 255, 0), 2)
            cv2.imshow("video result", frame)
            
    if cv2.waitKey(1) & 0xFF == ord("a"):
        break

cap.release()
cv2.destroyAllWindows()
