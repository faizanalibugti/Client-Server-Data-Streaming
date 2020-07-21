import cv2
import numpy as np
import requests
import time

img = cv2.imread('./steering_wheel_image1.jpg', 0)
rows, cols = img.shape
smoothed_angle = 0
i = 0


while True:
    last_time = time.time()
    url = "http://192.168.1.100:8000/steer.npy"
    r = requests.get(url)
    
    try:
        degrees = np.load('./steer.npy', allow_pickle=True)
        print(degrees)
    except:
        print(degrees)

    print("Delay: {}".format((time.time() - last_time)))


    # make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    # and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D(
        (cols/2, rows/2), -smoothed_angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow("Client", dst)

    i += 1
            
    # Press "q" to quit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break

