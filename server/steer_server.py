import tensorflow as tf
import scipy.misc
from nets.pilotNet import PilotNet
import cv2
import mss
import numpy as np
import time

FLAGS = tf.app.flags.FLAGS

"""model from nvidia's training"""
tf.app.flags.DEFINE_string(
    'model', './save/model_nvidia.ckpt',
    """Path to the model parameter file.""")

tf.app.flags.DEFINE_string(
    'steer_image', './steering_wheel_image.jpg',
    """Steering wheel image to show corresponding steering wheel angle.""")

if __name__ == '__main__':
    img = cv2.imread(FLAGS.steer_image, 0)
    rows, cols = img.shape

    with tf.Graph().as_default():
        smoothed_angle = 0
        i = 0

        # construct model
        model = PilotNet()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # restore model variables
            saver.restore(sess, FLAGS.model)

            with mss.mss() as sct:
                # Part of the screen to capture
                monitor = {"top": 400, "left": 0, "width": 640, "height": 200}
                #monitor = {"top": 200, "left": 100, "width": 640, "height": 480}

                while "Screen capturing":
                    last_time = time.time()

                    # Get raw pixels from the screen, save it to a Numpy array
                    screen = np.array(sct.grab(monitor))
                    screen = np.flip(screen[:, :, :3], 2)
                    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

                    steer = cv2.cvtColor(screen, cv2.COLOR_RGB2YUV)
                    
                    image = scipy.misc.imresize(steer, [66, 200]) / 255.0

                    steering = sess.run(
                        model.steering,
                        feed_dict={
                            model.image_input: [image],
                            model.keep_prob: 1.0
                        }
                    )

                    degrees = steering[0][0] * 180.0 / scipy.pi
                    
                    print("Predicted steering angle: " +
                          str(degrees) + " degrees")

                    np.save('steer.npy', degrees)

                    print("fps: {}".format(1 / (time.time() - last_time)))
                    
                    cv2.imshow("Neural Network Input", image)

                    #cv2.putText(screen, "FPs: {}".format((1 / (time.time() - last_time)), (40+250,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1,cv2.LINE_AA)
                    cv2.imshow("Screen Capture", screen)
                    #print("Captured image size: {} x {}").format(frame.shape[0], frame.shape[1])

                    # make smooth angle transitions by turning the steering wheel based on the difference of the current angle
                    # and the predicted angle
                    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (
                        degrees - smoothed_angle) / abs(degrees - smoothed_angle)
                    M = cv2.getRotationMatrix2D(
                        (cols/2, rows/2), -smoothed_angle, 1)
                    dst = cv2.warpAffine(img, M, (cols, rows))
                    cv2.imshow("Server", dst)

                    i += 1
                    
                    # Press "q" to quit
                    if cv2.waitKey(25) & 0xFF == ord("q"):
                        cv2.destroyAllWindows()
                        break
