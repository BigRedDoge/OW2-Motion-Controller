import mediapipe as mp
import pyautogui
from enum import Enum
import cv2
import matplotlib.pyplot as plt
import numpy as np

from move_detect import MoveDetect
from yolo import load_model, run_inference, visualize_output

def main():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    _, image = cap.read()
    height, width = image.shape[:2]
    detect = MoveDetect(width, height)
    model = load_model()

    while True:
        # Read feed
        _, image = cap.read()
        #image = cv2.resize(image, (320, 240))
        #image.flags.writeable = False
        #image = cv2.flip(image, 1)
        #image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        #annotated = image.copy()
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #height, width = image.shape[:2]
        #print(height, width)

        # Make detection
        """
        with mp_pose.Pose(
            min_detection_confidence=0.25,
            min_tracking_confidence=0.25) as pose:
            keypoints = pose.process(image)
        """
        # Draw detection
        """
        if keypoints:
            mp_drawing.draw_landmarks(
                image,
                keypoints.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        ability = detect.check_ability_move(keypoints)
        if ability is not None:
            ability.move()
            #pass
        """
        output, image = run_inference(image, model)
        output, image = visualize_output(output, image, model)
        #print(output)
        ability = detect.check_ability_move(output)
        #threading.Thread(target=ability.move).start()
        # Show to screen
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


