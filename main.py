import mediapipe as mp
import pyautogui
import win32api, win32con
from enum import Enum
import cv2
import math
import time
import threading

import torch
from torchvision import transforms
import sys
sys.path.append('yolov7')
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression_kpt
from yolov7.utils.plots import output_to_keypoint#, plot_skeleton_kpts

import matplotlib.pyplot as plt
import numpy as np

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

# find distance between two points
def find_distance(x1, x2, y1, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# calculate angle between 3 points
def find_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

class MoveDetect:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.lmPose = mp.solutions.pose.PoseLandmark

        self.prev_move = None

    def check_ability_move(self, keypoints):
        self.move = AbilityMove()
        print(self.move)
        if len(keypoints) > 0:
            self.get_keypoints(keypoints)
            self.check_keys()
        """
        lm = keypoints.pose_landmarks
        if lm:
            self.get_keypoints(lm)
            self.check_keys()
        """
        return self.move

    def check_keys(self):
        self.move.SHIFT = self.detect_shift()
        self.move.Q = self.detect_ult()
        self.move.E = self.detect_e()
        self.move.RIGHT_CLICK = self.detect_right_click()
        self.move.LEFT_CLICK = self.detect_left_click()
        self.move.D = self.detect_d()
        self.move.A = self.detect_a()
        self.move.W = self.detect_w()
        self.move.S = self.detect_s()
        self.move.LOOK_LEFT = self.detect_look_left()
        self.move.LOOK_RIGHT = self.detect_look_right()
        self.move.LOOK_UP = self.detect_look_up()
        self.move.LOOK_DOWN = self.detect_look_down()

    def get_keypoints(self, keypoints): #lm
        """
        Keypoint idx to body part
        0: nose
        1: right eye
        2: left eye
        3: right ear
        4: left ear
        5: right shoulder
        6: left shoulder
        7: right elbow
        8: left elbow
        9: right wrist
        10: left wrist
        11: right hip
        12: left hip
        13: right knee
        14: left knee
        15: right ankle
        16: left ankle
        """

        self.right_wrist_x = keypoints[9][0] 
        self.right_wrist_y = keypoints[9][1] 

        self.left_wrist_x = keypoints[10][0] 
        self.left_wrist_y = keypoints[10][1] 

        self.right_shoulder_x = keypoints[5][0] 
        self.right_shoulder_y = keypoints[5][1] 

        self.left_shoulder_x = keypoints[6][0] 
        self.left_shoulder_y = keypoints[6][1] 

        self.right_elbow_x = keypoints[7][0] 
        self.right_elbow_y = keypoints[7][1] 

        self.left_elbow_x = keypoints[8][0] 
        self.left_elbow_y = keypoints[8][1]  

        self.right_eye_outer_x = keypoints[1][0] 
        self.right_eye_outer_y = keypoints[1][1] 

        self.left_eye_outer_x = keypoints[2][0] 
        self.left_eye_outer_y = keypoints[2][1] 

        self.nose_x = keypoints[0][0] 
        self.nose_y = keypoints[0][1] 

        self.right_ear_x = keypoints[3][0] 
        self.right_ear_y = keypoints[3][1] 

        self.left_ear_x = keypoints[4][0] 
        self.left_ear_y = keypoints[4][1] 
        """
        self.right_wrist_x = int(lm.landmark[self.lmPose.RIGHT_WRIST].x * self.width) 
        self.right_wrist_y = int(lm.landmark[self.lmPose.RIGHT_WRIST].y * self.height) 

        self.left_wrist_x = int(lm.landmark[self.lmPose.LEFT_WRIST].x * self.width)
        self.left_wrist_y = int(lm.landmark[self.lmPose.LEFT_WRIST].y * self.height)

        self.right_shoulder_x = int(lm.landmark[self.lmPose.RIGHT_SHOULDER].x * self.width)
        self.right_shoulder_y = int(lm.landmark[self.lmPose.RIGHT_SHOULDER].y * self.height)

        self.left_shoulder_x = int(lm.landmark[self.lmPose.LEFT_SHOULDER].x * self.width)
        self.left_shoulder_y = int(lm.landmark[self.lmPose.LEFT_SHOULDER].y * self.height)

        self.right_elbow_x = int(lm.landmark[self.lmPose.RIGHT_ELBOW].x * self.width) 
        self.right_elbow_y = int(lm.landmark[self.lmPose.RIGHT_ELBOW].y * self.height) 

        self.left_elbow_x = int(lm.landmark[self.lmPose.LEFT_ELBOW].x * self.width) 
        self.left_elbow_y = int(lm.landmark[self.lmPose.LEFT_ELBOW].y * self.height) 

        self.right_eye_outer_x = int(lm.landmark[self.lmPose.RIGHT_EYE_OUTER].x * self.width)
        self.right_eye_outer_y = int(lm.landmark[self.lmPose.RIGHT_EYE_OUTER].y * self.height)

        self.left_eye_outer_x = int(lm.landmark[self.lmPose.LEFT_EYE_OUTER].x * self.width)
        self.left_eye_outer_y = int(lm.landmark[self.lmPose.LEFT_EYE_OUTER].y * self.height)

        self.nose_x = int(lm.landmark[self.lmPose.NOSE].x * self.width)
        self.nose_y = int(lm.landmark[self.lmPose.NOSE].y * self.height)

        self.right_ear_x = int(lm.landmark[self.lmPose.RIGHT_EAR].x * self.width)
        self.right_ear_y = int(lm.landmark[self.lmPose.RIGHT_EAR].y * self.height)

        self.left_ear_x = int(lm.landmark[self.lmPose.LEFT_EAR].x * self.width)
        self.left_ear_y = int(lm.landmark[self.lmPose.LEFT_EAR].y * self.height)
        """
        

    def detect_ult(self):
        if self.left_shoulder_x != -1 and self.left_wrist_x != -1 and self.left_shoulder_y != -1 and self.left_wrist_y != -1:
            wrist_dist = find_distance(self.left_wrist_x, self.right_wrist_x, self.left_wrist_y, self.right_wrist_y)
            #print(wrist_dist)
            if wrist_dist < (self.width / 8):
                return Abilities.Q
            else:
                return Abilities.NONE
        else:
            return Abilities.NONE

    def detect_e(self):
        if (self.left_elbow_y < self.left_shoulder_y) and (self.left_elbow_y != -1 and self.left_shoulder_y != -1):
            return Abilities.E
        else:
            return Abilities.NONE

    def detect_right_click(self):
        if self.left_shoulder_x != -1 and self.left_wrist_x != -1 and self.left_shoulder_y != -1 and self.left_wrist_y != -1:
            shoulder_wrist_dist = find_distance(self.left_shoulder_x, self.left_wrist_x, self.left_shoulder_y, self.left_wrist_y)
            #print(shoulder_wrist_dist)
            if shoulder_wrist_dist < (self.width / 16):
                return Abilities.RIGHT_CLICK
            else:
                return Abilities.NONE
        else:
            return Abilities.NONE

    def detect_left_click(self):
        shoulder_wrist_dist = find_distance(self.right_shoulder_x, self.right_wrist_x, self.right_shoulder_y, self.right_wrist_y)
        #print(shoulder_wrist_dist)
        if shoulder_wrist_dist < (self.width / 16):
            return Abilities.LEFT_CLICK
        else:
            return Abilities.NONE

    def detect_shift(self):
        if (self.right_elbow_y < self.right_shoulder_y) and (self.right_elbow_y != -1 and self.right_shoulder_y != -1):
            return Abilities.SHIFT
        else:
            return Abilities.NONE

    def detect_a(self):
        if (self.left_shoulder_x > (self.width / 3) * 2) and self.left_shoulder_x != -1:
            return Abilities.A
        else:
            return Abilities.NONE
            
    def detect_d(self):
        if (self.right_shoulder_x < self.width / 3) and self.right_shoulder_x != -1:
            return Abilities.D
        else:
            return Abilities.NONE

    def detect_w(self):
        if (find_distance(self.left_shoulder_x, self.right_shoulder_x, self.left_shoulder_y, self.right_shoulder_y) > 130) and (self.left_shoulder_x != -1 and self.right_shoulder_x != -1 and self.left_shoulder_y != -1 and self.right_shoulder_y != -1):
            return Abilities.W
        else:
            return Abilities.NONE

    def detect_s(self):
        #print(find_distance(self.left_shoulder_x, self.right_shoulder_x, self.left_shoulder_y, self.right_shoulder_y))
        if (find_distance(self.left_shoulder_x, self.right_shoulder_x, self.left_shoulder_y, self.right_shoulder_y) < 100) and (self.left_shoulder_x != -1 and self.right_shoulder_x != -1 and self.left_shoulder_y != -1 and self.right_shoulder_y != -1):
            return Abilities.S
        else:
            return Abilities.NONE
        
    def detect_look_left(self):
        print(self.left_eye_outer_x, self.nose_x)
        if (self.left_eye_outer_x > self.nose_x - 15) and (self.left_eye_outer_x != -1 and self.nose_x != -1):
            #print("LEFT")
            return Abilities.LOOK_LEFT
        else:
            return Abilities.NONE

    def detect_look_right(self):
        #print("right eye outer x: " + str(self.right_eye_outer_x))
        #print("right ear x: " + str(self.right_ear_x))
        if (self.right_eye_outer_x < self.nose_x + 15) and (self.right_eye_outer_x != -1 and self.nose_x != -1):
            #print("RIGHT")
            return Abilities.LOOK_RIGHT
        else:
            return Abilities.NONE

    def detect_look_up(self):
        if (self.right_eye_outer_y + 8 < self.right_ear_y and self.left_eye_outer_y + 8 < self.left_ear_y) and (self.right_eye_outer_y != -1 and self.right_ear_y != -1 and self.left_eye_outer_y != -1 and self.left_ear_y != -1):
            return Abilities.LOOK_UP
        else:
            return Abilities.NONE

    def detect_look_down(self):
        if (self.right_eye_outer_y > self.right_ear_y - 5 and self.left_eye_outer_y > self.left_ear_y - 5) and (self.right_eye_outer_y != -1 and self.right_ear_y != -1 and self.left_eye_outer_y != -1 and self.left_ear_y != -1):
            return Abilities.LOOK_DOWN
        else:
            return Abilities.NONE


class Abilities(Enum):
    SHIFT = "shift"
    Q = "q"
    E = "e"
    RIGHT_CLICK = "right click"
    LEFT_CLICK = "left click"
    D = "d"
    A = "a"
    W = "w"
    S = "s"
    LOOK_UP = "look up"
    LOOK_DOWN = "look down"
    LOOK_LEFT = "look left"
    LOOK_RIGHT = "look right"
    NONE = "none"


class AbilityMove:
    def __init__(self):
        self.SHIFT = Abilities.NONE
        self.Q = Abilities.NONE
        self.E = Abilities.NONE
        self.RIGHT_CLICK = Abilities.NONE
        self.LEFT_CLICK = Abilities.NONE
        self.D = Abilities.NONE
        self.A = Abilities.NONE
        self.W = Abilities.NONE
        self.S = Abilities.NONE
        self.LOOK_UP = Abilities.NONE
        self.LOOK_DOWN = Abilities.NONE
        self.LOOK_LEFT = Abilities.NONE
        self.LOOK_RIGHT = Abilities.NONE

    def move(self):
        if self.SHIFT == Abilities.SHIFT:
            pyautogui.keyDown('shift')
        else:
            pyautogui.keyUp('shift')

        if self.Q == Abilities.Q:
            pyautogui.keyDown('q')
        else:
            pyautogui.keyUp('q')

        if self.E == Abilities.E:
            pyautogui.keyDown('e')
        else:
            pyautogui.keyUp('e')

        if self.RIGHT_CLICK == Abilities.RIGHT_CLICK:
            pyautogui.mouseDown(button='right')
        else:
            pyautogui.mouseUp(button='right')
        
        if self.LEFT_CLICK == Abilities.LEFT_CLICK:
            pyautogui.mouseDown(button='left')
        else:
            pyautogui.mouseUp(button='left')

        if self.D == Abilities.D:
            pyautogui.keyDown('d')
        else:
            pyautogui.keyUp('d')

        if self.A == Abilities.A:
            pyautogui.keyDown('a')
        else:
            pyautogui.keyUp('a')

        if self.W == Abilities.W:
            pyautogui.keyDown('w')
        else:
            pyautogui.keyUp('w')

        if self.S == Abilities.S:
            pyautogui.keyDown('s')
        else:
            pyautogui.keyUp('s')

        if self.LOOK_UP == Abilities.LOOK_UP:
            #win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, -75, 0)
            pass
            
        if self.LOOK_DOWN == Abilities.LOOK_DOWN:
            #win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, 75, 0)
            pass

        if self.LOOK_LEFT == Abilities.LOOK_LEFT:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -200, 0)

        if self.LOOK_RIGHT == Abilities.LOOK_RIGHT:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 200, 0)


    def __repr__(self):
        return f"AbilityMove(shift: {self.SHIFT}, Q: {self.Q}, E: {self.E}, RCLICK: {self.RIGHT_CLICK}, \
            LCLICK: {self.LEFT_CLICK}, D: {self.D}, A: {self.A}, W: {self.W}, S: {self.S}, UP: {self.LOOK_UP}, \
            DOWN: {self.LOOK_DOWN}, LEFT: {self.LOOK_LEFT}, RIGHT: {self.LOOK_RIGHT})"


def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    model = torch.load('yolov7/yolov7-w6-pose.pt', map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        #print("Using CUDA")
        model.float().to(device)
    return model

def run_inference(image, model):
    #image = cv2.imread(url) # shape: (480, 640, 3)
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (768, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image).cuda() # torch.Size([3, 768, 960])
    # Turn image into batch
    image = image.unsqueeze(0) # torch.Size([1, 3, 768, 960])
    output, _ = model(image) # torch.Size([1, 45900, 57])
    return output, image

def visualize_output(output, image, model):
    output = non_max_suppression_kpt(output, 
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=model.yaml['nc'], # Number of Classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        keypoints = plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    return keypoints, nimg
    #plt.figure(figsize=(12, 12))
    #plt.axis('off')
    #plt.imshow(nimg)
    #plt.show()

def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    # Plot the skeleton and keypoints
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps
    
    keypoint = [[-1 , -1]] * 16
    
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
            cv2.putText(im, str(kid), (int(x_coord), int(y_coord)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            #print(f"X: {int(x_coord)}, Y: {int(y_coord)}")
            keypoint[kid] = ((int(x_coord), int(y_coord)))
            
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
    return keypoint


if __name__ == "__main__":
    main()


