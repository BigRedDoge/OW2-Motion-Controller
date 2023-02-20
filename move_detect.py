import mediapipe as mp

from abilities import Abilities, AbilityMove
from math_utils import find_distance, find_angle

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