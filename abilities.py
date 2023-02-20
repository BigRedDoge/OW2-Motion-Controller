import pyautogui
import win32api, win32con
from enum import Enum

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