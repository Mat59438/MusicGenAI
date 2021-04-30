# This is a sample Python script.
import musicalbeeps
import threading
import time


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

player = musicalbeeps.Player(volume = 0.3, mute_output = False)
playerL = musicalbeeps.Player(volume = 0.3, mute_output = False)
playerR = musicalbeeps.Player(volume = 0.3, mute_output = False)

class ThreadR (threading.Thread):
    def __init__(self, threadID, name):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
    def run(self):
      print("Starting " + self.name)
      right_elis()
      print("Exiting " + self.name)
    def clone(self):
        return ThreadR(self.threadID, self.name)

class ThreadL (threading.Thread):
    def __init__(self, threadID, name):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
    def run(self):
      print("Starting " + self.name)
      left_elis()
      print("Exiting " + self.name)
    def clone(self):
        return ThreadL(self.threadID, self.name)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def birthday():
    player.play_note("C", 0.125)
    player.play_note("C", 0.125)
    player.play_note("D", 0.25)
    player.play_note("C", 0.25)
    player.play_note("F", 0.25)
    player.play_note("E", 0.5)
    player.play_note("C", 0.125)
    player.play_note("C", 0.125)
    player.play_note("D", 0.25)
    player.play_note("C", 0.25)
    player.play_note("G", 0.25)
    player.play_note("F", 0.5)
    player.play_note("C", 0.125)
    player.play_note("C", 0.125)
    player.play_note("C4", 0.25)
    player.play_note("A", 0.25)
    player.play_note("F", 0.25)
    player.play_note("E", 0.25)
    player.play_note("D", 0.25)
    player.play_note("B", 0.125)
    player.play_note("B", 0.125)
    player.play_note("A", 0.25)
    player.play_note("F", 0.25)
    player.play_note("G", 0.25)
    player.play_note("F", 0.5)

def right_elis():
    playerL.play_note("E5", 0.125)
    playerL.play_note("D5#", 0.125)
    playerL.play_note("E5", 0.125)
    playerL.play_note("D5#", 0.125)
    playerL.play_note("E5", 0.125)
    playerL.play_note("B5", 0.125)
    playerL.play_note("D5b", 0.125)
    playerL.play_note("C5", 0.125)
    playerL.play_note("A4", 0.25)
    playerL.play_note("pause", 0.25)
    playerL.play_note("C4", 0.125)
    playerL.play_note("E4", 0.125)
    playerL.play_note("A4", 0.125)
    playerL.play_note("B4", 0.25)
    playerL.play_note("pause", 0.25)
    playerL.play_note("E4", 0.125)
    playerL.play_note("G4#", 0.125)
    playerL.play_note("B4", 0.125)
    playerL.play_note("C5", 0.25)
    playerL.play_note("pause", 0.25)
    playerL.play_note("E4", 0.125)
    playerL.play_note("E5", 0.125)
    playerL.play_note("D5#", 0.125)
    playerL.play_note("E5", 0.125)
    playerL.play_note("D5#", 0.125)
    playerL.play_note("E5", 0.125)
    playerL.play_note("B5", 0.125)
    playerL.play_note("D5b", 0.125)
    playerL.play_note("C5", 0.125)
    playerL.play_note("A4", 0.25)

def left_elis():
    playerR.play_note("pause", 1)
    playerR.play_note("A2", 0.125)
    playerR.play_note("E3", 0.125)
    playerR.play_note("A3", 0.125)
    playerR.play_note("pause", 0.375)
    playerR.play_note("E2", 0.125)
    playerR.play_note("G3#", 0.125)
    playerR.play_note("pause", 0.375)
    playerR.play_note("A2", 0.125)
    playerR.play_note("E3", 0.125)
    playerR.play_note("A3", 0.125)

def mario():
    player.play_note("E5", 0.125)
    player.play_note("E5", 0.25)
    player.play_note("E5", 0.25)
    player.play_note("C5", 0.125)
    player.play_note("E5", 0.25)
    player.play_note("G5", 0.25)
    player.play_note("G4", 0.25)

    player.play_note("pause", 0.125)
    player.play_note("C5", 0.25)
    player.play_note("pause", 0.125)
    player.play_note("G4", 0.25)
    player.play_note("pause", 0.125)
    player.play_note("E4", 0.25)
    player.play_note("pause", 0.125)
    player.play_note("A4", 0.25)
    player.play_note("B4", 0.25)
    player.play_note("B4b", 0.125)
    player.play_note("A4", 0.25)
    player.play_note("G4", 0.125)
    player.play_note("E5", 0.25)
    player.play_note("G5", 0.125)
    player.play_note("A5", 0.25)
    player.play_note("F5", 0.125)
    player.play_note("G5", 0.125)
    player.play_note("pause", 0.125)
    player.play_note("E5", 0.25)
    player.play_note("C5", 0.125)
    player.play_note("D5", 0.125)
    player.play_note("B4", 0.25)

    player.play_note("pause", 0.125)
    player.play_note("C5", 0.25)
    player.play_note("pause", 0.125)
    player.play_note("G4", 0.25)
    player.play_note("pause", 0.125)
    player.play_note("E4", 0.25)
    player.play_note("pause", 0.125)
    player.play_note("A4", 0.25)
    player.play_note("B4", 0.25)
    player.play_note("B4b", 0.125)
    player.play_note("A4", 0.25)
    player.play_note("G4", 0.125)
    player.play_note("E5", 0.25)
    player.play_note("G5", 0.125)
    player.play_note("A5", 0.25)
    player.play_note("F5", 0.125)
    player.play_note("G5", 0.125)
    player.play_note("pause", 0.125)
    player.play_note("E5", 0.25)
    player.play_note("C5", 0.125)
    player.play_note("D5", 0.125)
    player.play_note("B4", 0.25)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    threader = ThreadR(1, "Thread-R")
    threadless = ThreadL(2, "Thread -L")

    #threader.start()
    #threadless.start()
    #time.sleep(6.5)
    #threader = threader.clone()
    #threadless = threadless.clone()
    #threader.start()
    #threadless.start()
    #time.sleep(6.5)
    #threader = threader.clone()
    #threadless = threadless.clone()
    #threader.start()
    #threadless.start()
    #time.sleep(6.5)
    mario()

    #print('Happy birthday!')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
