# This is a sample Python script.
import musicalbeeps
import threading
import time
from music21 import *
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend as K
import random

#specify the path
path = 'Beethoven/'

#read all the filenames
files = [i for i in os.listdir(path) if i.endswith(".mid")]



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


# defining function to read MIDI files
def read_midi(file):
    print("Loading Music File:", file)

    notes = []
    notes_to_parse = None

    # parsing a midi file
    midi = converter.parse(file)

    # grouping based on different instruments
    s2 = instrument.partitionByInstrument(midi)

    # Looping over all the instruments
    for part in s2.parts:

        # select elements of only piano
        if 'Piano' in str(part):

            notes_to_parse = part.recurse()

            # finding whether a particular element is note or a chord
            for element in notes_to_parse:

                # note
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))

                # chord
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

    return np.array(notes)


def lstm():
    model = Sequential()
    model.add(LSTM(128,return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(K.n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model


def convert_to_midi(prediction_output):
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:

        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                cn = int(current_note)
                new_note = note.Note(cn)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)

            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        # pattern is a note
        else:

            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 1
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='music.mid')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # threader = ThreadR(1, "Thread-R")
    # threadless = ThreadL(2, "Thread -L")

    # threader.start()
    # threadless.start()
    # time.sleep(6.5)
    # threader = threader.clone()
    # threadless = threadless.clone()
    # threader.start()
    # threadless.start()
    # time.sleep(6.5)
    # threader = threader.clone()
    # threadless = threadless.clone()
    # threader.start()
    # threadless.start()
    # time.sleep(6.5)
    # mario()
    # reading each midi file
    notes_array = np.array([read_midi(path + i) for i in files], dtype=object)

    # converting 2D array into 1D array
    notes_ = [element for note_ in notes_array for element in note_]

    # No. of unique notes
    unique_notes = list(set(notes_))
    print(len(unique_notes))

    # computing frequency of each note
    freq = dict(Counter(notes_))

    # consider only the frequencies
    no = [count for _, count in freq.items()]
    print(no)
    plt.hist(no)
    plt.show()
    frequent_notes = [note_ for note_, count in freq.items() if count >= 50]
    print(len(frequent_notes))
    new_music = []

    for notes in notes_array:
        temp = []
        for note_ in notes:
            if note_ in frequent_notes:
                temp.append(note_)
        new_music.append(temp)

    new_music = np.array(new_music, dtype=object)
    no_of_timesteps = 32
    x = []
    y = []

    for note_ in new_music:
        for i in range(0, len(note_) - no_of_timesteps, 1):
            # preparing input and output sequences
            input_ = note_[i:i + no_of_timesteps]
            output = note_[i + no_of_timesteps]

            x.append(input_)
            y.append(output)

    x = np.array(x)
    y = np.array(y)
    unique_x = list(set(x.ravel()))
    x_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_x))
    # preparing input sequences
    x_seq = []
    for i in x:
        temp = []
        for j in i:
            # assigning unique integer to every note
            temp.append(x_note_to_int[j])
        x_seq.append(temp)

    x_seq = np.array(x_seq)
    unique_y = list(set(y))
    y_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_y))
    y_seq = np.array([y_note_to_int[i] for i in y])
    x_tr, x_val, y_tr, y_val = train_test_split(x_seq, y_seq, test_size=0.2, random_state=0)

    K.clear_session()
    model = Sequential()

    # embedding layer
    model.add(Embedding(len(unique_x), 100, input_length=32, trainable=True))

    model.add(Conv1D(64, 3, padding='causal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))

    model.add(Conv1D(128, 3, activation='relu', dilation_rate=2, padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))

    model.add(Conv1D(256, 3, activation='relu', dilation_rate=4, padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))

    # model.add(Conv1D(256,5,activation='relu'))
    model.add(GlobalMaxPool1D())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(len(unique_y), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    model.summary()
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    history = model.fit(np.array(x_tr), np.array(y_tr), batch_size=128, epochs=50,
                        validation_data=(np.array(x_val), np.array(y_val)), verbose=1, callbacks=[mc])

    model = load_model('best_model.h5')
    ind = np.random.randint(0, len(x_val) - 1)

    random_music = x_val[ind]

    predictions = []
    for i in range(10):
        random_music = random_music.reshape(1, no_of_timesteps)

        prob = model.predict(random_music)[0]
        y_pred = np.argmax(prob, axis=0)
        predictions.append(y_pred)

        random_music = np.insert(random_music[0], len(random_music[0]), y_pred)
        random_music = random_music[1:]

    print(predictions)

    x_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_x))
    predicted_notes = [x_int_to_note[i] for i in predictions]
    convert_to_midi(predicted_notes)

    print('Happy birthday!')


