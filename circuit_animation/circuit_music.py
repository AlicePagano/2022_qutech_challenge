
import musicalbeeps as mb


# Air on thr G string
notes = ["F4#", "B4", "G4", "E4", "D4", "C4#", "D4", "C4#", "B3", "A3", "A4", "F4#", "C4", "B3", "E4", "D4#", "A4","G4"]
tempo = [18,      1,    1,    1,    1,    1,    1,     4,    1,    3,    8,   1,     1,   1 ,   1,  1,      1,    1]
notes += ["G4", "E4", "B3", "A3", "D4", "C4#", "G4", "F4#", "F4#", "G#4", "A4", "D4",   "D4",   "E4",  "F4#", "F4#", "E4","E4", "D4", "C4#", "B3", "B3", "C4#", "D4", "D4", "C4#", "B3", "A3"]
tempo += [8,      1,    1,    1,    1,    1,    1,     1,    6,    1,      1,     2,     0.5,   0.5 ,   1,     1,      1,    1,  1,   1,    1,    0.5,   0.5,   1,    2,     1,    1,   8]

error = mb.Player(volume = 0.8, mute_output=True)
bach = mb.Player(volume = 0.3, mute_output=False)
for n, tune in enumerate(notes):

	# Here can include actual circuit measured error sequence
    if (n+5)% 8 == 0:
        error.play_note("Eb", 0.3)
    
    bach.play_note(tune, tempo[n]/3.)