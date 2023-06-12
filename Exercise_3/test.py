import PySimpleGUI as sg
import os

# Define the GUI layout
layout = [
    [sg.Text('Select a file:')],
    [sg.Input(key='-FILE-', enable_events=True), sg.FileBrowse()],
    [sg.Button('Submit')]
]

# Create the GUI window
window = sg.Window('File Selection', layout)

# Event loop
while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED:
        break
    elif event == 'Submit':
        filename = values['-FILE-']
        basename = os.path.basename(filename)
        print('Selected file:', basename)
        break

# Close the window
window.close()
