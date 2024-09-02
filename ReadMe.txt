GestureToText(GTT) is a Python Sign language recognition application using action action recognition
that utilizes MediaPipe's Holistic model to recognize human actions from webcam video feeds. The application
provides a graphical user interface (GUI) for users to interact with the system, collect action data, and perform action recognition.

Features
Action Data Collection
-Collect action data from the webcam video feed
-Save collected data to a directory for future use
-Option to change the video source (e.g., from PC camera to external camera)

Action Recognition
-Load a trained model to recognize actions from the collected data
-Perform action recognition on the webcam video feed
-Display the recognized action in the GUI

GUI Features
-Start and stop buttons to control the action recognition process
-Add action button to collect new action data
-Clear button to clear the text output
-Menu bar with options to change the video source, train the model, and exit the application
-Text output to display the recognized actions

Technical Requirements
-Python 3.8 or later
-OpenCV 4.5 or later
-MediaPipe 0.8 or later
-TensorFlow 2.5 or later
-PyQt6 6.2 or later

Getting Started
Installation:
Clone the repository from GitHub.
Install the required dependencies using pip install -r requirements.txt.

Running the Application:
Execute the main script (main.py).
The GUI will appear, allowing you to interact with the application.


Usage
-Collecting Action Data
-Click the "Add Action" button to open the "Add Action" dialog.
-Enter the action name and click "OK" to start collecting data.
-Perform the action in front of the webcam, and the system will collect data for 20 sequences.
-The collected data will be saved to a directory named "Actions_Data".

Performing Action Recognition
-Click the "Start" button to start the action recognition process.
-Perform an action in front of the webcam, and the system will recognize the action and -display it in the text output.
-Click the "Stop" button to stop the action recognition process.

Troubleshooting
-If the application crashes or freezes, try restarting the application or checking the -system resources.
-If the action recognition is not accurate, try adjusting the model hyperparameters or collecting more data.


Contributing
Contributions are welcome! If you'd like to contribute to the project, please fork the repository and submit a pull request.

Acknowledgments
MediaPipe: A framework for building multimodal applied machine learning pipelines.
OpenCV: A computer vision library.
TensorFlow: A machine learning library.
PyQt6: A set of Python bindings for the Qt application framework.

Authors
[Abayitey Maxwell and Cobbinah Anthony] - Initial development and maintenance.
