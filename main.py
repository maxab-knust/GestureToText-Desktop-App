import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QImage, QPixmap,QIcon
from PyQt6.QtWidgets import QMessageBox, QSplashScreen
from PyQt6.QtCore import Qt
import sys
import warnings
import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import time
import threading
# Suppress DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

#-----------------------------GESTURETOTEXT DESKTOP APPLICATION-------------------------------------------
class Ui_SLRSMainWindow(object):
    def __init__(self, SLRSMainWindow):
        """
        Initialize the UI of the Sign Language Recognition System (SLRS) window.

        Parameters
        ----------
        SLRSMainWindow : QtWidgets.QMainWindow
            The main window object of the SLRS application.

        Attributes
        ----------
        action_thread : threading.Thread
            The thread that runs the action recognition function.
        stop_event : threading.Event
            The event that signals when to stop the action recognition function.
        """

        super().__init__()
        self.setupUi(SLRSMainWindow)
        self.action_thread = None  # Thread to run the action recognition function
        self.stop_event = threading.Event()  # Event to signal when to stop the recognition

    def setupUi(self, SLRSMainWindow):
        """
        Set up the UI for the Sign Language Recognition System (SLRS) window.

        This function sets up the UI for the SLRS window, including the main window, menu bar, and status bar.
        It also sets up the layout for the video output frame, text output frame, and buttons.

        Parameters
        ----------
        SLRSMainWindow : QtWidgets.QMainWindow
            The main window object of the SLRS application.

        Attributes
        ----------
        action_thread : threading.Thread
            The thread that runs the action recognition function.
        stop_event : threading.Event
            The event that signals when to stop the action recognition function.
        """

        SLRSMainWindow.setObjectName("SLRSMainWindow")
        SLRSMainWindow.setWindowIcon(QIcon('icon.png'))
        self.centralwidget = QtWidgets.QWidget(parent=SLRSMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setContentsMargins(-1, 0, -1, -1)
        self.gridLayout.setObjectName("gridLayout")
        
        # Horizontal layout for Start and Stop buttons
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.start_button = QtWidgets.QPushButton(parent=self.centralwidget)
        self.start_button.setEnabled(True)
        self.start_button.setMaximumSize(QtCore.QSize(16777215, 100))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.start_button.setFont(font)
        self.start_button.setObjectName("start_button")
        self.horizontalLayout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start_action_recognition)
        
        self.stop_button = QtWidgets.QPushButton(parent=self.centralwidget)
        self.stop_button.setEnabled(True)
        self.stop_button.setMaximumSize(QtCore.QSize(16777215, 100))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.stop_button.setFont(font)
        self.stop_button.setObjectName("stop_button")
        self.horizontalLayout.addWidget(self.stop_button)
        self.stop_button.clicked.connect(self.stop_action_recognition)
        
        self.gridLayout.addLayout(self.horizontalLayout, 2, 0, 1, 1)
        
        # Horizontal layout for Add Action and Clear buttons
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.addaction_button = QtWidgets.QPushButton(parent=self.centralwidget)
        self.addaction_button.setMaximumSize(QtCore.QSize(16777215, 100))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.addaction_button.setFont(font)
        self.addaction_button.setObjectName("addaction_button")
        self.horizontalLayout_2.addWidget(self.addaction_button)
        self.addaction_button.clicked.connect(self.show_add_action_dialog)  # Connect to the dialog function
        
        self.clear_button = QtWidgets.QPushButton(parent=self.centralwidget)
        self.clear_button.setMaximumSize(QtCore.QSize(16777215, 100))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.clear_button.setFont(font)
        self.clear_button.setObjectName("clear_button")
        self.horizontalLayout_2.addWidget(self.clear_button)
        self.clear_button.clicked.connect(self.clear)
        
        self.gridLayout.addLayout(self.horizontalLayout_2, 3, 0, 1, 1)
        
        # Video output frame
        self.vidoutput_frame = QtWidgets.QFrame(parent=self.centralwidget)
        self.vidoutput_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.vidoutput_frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.vidoutput_frame.setObjectName("vidoutput_frame")
        self.vid_label = QtWidgets.QLabel(parent=self.vidoutput_frame)
        self.vid_label.setGeometry(QtCore.QRect(90, 5, 431, 241))
        self.vid_label.setMinimumSize(700, 500)
        self.vid_label.setObjectName("vid_label")
        self.gridLayout.addWidget(self.vidoutput_frame, 0, 0, 2, 1)
        
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        
        # Text output
        self.textoutput = QtWidgets.QTextBrowser(parent=self.centralwidget)
        self.textoutput.setMaximumSize(QtCore.QSize(670, 16777215))
        self.textoutput.setObjectName("textoutput")
        self.gridLayout_2.addWidget(self.textoutput, 0, 1, 1, 1)
        
        #Customizing Textbox
        font.setPointSize(20)  
        self.textoutput.setFont(font)
        
        SLRSMainWindow.setCentralWidget(self.centralwidget)
        
        # Menu bar setup
        self.help_menu = QtWidgets.QMenuBar(parent=SLRSMainWindow)
        self.help_menu.setGeometry(QtCore.QRect(0, 0, 735, 22))
        self.help_menu.setObjectName("help_menu")
        self.file_menu = QtWidgets.QMenu(parent=self.help_menu)
        self.file_menu.setObjectName("file_menu")
        self.settings_menu = QtWidgets.QMenu(parent=self.help_menu)
        self.settings_menu.setObjectName("settings_menu")
        self.Change_Vid_Source_menu = QtWidgets.QMenu(parent=self.help_menu)   #vid source
        self.Change_Vid_Source_menu.setObjectName("Change_Vid_Source")
        self.menuHelp_2 = QtWidgets.QMenu(parent=self.help_menu)
        self.menuHelp_2.setObjectName("menuHelp_2")
        SLRSMainWindow.setMenuBar(self.help_menu)
        self.statusbar = QtWidgets.QStatusBar(parent=SLRSMainWindow)
        self.statusbar.setObjectName("statusbar")
        SLRSMainWindow.setStatusBar(self.statusbar)
        
        # Actions for file 
        self.actionOpenVid = QtGui.QAction(parent=SLRSMainWindow)
        self.actionOpenVid.setObjectName("actionOpenVid")
        self.actionStop = QtGui.QAction(parent=SLRSMainWindow)
        self.actionStop.setObjectName("actionStop")
        self.actionAdd_Action = QtGui.QAction(parent=SLRSMainWindow)
        self.actionAdd_Action.setObjectName("actionAdd_Action")
        self.actionClear = QtGui.QAction(parent=SLRSMainWindow)
        self.actionClear.setObjectName("actionClear")
        self.actionExit = QtGui.QAction(parent=SLRSMainWindow)
        self.actionExit.setObjectName("actionExit")
        
        # Actions for settings
        self.actionTrain_Model = QtGui.QAction(parent=SLRSMainWindow)
        self.actionTrain_Model.setObjectName("actionTrain_Model")
        
        # Actions for different video sources
        self.actionVid_source_0 = QtGui.QAction("Source 0", parent=SLRSMainWindow)
        self.actionVid_source_0.setObjectName("PC_CAM")
        self.actionVid_source_1 = QtGui.QAction("Source 1", parent=SLRSMainWindow)
        self.actionVid_source_1.setObjectName("External CAM")
        
        # Actions for settings
        self.actionHelp = QtGui.QAction(parent=SLRSMainWindow)
        self.actionHelp.setObjectName("actionHelp")
        
        #Add Actions for file menu
        self.file_menu.addAction(self.actionOpenVid)
        self.file_menu.addAction(self.actionStop)
        self.file_menu.addAction(self.actionAdd_Action)
        self.file_menu.addAction(self.actionClear)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.actionExit)
        
        #Add Actions to Settings menu
        self.settings_menu.addAction(self.actionTrain_Model)
        
        # Add actions to the Change Vid Source submenu
        self.Change_Vid_Source_menu.addAction(self.actionVid_source_0)
        self.actionVid_source_0.setCheckable(True)
        self.Change_Vid_Source_menu.addAction(self.actionVid_source_1)
        self.actionVid_source_1.setCheckable(True)
        
        self.menuHelp_2.addAction(self.actionHelp)
        
        #Add menus to menubar
        self.help_menu.addAction(self.file_menu.menuAction())
        self.help_menu.addAction(self.settings_menu.menuAction())
        self.help_menu.addAction(self.Change_Vid_Source_menu.menuAction())
        self.help_menu.addAction(self.menuHelp_2.menuAction())
        
        # Connect menu actions to functions
        self.actionOpenVid.triggered.connect(self.update)
        self.actionStop.triggered.connect(self.stop_action_recognition)
        self.actionExit.triggered.connect(self.closewindow)
        self.actionTrain_Model.triggered.connect(self.train_model)
        self.actionAdd_Action.triggered.connect(self.show_add_action_dialog) 
        self.actionVid_source_0.triggered.connect(lambda: self.change_vid_source(0))
        self.actionVid_source_1.triggered.connect(lambda: self.change_vid_source(1))
        self.actionHelp.triggered.connect(self.help)
        
        self.retranslateUi(SLRSMainWindow)
        QtCore.QMetaObject.connectSlotsByName(SLRSMainWindow)
        
        # Initialize video capture and holistic model
        self.vid_source=0      #default camera driver (pc cam)
        self.vid = cv2.VideoCapture(self.vid_source) 
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.running = False
        self.delay = 5  # Update delay in milliseconds

    def retranslateUi(self, SLRSMainWindow):
        """
        Retranslate the UI with the correct translations.
        This function is called whenever the language of the application changes.
        """
        _translate = QtCore.QCoreApplication.translate
        SLRSMainWindow.setWindowTitle(_translate("SLRSMainWindow", "GestureToText"))
        self.start_button.setText(_translate("SLRSMainWindow", "START"))
        self.stop_button.setText(_translate("SLRSMainWindow", "STOP"))
        self.addaction_button.setText(_translate("SLRSMainWindow", "ADD ACTION"))
        self.clear_button.setText(_translate("SLRSMainWindow", "CLEAR"))
        self.file_menu.setTitle(_translate("SLRSMainWindow", "File"))
        self.settings_menu.setTitle(_translate("SLRSMainWindow", "Settings"))
        self.Change_Vid_Source_menu.setTitle(_translate("SLRSMainWindow", "Change Vid Source"))
        self.menuHelp_2.setTitle(_translate("SLRSMainWindow", "Help"))
        self.actionOpenVid.setText(_translate("SLRSMainWindow", "Open Video"))
        self.actionStop.setText(_translate("SLRSMainWindow", "Stop"))
        self.actionAdd_Action.setText(_translate("SLRSMainWindow", "Add Action"))
        self.actionClear.setText(_translate("SLRSMainWindow", "Clear"))
        self.actionExit.setText(_translate("SLRSMainWindow", "Exit"))
        self.actionTrain_Model.setText(_translate("SLRSMainWindow", "Train Actions"))
        self.actionVid_source_0.setText(_translate("SLRSMainWindow", "PC CAM"))
        self.actionVid_source_1.setText(_translate("SLRSMainWindow", "External CAM"))
        self.actionHelp.setText(_translate("SLRSMainWindow", "Help"))

    def help(self):
        """
        Opens the ReadMe.txt file which contains information on how to use the application,
        how to collect data for training, and how to train the model.
        """
        os.startfile('ReadMe.txt')
        
    def clear(self):
        """
        Clears the text output area.
        """
        self.textoutput.clear()
        
    def closeEvent(self, event):
        """
        Overrides the closeEvent of the QMainWindow to release the camera and close the MediaPipe holistic model when the window is closed.
        """
        self.vid.release()  # Release the camera
        self.holistic.close()  # Close the MediaPipe holistic model
        super().closeEvent(event)
    
    def closewindow(self):
        """
        Closes the application window and exits the application.
        """
        sys.exit()
        
    def change_vid_source(self, source):
        """Change the video source to the selected source and show as checked."""
        if source == 0:                                     #PC CAM
            self.actionVid_source_0.setChecked(True)
            self.actionVid_source_1.setChecked(False)
        else:                                              #EXTERNAL CAM                     
            self.actionVid_source_0.setChecked(False)
            self.actionVid_source_1.setChecked(True)       
        self.vid.release()  # Release the current video capture object
        self.vid = cv2.VideoCapture(source)  # Change the video source
        self.update()  # Update the display with the new video source
        
    def start_action_recognition(self):
        if self.action_thread is None or not self.action_thread.is_alive():
            self.stop_event.clear()
            self.action_thread = threading.Thread(target=self.run_action_recognition)
            self.action_thread.start()
            self.start_button.setDisabled(True)

    def stop_action_recognition(self):
        """
        Stops the action recognition thread and clears the video label and text output area when the stop button is clicked.
        """
        if self.action_thread and self.action_thread.is_alive():
            self.stop_event.set()
            self.action_thread.join()
            self.start_button.setDisabled(False)        
        self.vid_label.setPixmap(QtGui.QPixmap())
        
    def run_action_recognition(self):
        """
        Starts the action recognition process in a separate thread using the provided stop event, video label, and text output area.

        The action recognition process uses the MediaPipe Holistic model to make detections and a trained model to predict the action.

        The video label and text output area are updated in real-time with the current action being recognized.

        The action recognition process is stopped when the stop button is clicked or the window is closed.
        """
        action_recognition(self.stop_event, self.vid_label, self.textoutput, self.vid_source)
        
    def update(self):
        """
        Updates the video label with the current frame from the video source using the MediaPipe Holistic model.

        The update process is done in real-time and uses a single shot timer to call itself again after a delay.

        The video label is updated with the current frame from the video source and the detections from the MediaPipe Holistic model.

        The update process is stopped when the window is closed.
        """
        if not self.running:
            return
        self.running=True
        ret, frame = self.vid.read()
        if ret:
            frame, results = mediapipe_detection(frame, self.holistic)
            draw_styled_landmarks(frame, results)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.vid_label.setPixmap(QPixmap.fromImage(q_image))
        QtCore.QTimer.singleShot(self.delay, self.update)
    
    def show_success_dialog(self):
        """
        Shows a success dialog with the title "Success" and the message "Operation completed successfully."

        This method is used to show a success dialog after a successful operation, such as collecting action data or training a model.

        The dialog has a single "Ok" button and will close automatically when the button is clicked.
        """
        msg_box = QMessageBox()
        msg_box.setWindowIcon(QIcon('icon.png'))
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setWindowTitle("Success")
        msg_box.setText("Operation completed successfully.")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()
    
    def show_error_dialog(self, message, title):
        """
        Shows an error dialog with the specified message and title.

        This method is used to show an error dialog after an operation fails, such as collecting action data or training a model.

        The dialog has a single "Ok" button and will close automatically when the button is clicked.

        Args:
            message (str): The main message text to display in the dialog box.
            title (str): The title of the dialog box.
        """
        # Create an instance of QMessageBox
        msg_box = QMessageBox()
        msg_box.setWindowIcon(QIcon('icon.png'))

        # Set the icon to Critical for an error message
        msg_box.setIcon(QMessageBox.Icon.Critical)

        # Set the main message text
        msg_box.setText(message)

        # Set the title of the dialog box
        msg_box.setWindowTitle(title)

        # Add a standard OK button to close the dialog
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)

        # Display the message box
        msg_box.exec()      
        
    def collect_action_data(self, action, no_sequences=20, sequence_length=30, min_detection_confidence=0.5, min_tracking_confidence=0.5, vid_source=0):
        """
        Collects action data from the webcam video feed and saves it to a directory.

        This method uses the MediaPipe holistic model to detect the pose, face, and hand landmarks of a person in the video feed.
        It collects data for 20 sequences, where each sequence is 30 frames long.
        The collected data is saved to a directory named "Actions_Data" in the current working directory.

        The method also displays the recognized action in the GUI.

        Args:
            action (str): The name of the action to collect data for.
            no_sequences (int): The number of sequences to collect data for. Defaults to 20.
            sequence_length (int): The length of each sequence in frames. Defaults to 30.
            min_detection_confidence (float): The minimum confidence threshold to detect the pose and hand landmarks. Defaults to 0.5.
            min_tracking_confidence (float): The minimum confidence threshold to track the pose and hand landmarks. Defaults to 0.5.
            vid_source (int): The video source to use. Defaults to 0, which is the default camera.
        """
        data_path = os.path.join('Actions_Data')
        actions = [action]

        # Create the data directory if it doesn't exist
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        # Collect data for action using the mediapipe holistic model
        cap = cv2.VideoCapture(vid_source)
        with mp_holistic.Holistic(min_detection_confidence=min_detection_confidence,
                                  min_tracking_confidence=min_tracking_confidence) as holistic:
            for action in actions:
                action_dir = os.path.join(data_path, action)
                if not os.path.exists(action_dir):
                    os.makedirs(action_dir)
                    
                for countdown in range(5, 0, -1):
                        ret, frame = cap.read()
                        cv2.putText(frame, f'Starting in {countdown}', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', frame)
                        cv2.waitKey(1000)
                        
                for sequence in range(no_sequences):
                    sequence_dir = os.path.join(action_dir, str(sequence))
                    if not os.path.exists(sequence_dir):
                        os.makedirs(sequence_dir)
                            
                    for frame_num in range(sequence_length):
                        ret, frame = cap.read()
                        image, results = self.mediapipe_detection(frame, holistic)
                        self.draw_styled_landmarks(image, results)

                        if frame_num == 0:
                            cv2.putText(image, 'STARTING DATA COLLECTION', (120, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, f'Collecting data frames for {action} Video Number {sequence}', (15, 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(2000)
                        else:
                            cv2.putText(image, f'Collecting data frames for {action} Video Number {sequence}', (15, 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', image)

                        keypoints = self.extract_keypoints(results)
                        npy_path = os.path.join(sequence_dir, str(frame_num) + '.npy')
                        np.save(npy_path, keypoints)
                        #Skip the data collection
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
        cap.release()
        cv2.destroyAllWindows()
        
        #take vid source to default (ie. pc cam)
        self.vid = cv2.VideoCapture(0)  # Change the video source
        self.update()

    def show_add_action_dialog(self):
        """
        Create the Add Action Dialog

        This function creates a dialog for users to input a new action name.
        The dialog has a text input field and an OK and Cancel button.
        If the user clicks OK, this function will call collect_action_data
        to start collecting data for the new action.
        """
        # Create the Add Action Dialog
        AddActionDialog = QtWidgets.QDialog()
        AddActionDialog.setObjectName("AddActionDialog")
        AddActionDialog.setEnabled(True)
        AddActionDialog.resize(393, 134)
        AddActionDialog.setWindowIcon(QIcon('icon.png'))
        
        buttonBox = QtWidgets.QDialogButtonBox(parent=AddActionDialog)
        buttonBox.setGeometry(QtCore.QRect(40, 80, 341, 32))
        buttonBox.setAutoFillBackground(False)
        buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        buttonBox.setCenterButtons(True)
        buttonBox.setObjectName("buttonBox")
        
        text_actionname = QtWidgets.QLineEdit(parent=AddActionDialog)
        text_actionname.setGeometry(QtCore.QRect(10, 30, 371, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        text_actionname.setFont(font)
        text_actionname.setObjectName("text_actionname")
        
        label = QtWidgets.QLabel(parent=AddActionDialog)
        label.setGeometry(QtCore.QRect(20, 0, 81, 31))
        label.setObjectName("label")
        
        _translate = QtCore.QCoreApplication.translate
        AddActionDialog.setWindowTitle(_translate("AddActionDialog", "Add Action"))
        label.setText(_translate("AddActionDialog", "ACTION NAME:"))
        
        buttonBox.accepted.connect(AddActionDialog.accept)
        buttonBox.rejected.connect(AddActionDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(AddActionDialog)
        
        # Show the dialog and get the action name if accepted
        if AddActionDialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            action_name = text_actionname.text().upper().strip()
            actions=action_names("Actions_Data")
            if not action_name:
                message="Action name cannot be empty"
                self.show_error_dialog(message, 'Error')
            elif action_name in actions:
                message="Name already exist in database"
                self.show_error_dialog(message, 'Error')
            else:
                self.collect_action_data(action_name)

    def train_model(self):
        """
        Train a model using the collected action data.

        This function trains a model using the collected action data in the 'Actions_Data' directory. The model is a
        sequence-to-sequence model using LSTMs, and is trained using the Adam optimizer and categorical cross-entropy
        loss. The model is saved as 'Action.h5' in the current working directory.
        """
        # Initialize PyQt application
        app = QtWidgets.QApplication([])

        # Set up parameters for training
        data_path = 'Actions_Data'
        sequence_length = 30
        test_size = 0.05
        epochs = 2000

        # Get action labels
        actions = np.array(action_names(data_path))
        print(f"Actions: {actions}")

        # Label mapping
        label_map = {label: num for num, label in enumerate(actions)}

        # Prepare sequences and labels
        sequences, labels = [], []
        for action in actions:
            action_path = os.path.join(data_path, action)
            for sequence in np.array(os.listdir(action_path)).astype(int):
                window = []
                for frame_num in range(sequence_length):
                    res = np.load(os.path.join(action_path, str(sequence), f"{frame_num}.npy"))
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])

        X = np.array(sequences)
        y = keras.utils.to_categorical(labels).astype(int)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        #locate logs directory 
        log_dir = os.path.join('Logs')
        tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir) #to store training data 
        
        # Define the model
        model = keras.models.Sequential([
            keras.layers.Input(shape=(X.shape[1], X.shape[2])),
            keras.layers.LSTM(64, return_sequences=True, activation='relu'),
            keras.layers.LSTM(128, return_sequences=True, activation='relu'),
            keras.layers.LSTM(64, return_sequences=False, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(actions.shape[0], activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        # Train the model
        try:
            model.fit(X_train, y_train, epochs, verbose=0, callbacks=[tb_callback])
            model.save('Action.h5')
            print("Model training complete and saved as 'Action.h5'")

            # Evaluate the model
            yhat = model.predict(X_test)
            ytrue = np.argmax(y_test, axis=1).tolist()
            yhat = np.argmax(yhat, axis=1).tolist()
            accuracy = accuracy_score(ytrue, yhat)
            self.textoutput.insertPlainText(f"Accuracy: {accuracy} \n Training completed...")
            self.statusbar.showMessage("Training completed...")
            
            self.show_success_dialog()

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            self.show_error_dialog("Error training model", "Training Error")
            return None

def action_names(directory):
    """Retrieve the names of action subfolders from the specified directory."""
    subfolders = []
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                subfolders.append(item)
    except FileNotFoundError:
        print(f"Directory {directory} not found.")
    return subfolders

def mediapipe_detection(image, model):
    """
    Process the image and perform holistic detection using MediaPipe.

    Parameters
    ----------
    image : numpy array
        The image to be processed.
    model : mediapipe.solutions.holistic.Holistic
        The MediaPipe Holistic model.

    Returns
    -------
    image : numpy array
        The processed image.
    results : mediapipe.solutions.holistic.Holistic
        The results of the holistic detection.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw pose, left and right hands landmarks
    """
    Draw pose, left hand, and right hand connections on the image using the MediaPipe Drawing Utilities.

    Parameters
    ----------
    image : numpy array
        The image to draw on.
    results : mediapipe.solutions.holistic.Holistic
        The results of the holistic detection.

    Returns
    -------
    image : numpy array
        The processed image.
    """
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    """
    Extract keypoints from the results for pose, left hand, and right hand.

    Parameters
    ----------
    results : mediapipe.solutions.holistic.Holistic
        The results of the holistic detection.

    Returns
    -------
    keypoints : numpy array
        The extracted keypoints.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])


def action_recognition(stop_event, vid_label, textoutput, vid_source):
    """Perform action recognition using the webcam feed and a trained model."""
    try:
        # Load the trained model
        model = keras.models.load_model('Action.h5')
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])  

        # Get action names
        directory_path = 'Actions_Data'
        actions = np.array(action_names(directory_path))

        # Set up MediaPipe Holistic model
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Initialize variables
        sequence = []
        sentence = []
        threshold = 0.6
        window_size = 30
        window_stride = 10

        # Capture video from the webcam
        cap = cv2.VideoCapture(vid_source)

        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # Perform action prediction
            if len(sequence) >= window_size:
                window_sequence = sequence[-window_size:]
                res = model.predict(np.expand_dims(window_sequence, axis=0))[0]
                action = actions[np.argmax(res)]
                confidence = res[np.argmax(res)]
                
                if confidence > threshold:
                    if len(sentence) > 0:
                        if action != sentence[-1]:
                            sentence.append(action)
                            print("New action detected:", action)  # Print the new action
                            textoutput.insertPlainText(f"{action} ")
                    else:
                        sentence.append(action)
                        print("First action detected:", action)
                        textoutput.insertPlainText(f"{action} ")

                # Remove old frames from the sequence
                sequence = sequence[-window_stride:]

            # Convert the image back to BGR for display
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            vid_label.setPixmap(QPixmap.fromImage(q_img))

            # Display results on the screen
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence[-10:]), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            #cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        holistic.close()

    except Exception as e:
        print(f"An error occurred: {e}")

    print('Action recognition complete.')

def main():
    """
    Main entry point of the application.

    This function initializes the application, creates a splash screen, and then
    opens the main window. It also sets up the UI elements and starts the action
    recognition thread.

    :return: None
    """
    app = QtWidgets.QApplication(sys.argv)
    # Create and display the splash screen
    pixmap = QPixmap("splash.png")  # Replace with your splash image path
     # Resize the pixmap to the desired size
    resized_pixmap = pixmap.scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio)  # Example size 400x300
    splash = QSplashScreen(resized_pixmap)
    splash.show()
    # Simulate a delay for loading resources (e.g., database, files, etc.)
    time.sleep(2)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_SLRSMainWindow(MainWindow)
    MainWindow.showMaximized()  # Ensure the main window opens maximized
    # Close the splash screen
    splash.finish(MainWindow)
    ui.actionVid_source_0.setChecked(True)
    ui.running = True
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
