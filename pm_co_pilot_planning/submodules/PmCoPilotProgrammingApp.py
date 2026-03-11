import sys
import time
import os
from rclpy.node import Node

import speech_recognition as sr
import pyttsx3
from pydub import AudioSegment
from pydub.playback import play
import io
import tempfile
from openai import OpenAI

from PyQt6.QtGui import QCursor, QIcon, QAction, QFont, QPalette, QColor, QTextCursor, QTextBlockFormat, QTextCharFormat, QFontMetrics
from PyQt6.QtWidgets import QComboBox, QSizePolicy, QLabel, QWidgetAction, QMenuBar, QDialog, QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QStyleFactory
from PyQt6.QtCore import Qt, QEvent, QObject, pyqtSignal, QThread, QSize, QRect, QPoint, pyqtSlot

from pm_co_pilot_planning.submodules.agent import Agent
from ros_sequential_action_programmer.submodules.RosSequentialActionProgrammer import RosSequentialActionProgrammer

class SpeechWorker(QObject):
    
    # textReceived = pyqtSignal(str)
    errorOccurred = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, service_node: Node):
        super().__init__()
        self.service_node = service_node
        
        self.client = OpenAI()
        self.client.api_key = os.environ["OPENAI_API_KEY"]

    def run(self):
        # Initialize the recognizer
        r = sr.Recognizer()
        with sr.Microphone() as source:
            self.service_node.get_logger().info("Speech Recognition initialized!")
            try:
                audio = r.listen(source,timeout=2,phrase_time_limit=20)
                
                # Convert the audio data to an audio file format (e.g., WAV)
                audio_data = audio.get_wav_data()
                # Use io.BytesIO to create a file-like object from the byte data
                wav_audio = io.BytesIO(audio_data)

                # Load the audio file using pydub (interprets the BytesIO object as a WAV file)
                sound = AudioSegment.from_file(wav_audio, format="wav")

                # Convert the audio to MP3 and save to an in-memory file
                mp3_audio = io.BytesIO()
                sound.export(mp3_audio, format="mp3")
                mp3_audio.seek(0) 

                with open("output.mp3", "wb") as mp3_file:
                    mp3_file.write(mp3_audio.getvalue())

                audio_file = open("output.mp3", "rb")

                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )

                # Use pydub to play back the audio file
                # sound = AudioSegment.from_file(audio_file, format="wav")
                # play(sound)  # This plays the sound to the user

                self.service_node.get_logger().info(f"Audio recorded {transcript}")

                # text = r.recognize_whisper(audio)
                # self.textReceived.emit(transcript.text)

            except sr.UnknownValueError:
                self.errorOccurred.emit("Speech Recognition could not understand audio")
            except sr.RequestError as e:
                self.errorOccurred.emit(f"Could not request results from Speech Recognition service; {e}")

        self.finished.emit(transcript.text)
        
class MessageWorker(QObject):
    finished = pyqtSignal(str)  # Signal to emit the response
    update_status = pyqtSignal(str) # Signal to emit status updates
    chunk_received = pyqtSignal(str)  # Signal to emit each chunk of the response
    sequence_modified = pyqtSignal()  # Signal to emit when sequence is modified

    def __init__(self, agent, message, service_node):
        super().__init__()
        self.agent = agent
        self.message = message
        self.service_node = service_node

    def run(self):
        self.update_status.emit("Agent working...")

        final_response = self.agent.handle_user_input(self.message)
        self.service_node.get_logger().info(f"Final response: {final_response}")
        self.update_status.emit("Ready for your input")
        # Emit signal to refresh GUI after agent completes
        self.sequence_modified.emit()
        self.finished.emit(final_response)

class ChatDisplay(QTextEdit):
    def __init__(self, parent=None):
        super(ChatDisplay, self).__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Helvetica", 12))

    def append_question(self, speaker, text):
        self._append_text(speaker, text, QColor("#FFA07A"))  # Light Salmon color for questions

    def append_reply(self, speaker, text):
        self._append_text(speaker, text, QColor("#7FFFD4"))  # Aquamarine color for replies

    def _append_text(self,speaker, text, color):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        format = QTextCharFormat()
        format.setForeground(color)
        cursor.setCharFormat(format)
        cursor.insertText(speaker + text + '\n')
        self.setTextCursor(cursor)

    def clear(self) -> None:
        return super().clear()
    
class ConfirmationDialog(QDialog):
    def __init__(self, question, parent=None):
        super().__init__(parent)

        self.setWindowTitle("User Confirmation")
        self.setFixedSize(200, 100) 

        # Layout
        layout = QVBoxLayout(self)

        # Add a label
        label = QLabel(question)
        layout.addWidget(label)

        # Yes and No buttons
        self.yes_button = QPushButton("Yes", self)
        self.yes_button.clicked.connect(self.accept)
        layout.addWidget(self.yes_button)

        self.no_button = QPushButton("No", self)
        self.no_button.clicked.connect(self.reject)
        layout.addWidget(self.no_button)

class AdapterSelectionDialog(QDialog):
    """
    A dialog that prompts the user to select an LLM adapter before the main window is shown.
    """
    def __init__(self, adapter_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Your Adapter")

        self.adapter_names = adapter_names
        self.selected_adapter = None  # Will store the user's choice

        layout = QVBoxLayout()

        # ComboBox
        self.combo = QComboBox()
        self.setGeometry(200, 200, 500, 300)
        self.combo.addItems(self.adapter_names)
        layout.addWidget(QLabel("Please select an adapter:"))
        layout.addWidget(self.combo)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.on_ok_clicked)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def on_ok_clicked(self):
        # Store the user selection and close the dialog with accept()
        self.selected_adapter = self.combo.currentText()
        self.accept()


class AdapterLoadWorker(QObject):
    initializationComplete = pyqtSignal(str, object)  # Signal to emit when initialization is complete, passing the initialized assistantAPI
    finished = pyqtSignal()  # Signal to emit when the worker is done

    def __init__(self, adapter_name: str, factory_callable, parent: QObject = None):
        super().__init__()
        self.adapter_name = adapter_name
        self.factory_callable = factory_callable

    def run(self):
        adapter_instance = self.factory_callable()

        # Emit signal once initialization is complete
        self.initializationComplete.emit(self.adapter_name, adapter_instance)     

        self.finished.emit()

class PmCoPilotProgrammingApp(QMainWindow):
    sequence_modified = pyqtSignal()  # Signal to notify parent when sequence changes
    
    def __init__(self, service_node:Node, rsap_instance: RosSequentialActionProgrammer = None):
        super().__init__()

        self.setupUI()
        self.apply_style()

        self.update_status_display("Initialization of Assisstant")

        # self.speech_engine = pyttsx3.init()
        
        self.service_node = service_node
        self.rsap_instance = rsap_instance

        # Create agent with shared RSAP instance
        self.agent = Agent(service_node=service_node, thread_id="thread_1", rsap_instance=rsap_instance)

        self.update_status_display("Assistant initialized and ready for your input!")



    def setupUI(self):
        # Apply a style suitable for dark mode
        self.setStyle(QStyleFactory.create("Fusion"))

        # Set the main window's properties
        self.setWindowTitle("PmCoPilot")
        self.setGeometry(100, 100, 800, 600)
        # self.setWindowIcon(QIcon("path_to_icon.png"))  # Set an Apple-style icon

        # Apply a palette for dark mode
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#1e1e1e"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#2e2e2e"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
        self.setPalette(palette)

        # Create the central widget and set the layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create the chat history display
        self.chat_history = ChatDisplay()
        self.layout.addWidget(self.chat_history)

        # Add a label for displaying status
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Helvetica", 12))
        self.layout.addWidget(self.status_label)

        # Add a text edit for typing messages
        self.message_input = QTextEdit()
        self.message_input.setFont(QFont("Helvetica", 12))
        self.message_input.installEventFilter(self)
        # Calculate the height for 4 rows of text
        fontMetrics = QFontMetrics(self.message_input.font())
        textHeight = fontMetrics.lineSpacing() * 8  # 4 rows of text
        self.message_input.setFixedHeight(textHeight)

        # Set size policy: allow horizontal resizing but keep vertical size fixed
        sizePolicy = self.message_input.sizePolicy()
        sizePolicy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        sizePolicy.setVerticalPolicy(QSizePolicy.Policy.Fixed)
        self.message_input.setSizePolicy(sizePolicy)
        self.layout.addWidget(self.message_input)


        # Add a send button
        self.send_button = QPushButton("Send")
        self.send_button.setFont(QFont("Helvetica", 12))
        # self.send_button.setIcon(QIcon("path_to_send_icon.png"))  # Optional: Set an icon for the button
        self.send_button.clicked.connect(self.on_send_clicked)
        self.layout.addWidget(self.send_button)

        # Add a listen button
        self.listen_button = QPushButton("Listen")
        self.listen_button.setFont(QFont("Helvetica", 12))
        self.listen_button.clicked.connect(self.startSpeechRecognition)
        self.layout.addWidget(self.listen_button)

        # Add menu bar
        menu_bar = self.menuBar()  # This will create a menu bar

        # Create an "Edit" menu
        edit_menu = menu_bar.addMenu("Edit")

        # Add "New Thread" action
        new_thread_action = QAction("New Thread", self)
        new_thread_action.triggered.connect(self.start_new_conversation) 
        edit_menu.addAction(new_thread_action)

        # Add "New Assistant" action
        new_assistant_action = QAction("Update Assistant Files", self)
        new_assistant_action.triggered.connect(self.update_assistant_files)  
        edit_menu.addAction(new_assistant_action)

        # Set the minimum size for the window
        self.setMinimumSize(QSize(800, 600))

    def apply_style(self):
        style_sheet = """
            QTextEdit {
                border: 1px solid #444;
                border-radius: 5px;
                padding: 5px;
                background-color: #2e2e2e;
                color: #ffffff;
            }
            QPushButton {
                border: 1px solid #444;
                border-radius: 5px;
                padding: 5px;
                background-color: #5e5e5e;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #3e3e3e;
            }
            QLabel {
                color: #ffffff;
            }
        """
        self.setStyleSheet(style_sheet)


        """
        Called when the QThread finishes, regardless of success or failure.
        """
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QCursor
        # Restore normal cursor
        QApplication.restoreOverrideCursor()

    def start_new_conversation(self):
        self.service_node.get_logger().info("New conversation will be started!")
        try:
            # currently not implemented
            pass
        except Exception as e:
            self.service_node.get_logger().error(f"New conversation could not be created: Error {e}!")
        
        self.chat_history.clear()
        self.update_status_display("New conversation started! Assistant ready for your input!")
        self.service_node.get_logger().info("New conversation started!")

    def update_assistant_files(self):
        self.service_node.get_logger().info("Assistant will be updated with new file")

        
        self.chat_history.clear()
        self.update_status_display("New file added to the assistant! Assistant ready for your input!")
        self.service_node.get_logger().info("Assistant updated!")


    def startSpeechRecognition(self):
        self.service_node.get_logger().info(f"start speech recognition")
        self.thread = QThread()
        self.worker = SpeechWorker(self.service_node)
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.run)

        # self.worker.textReceived.connect(self.handleSpeechInput)
        self.worker.errorOccurred.connect(self.handleError)
        self.worker.finished.connect(self.thread.quit)  # Ensure worker emits a finished signal when done
        self.worker.finished.connect(self.worker.deleteLater)  # Cleanup worker after finishing
        self.thread.finished.connect(self.thread.deleteLater)  # Cleanup thread after it finishes

        self.worker.finished.connect(self.handleSpeechInput)

        self.thread.start()

    def handleSpeechInput(self, message):
        # Process the recognized text as input
        self.service_node.get_logger().info(f"Recognized text: {message}")
        self.chat_history.append_question("User: ", message)
        self.start_message_processing_thread(message)
        self.message_input.clear()

    def handleError(self, error_message):
        # Handle errors here
        self.service_node.get_logger().info(f"Error: {error_message}")

    def onProcessedMessage(self, response):
        self.client = OpenAI()
        self.client.api_key = os.environ["OPENAI_API_KEY"]
        
        speech_repsonse = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=response
        )
        # Save the speech response to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmpfile:
            tmpfile.write(speech_repsonse.content)
            tmp_filename = tmpfile.name

        # Load the temporary file into an AudioSegment
        sound = AudioSegment.from_file(tmp_filename)

        # Play the audio
        play(sound)
        os.remove(tmp_filename)
        # Use pyttsx3 to speak out the response
        # self.speech_engine.say(response)
        # self.speech_engine.runAndWait()

    def eventFilter(self, obj, event):
        """Enter sends message, Shift+Enter inserts newline."""
        if obj is self.message_input and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    # Shift+Enter: insert newline (default behavior)
                    return False
                else:
                    # Enter: send message
                    self.on_send_clicked()
                    return True
        return super().eventFilter(obj, event)

    def on_send_clicked(self):
        message = self.message_input.toPlainText()
        self.chat_history.append_question("User: ", message)
        self.start_message_processing_thread(message)
        self.message_input.clear()

    def start_message_processing_thread(self, message):
        self.thread = QThread()
        self.worker = MessageWorker(self.agent, message, self.service_node)
        self.worker.moveToThread(self.thread)

        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # New signal to handle partial chunks
        self.worker.chunk_received.connect(self.handle_partial_chunk)

        # The final result once the worker is done
        self.worker.finished.connect(self.handle_processed_message)

        # Optionally update the GUI with statuses
        self.worker.update_status.connect(self.update_status_display)
        
        # Connect sequence modified signal to parent
        self.worker.sequence_modified.connect(self.on_sequence_modified)

        self.thread.start()

    def handle_partial_chunk(self, chunk_text):
        """
        Slot to receive partial chunks from the worker.
        Append the chunk to a text display or do whatever is needed.
        """
        self.chat_history.append_reply(chunk_text)  # or some other widget


    def handle_processed_message(self, response):
        self.chat_history.append_reply("Agent: ", response)
        # self.onProcessedMessage(response=response)

    def update_status_display(self, status_message):
        self.status_label.setText(f"Status: {status_message}")


    def closeEvent(self, event: QEvent):
        # Custom actions to perform when the window is closing
        self.cleanup()
        # if self.thread is not None and self.thread.isRunning():
        #     self.thread.quit()  # Request the thread to quit
        #     self.thread.wait()  # Wait for the thread to finish
        event.accept()  # Proceed with the window closing

    def on_sequence_modified(self):
        """Emit signal to parent RSAP window to refresh the GUI"""
        self.sequence_modified.emit()

    def cleanup(self):
        # Ask user if task was successfully executed
        task_success = None
        comment = None
        if hasattr(self, 'agent') and self.agent.interaction_log:
            from PyQt6.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QLabel, QTextEdit, QPushButton, QHBoxLayout
            
            # Create custom dialog with comment field
            dialog = QDialog(self)
            dialog.setWindowTitle("Task Completion")
            dialog.setMinimumWidth(400)
            
            layout = QVBoxLayout()
            
            # Question label
            question_label = QLabel("Was the task successfully executed?")
            layout.addWidget(question_label)
            
            # Comment text field
            comment_label = QLabel("Comment (optional):")
            layout.addWidget(comment_label)
            
            comment_field = QTextEdit()
            comment_field.setMaximumHeight(100)
            comment_field.setPlaceholderText("Enter any comments about the task execution...")
            layout.addWidget(comment_field)
            
            # Buttons
            button_layout = QHBoxLayout()
            yes_button = QPushButton("Yes")
            partly_button = QPushButton("Partly")
            no_button = QPushButton("No")
            cancel_button = QPushButton("Cancel")
            
            button_layout.addWidget(yes_button)
            button_layout.addWidget(partly_button)
            button_layout.addWidget(no_button)
            button_layout.addWidget(cancel_button)
            layout.addLayout(button_layout)
            
            dialog.setLayout(layout)
            
            # Connect buttons
            result = {'success': None}
            def on_yes():
                result['success'] = True
                dialog.accept()
            def on_partly():
                result['success'] = "partly"
                dialog.accept()
            def on_no():
                result['success'] = False
                dialog.accept()
            def on_cancel():
                result['success'] = None
                dialog.reject()
            
            yes_button.clicked.connect(on_yes)
            partly_button.clicked.connect(on_partly)
            no_button.clicked.connect(on_no)
            cancel_button.clicked.connect(on_cancel)
            
            # Show dialog
            if dialog.exec() == QDialog.DialogCode.Accepted:
                task_success = result['success']
                comment_text = comment_field.toPlainText().strip()
                if comment_text:
                    comment = comment_text

        # Save interaction log before closing
        if hasattr(self, 'agent'):
            self.agent.save_interaction_log(task_success=task_success, comment=comment)

        self.service_node.get_logger().info("Application closed.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = PmCoPilotProgrammingApp()
    mainWin.show()
    sys.exit(app.exec())
