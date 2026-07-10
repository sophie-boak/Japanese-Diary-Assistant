import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QStackedWidget,
    QHBoxLayout,
    QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFontDatabase, QFont, QRegion, QPainterPath, QPainter, QPen, QColor, QMovie
from google import genai
from google.cloud import vision
from dotenv import load_dotenv
import cv2
import numpy as np

load_dotenv()
# Parses a .env file and loads its key-value pairs into the applications's
# environment variables. By default, it searches for a file named .env in the current working 
# directory or traverses up the directory tree to find one

client = genai.Client()

# ----------------------------
# Gemini Thread
# ----------------------------
class GeminiThread(QThread):
    feedback_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    def __init__(self, extracted_text):
        super().__init__()
        self.text = extracted_text

    def run(self):
        client = genai.Client()
        try:
            response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=f"""You are a Japanese language teacher.

            The following text is a Japanese diary written by a learner. Your task is to give helpful feedback for learning.
            Please do all the following:
            1. Provide a corrected version of the diary in natural Japanese.
            2. List the mistakes made by the learner.
            3. Explain each mistake clearly in English.
            4. Suggest more natural or native-like expressions where appropriate.

            Constraints:
            - Do NOT change the meaning of the diary
            - Be kind and encouraging in your feedback
            
            Diary entry from learner:
            {self.text}

            Output format:
            [Corrected Diary]
            ...

            [Mistakes and Explanations]
            - Original: ...
            Correction: ...
            Explanation: ...

            [Natural Suggestions]
            - ...

            # Please provide your feedback:"""
            )
            self.feedback_signal.emit(response.text)
        except Exception as e:
            self.error_signal.emit(str(e))

# ----------------------------
# OCR Thread
# ----------------------------
class OCRThread(QThread):
    extracted_text = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    def __init__(self, photo):
        super().__init__()
        self.photo = photo

    def run(self):
        client = vision.ImageAnnotatorClient()
        # need to encode the numpy array into a proper image format then convert bytes
        success, buffer = cv2.imencode('.jpg', self.photo) # wraps the pixel data into a proper jpeg container so that the API knows what its receiving
        # _ = is a throwaway variable name, but it would be a boolean if the encoding worked or not
        # buffer is a numpy array of bytes representing the complete jpeg - still wrapped in a numpy array object
        if not success:
            self.error_signal.emit("Failed to encode image")
            return
        photo_bytes = buffer.tobytes()
        image = vision.Image(content=photo_bytes) # google cloud vision has its own image object - wrapper that tells the API this is the image data you want to analyse
        try: 
            # code that might fail - attempt this and watch for anything going wrong
            response = client.document_text_detection(image=image) # response is an object containing everything it found
            text_found = response.full_text_annotation.text
            if text_found == "":
                self.error_signal.emit("No text was detected")
                return
            else:
                self.extracted_text.emit(text_found)
        # catch anything that is an Exception or inherits from it - which covers almost every possible error
        # "as e" creates a new variable that holds the specific exception instance that was raised
        except Exception as e:
            # if anything goes wrong, do this instead of crashing
            self.error_signal.emit(str(e))
    
# ----------------------------
# Camera Thread (runs in background)
# ----------------------------
class CameraThread(QThread):
    # pyqtSignal(np.ndarray) is better practice as it is more explicit only allowing numpy array and catches bugs early
    frame_ready = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self.running = True # keeps loop alive
        self.cap = None #  webcam connection
    
    # this is required in QThread, everything inside here runs in the background thread
    def run(self):
        # connect to webcam
        self.cap = cv2.VideoCapture(0)
        
        while self.running: # keep grabbing frames until told to stop
            ret, frame = self.cap.read() # ret = did it work? and then frame = image
            if ret: # if successful 
                self.frame_ready.emit(frame) # send frame to UI

    # need a way to exit cleanly
    def stop(self):
        self.running = False 
        # wait until thread finishes safely
        self.wait()
        # only try to release the camera if it actually exists, otherwise you could get errors on calling .release() on None
        if self.cap: 
            # stop using the webcam and free it properly
            self.cap.release()

class CameraOverlay(QWidget):
    def __init__(self, camera_display):
        super().__init__(camera_display)
        self.setGeometry(0, 0, 640, 480)
        # makes the background fully transparent
        self.setAttribute(Qt.WA_TranslucentBackground)

    def paintEvent(self, event):
        # creates a painter attached to this widget
        painter = QPainter(self)
        pen = QPen(QColor("white"), 3)
        painter.setPen(pen)
        # top left corner
        painter.drawLine(20, 20, 20, 50)
        painter.drawLine(20, 20, 50, 20)
        # top right corner
        painter.drawLine(620, 20, 590, 20)
        painter.drawLine(620, 20, 620, 50)
        # bottom left corner
        painter.drawLine(20, 460, 20, 430)
        painter.drawLine(20, 460, 50, 460)
        # bottom right corner
        painter.drawLine(620, 460, 590, 460)
        painter.drawLine(620, 460, 620, 430)
        # cleans up when you are done - always required
        painter.end()

class CameraView(QWidget):
    def __init__(self, main_window):
        super().__init__()
        
        self.main_window = main_window

        # page components
        # title
        self.title = QLabel("写真を撮ろう")
        self.title.setFont(self.main_window.title_font)
        self.title.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        # camera display
        self.camera_display = QLabel("Camera Display")
        self.camera_display.setFont(self.main_window.body_font)
        self.camera_display.setFixedSize(640, 480)
        self.camera_display.setAlignment(Qt.AlignCenter)

        # curved corners of camera display
        path = QPainterPath()
        path.addRoundedRect(0, 0, 640, 480, 20, 20)
        self.camera_display.setMask(QRegion(path.toFillPolygon().toPolygon()))

        # camera overlay
        self.overlay = CameraOverlay(self.camera_display)
        self.overlay.show()

        # capture button
        self.capture_button = QPushButton("パチャっとね～")
        self.capture_button.setFont(self.main_window.button_font)
        self.capture_button.clicked.connect(self.capture_photo)
        self.capture_button.setFixedSize(300, 70)
        self.capture_button.setStyleSheet("""
            QPushButton{
                border-radius: 30px;
                background-color: #c7c98a; 
            }
            QPushButton:hover{
                background-color: #b9bc7a;                               
            }
        """)

        # totoro picture 
        self.totoro_picture = QLabel()
        picture = QPixmap("graphics/totoro-walking/1. big totoro.png")
        picture = picture.scaled(250, 150, Qt.KeepAspectRatio)
        self.totoro_picture.setPixmap(picture)
        self.totoro_picture.setAlignment(Qt.AlignBottom | Qt.AlignRight)

        # title icon
        self.title_icon = QLabel()
        icon = QPixmap("graphics/title-icons/leaf.png")
        icon = icon.scaled(45, 45, Qt.KeepAspectRatio)
        self.title_icon.setPixmap(icon)

        # camera view page layout 
        layout = QVBoxLayout()

        # title layout
        title_layout = QHBoxLayout()
        title_layout.addWidget(self.title_icon, alignment=Qt.AlignVCenter)
        title_layout.addWidget(self.title, alignment=Qt.AlignVCenter)
        title_layout.setAlignment(Qt.AlignHCenter)
        layout.addLayout(title_layout) 

        # rest of the layout
        layout.addWidget(self.camera_display, alignment=Qt.AlignHCenter)
        layout.addWidget(self.capture_button, alignment=Qt.AlignHCenter)
        layout.addWidget(self.totoro_picture)

        self.setLayout(layout)

    def capture_photo(self):
        # captures the current frame, pauses the livestream and moves onto next page
        # guard clause: exits the method immediately if the conditions aren't safe to proceed
        if self.current_frame is None:
            return
        # save the current frame to the main_window global variable
        self.main_window.captured_frame = self.current_frame
        # move to next page of photo preview
        self.main_window.upload_page.stack.setCurrentIndex(1)

    def update_frame(self, frame):
        # bridges between raw camera frame from OpenCV to Qt UI
        # self.frame_ready.emit(frame): that frame is the arugment being sent through the signal
        # remember lastest frame
        self.current_frame = frame
        # OpenCV uses BGR, Qt uses RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # get image dimensions
        h, w, ch = rgb_frame.shape
        # how many bytes are in one row of pixels
        bytes_per_line = ch * w
        # convert numpy to qimage
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # rgb.data is the raw pixel buffer
        # w, h dimensions
        # Format_RGB888 is 3 channel RGB image
        # convert Qimage to QPixmap
        # QImage: low lebel image data and QPixmap is optimised for displaying on screen
        pixmap = QPixmap.fromImage(qt_image)

        # display it in the UI
        # puts the image into QLabel, resizes, keeps aspect ratio (not stretching) and smooths quality
        self.camera_display.setPixmap(
            pixmap.scaled(
                self.camera_display.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )
    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()

    def showEvent(self, event):
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.start()
        event.accept()
    
    def hideEvent(self, event):
        self.camera_thread.stop()
        event.accept()

class PhotoPreview(QWidget):
    def __init__(self, main_window):
        super().__init__()
        
        self.main_window = main_window
        
        # page components
        # title
        self.title = QLabel("写真をチェックしよう")
        self.title.setFont(self.main_window.title_font)
        self.title.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        # captured photo preview
        self.photo_preview_display = QLabel()
        self.photo_preview_display.setFixedSize(640, 480)

        # curved corners of photo preview
        path = QPainterPath()
        path.addRoundedRect(0, 0, 640, 480, 20, 20)
        self.photo_preview_display.setMask(QRegion(path.toFillPolygon().toPolygon()))

        # camera overlay
        self.overlay = CameraOverlay(self.photo_preview_display)
        self.overlay.show()

        # retake photo button
        self.retake_button = QPushButton("もういっかい～")
        self.retake_button.setFont(self.main_window.button_font)
        self.retake_button.clicked.connect(self.retake_photo)
        self.retake_button.setFixedSize(300, 70)
        self.retake_button.setStyleSheet("""
            QPushButton{
                border-radius: 30px;
                background-color: #c7c98a; 
            }
            QPushButton:hover{
                background-color: #b9bc7a;                               
            }
        """)

        # extract button
        self.extract_button = QPushButton("文字を読んでね～")
        self.extract_button.setFont(self.main_window.button_font)
        self.extract_button.clicked.connect(self.extract_text)
        self.extract_button.setFixedSize(300, 70)
        self.extract_button.setStyleSheet("""
            QPushButton{
                border-radius: 30px;
                background-color: #c7c98a; 
            }
            QPushButton:hover{
                background-color: #b9bc7a;                               
            }
        """)

        # totoro picture 
        self.totoro_picture = QLabel()
        picture = QPixmap("graphics/totoro-walking/1. big totoro.png")
        picture = picture.scaled(250, 150, Qt.KeepAspectRatio)
        self.totoro_picture.setPixmap(picture)
        self.totoro_picture.setAlignment(Qt.AlignBottom | Qt.AlignRight)

        # title icon
        self.title_icon = QLabel()
        icon = QPixmap("graphics/title-icons/leaf.png")
        icon = icon.scaled(45, 45, Qt.KeepAspectRatio)
        self.title_icon.setPixmap(icon)

        # photo preview layout
        layout = QVBoxLayout()

        # title layout
        title_layout = QHBoxLayout()
        title_layout.addWidget(self.title_icon, alignment=Qt.AlignVCenter)
        title_layout.addWidget(self.title, alignment=Qt.AlignVCenter)
        title_layout.setAlignment(Qt.AlignHCenter)
        layout.addLayout(title_layout) 
 
        layout.addWidget(self.photo_preview_display, alignment=Qt.AlignHCenter)
        
        # button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.retake_button)
        button_layout.addWidget(self.extract_button)
        button_layout.setAlignment(Qt.AlignHCenter)
        layout.addLayout(button_layout)
        layout.addWidget(self.totoro_picture)

        self.setLayout(layout)
    
    def showEvent(self, event):
        if self.main_window.captured_frame is None:
            event.accept()
            return
        self.photo_preview = self.main_window.captured_frame
        rgb_frame = cv2.cvtColor(self.photo_preview, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # display it in the UI
        # puts the image into QLabel, resizes, keeps aspect ratio (not stretching) and smooths quality
        self.photo_preview_display.setPixmap(
            pixmap.scaled(
                self.photo_preview_display.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )
        # tells qt my side is done, you can do your part now
        event.accept()
    
    def retake_photo(self):
        self.main_window.upload_page.stack.setCurrentIndex(0)

    def extract_text(self):
        # create an instance of OCRThread passing the captured frame
        self.ocr_thread = OCRThread(self.main_window.captured_frame)
        # connect its signals to methods that handle the results
        self.ocr_thread.extracted_text.connect(self.on_text_extracted)
        self.ocr_thread.error_signal.connect(self.on_error)
        self.ocr_thread.start() # never call run() directly on a thread as it runs on the main UI thread which would freeze your app while waiting for the API
        # navigate to the loading page
        self.main_window.stack.setCurrentIndex(2)

    def on_text_extracted(self, text):
        # runs when the OCR succeeds
        # update the extracted_text variable in the main_window
        self.main_window.extracted_text = text
        # navigate to the edit page
        self.main_window.stack.setCurrentIndex(1)

    def on_error(self, error_message):
        # runs when something goes wrong
        # update the error message in the main_window
        self.main_window.error_message = error_message
        # navigate to the error page
        self.main_window.stack.setCurrentIndex(4)

# ----------------------------
# UPLOAD PAGE 
# two states - camera view and photo preview
# ----------------------------
class UploadPage(QWidget):
    # when this page is created, run this setup code
    # you need to pass the main_window as it owns the qstackedwidget, shared data and all pages
    def __init__(self, main_window):
        # run the setup code of the parent class first, initialise the QWidget part of this page
        super().__init__()

        # store the main window inside this page so all methods in this class can use it later
        self.main_window = main_window

        # create pages within upload page: camera_view and photo_preview
        self.stack = QStackedWidget()

        # create pages
        self.camera_view = CameraView(self.main_window)
        self.photo_preview = PhotoPreview(self.main_window)

        # add pages to the stack
        self.stack.addWidget(self.camera_view)
        self.stack.addWidget(self.photo_preview)
        
        # set the initial page to camera_view
        self.stack.setCurrentIndex(0)

        # add to layout 
        layout = QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)

# ----------------------------
# EDIT PAGE
# ----------------------------
class EditPage(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        layout = QVBoxLayout()

        # title
        self.title = QLabel("文章を直そう")
        self.title.setFont(self.main_window.title_font)
        self.title.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        # title icon
        self.title_icon = QLabel()
        icon = QPixmap("graphics/title-icons/acorn.png")
        icon = icon.scaled(45, 45, Qt.KeepAspectRatio)
        self.title_icon.setPixmap(icon)
        
        # next button
        next_button = QPushButton("AIにお願いだよ～")
        next_button.setFont(self.main_window.button_font)
        next_button.setFixedSize(300, 70)
        next_button.setStyleSheet("""
            QPushButton{
                border-radius: 30px;
                background-color: #c7c98a; 
            }
            QPushButton:hover{
                background-color: #b9bc7a;                               
            }
        """)

        # back button        
        back_button = QPushButton("もどるよ～")
        back_button.setFont(self.main_window.button_font)
        back_button.setFixedSize(300, 70)
        back_button.setStyleSheet("""
            QPushButton{
                border-radius: 30px;
                background-color: #c7c98a; 
            }
            QPushButton:hover{
                background-color: #b9bc7a;                               
            }
        """)

        # edit page overlay
        edit_page_overlay = QLabel()
        overlay_picture = QPixmap("overlays/edit-page-overlay.png")
        overlay_picture = overlay_picture.scaled(1000, 625, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        edit_page_overlay.setPixmap(overlay_picture)
        edit_page_overlay.setFixedSize(1000, 625)

        # text edit
        self.text_edit = QTextEdit(edit_page_overlay)
        self.text_edit.setFont(self.main_window.body_font)
        self.text_edit.setPlaceholderText("Extracted text will appear here...")
        self.text_edit.setMinimumHeight(300)
        self.text_edit.setStyleSheet("background: transparent; border: none;")
        self.text_edit.setGeometry(150, 130, 700, 330)
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # connect functionality of buttons
        next_button.clicked.connect(self.feedback_ai)
        back_button.clicked.connect(self.back_page)

        # totoro picture
        self.totoro_picture = QLabel()
        picture = QPixmap("graphics/totoro-walking/2. medium totoro.png")
        picture = picture.scaled(250, 150, Qt.KeepAspectRatio)
        self.totoro_picture.setPixmap(picture)
        self.totoro_picture.setAlignment(Qt.AlignBottom | Qt.AlignRight)
    
        # layout      
        # title layout
        title_layout = QHBoxLayout()
        title_layout.addWidget(self.title_icon, alignment=Qt.AlignVCenter)
        title_layout.addWidget(self.title, alignment=Qt.AlignVCenter)
        title_layout.setAlignment(Qt.AlignHCenter)
        layout.addLayout(title_layout) 

        layout.addWidget(edit_page_overlay, alignment=Qt.AlignHCenter)

        # button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(next_button)
        button_layout.addWidget(back_button)
        button_layout.setAlignment(Qt.AlignHCenter)
        layout.addLayout(button_layout)
        layout.addWidget(self.totoro_picture)

        self.setLayout(layout)

    def showEvent(self, event):
        if self.main_window.extracted_text == "":
            self.text_edit.setPlainText("No text is available")
        else:
            self.text_edit.setPlainText(self.main_window.extracted_text)
        event.accept()

    def feedback_ai(self):
        # update potentially correct extracted text
        self.main_window.extracted_text = self.text_edit.toPlainText()
        # create a gemini thread
        self.gemini_thread = GeminiThread(self.main_window.extracted_text)
        # connect its signals to handler methods
        self.gemini_thread.feedback_signal.connect(self.on_feedback_ai)
        self.gemini_thread.error_signal.connect(self.on_error)
        # start the thread
        self.gemini_thread.start()
        # navigate to the loading page 2
        self.main_window.stack.setCurrentIndex(2)

    def on_feedback_ai(self, feedback):
        # update the ai feedback variable in the main_window
        self.main_window.ai_feedback = feedback
        # navigate to the feedback page
        self.main_window.stack.setCurrentIndex(3)

    def on_error(self, error_message):
        # update the error message in the main_window
        self.main_window.error_message = error_message
        # navigate to the error page
        self.main_window.stack.setCurrentIndex(4)

    def back_page(self):
        # go to the upload page index 0
        self.main_window.stack.setCurrentIndex(0)
        # but also make sure upload page inner stack is set to 0 too to retake the photo
        self.main_window.upload_page.stack.setCurrentIndex(0)

# ----------------------------
# FEEDBACK PAGE
# ----------------------------
class FeedbackPage(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        layout = QVBoxLayout()

        # title
        title = QLabel("AIのアドバイス")
        title.setFont(self.main_window.title_font)
        title.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        # title icon
        self.title_icon = QLabel()
        icon = QPixmap("graphics/title-icons/seedling.png")
        icon = icon.scaled(45, 45, Qt.KeepAspectRatio)
        self.title_icon.setPixmap(icon)

        # button
        button = QPushButton("もどるよ～")
        button.setFont(self.main_window.button_font)
        button.setFixedSize(300, 70)
        button.setStyleSheet("""
            QPushButton{
                border-radius: 30px;
                background-color: #c7c98a; 
            }
            QPushButton:hover{
                background-color: #b9bc7a;                               
            }
        """)

        # AI feedback overlay
        ai_feedback_overlay = QLabel()
        overlay_picture = QPixmap("overlays/ai-feedback-overlay.png")
        overlay_picture = overlay_picture.scaled(1000, 625, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ai_feedback_overlay.setPixmap(overlay_picture)

        # AI feedback display
        self.ai_output = QTextEdit(ai_feedback_overlay)
        self.ai_output.setReadOnly(True)
        self.ai_output.setFont(self.main_window.body_font)
        self.ai_output.setPlaceholderText("AI feedback will appear here...")
        self.ai_output.setMinimumHeight(400)
        self.ai_output.setStyleSheet("background: transparent; border: none;")
        self.ai_output.setGeometry(130, 80, 750, 300)
        self.ai_output.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ai_output.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # totoro picture 
        self.totoro_picture = QLabel()
        picture = QPixmap("graphics/totoro-walking/3. little totoro.png")
        picture = picture.scaled(250, 150, Qt.KeepAspectRatio)
        self.totoro_picture.setPixmap(picture)
        self.totoro_picture.setAlignment(Qt.AlignBottom | Qt.AlignRight)

        button.clicked.connect(self.back_page)
    
        # layout
        # title layout
        title_layout = QHBoxLayout()
        title_layout.addWidget(self.title_icon, alignment=Qt.AlignVCenter)
        title_layout.addWidget(title, alignment=Qt.AlignVCenter)
        title_layout.setAlignment(Qt.AlignHCenter)
        layout.addLayout(title_layout) 
        
        layout.addWidget(ai_feedback_overlay, alignment=Qt.AlignHCenter)
        layout.addWidget(button, alignment=Qt.AlignHCenter)
        layout.addWidget(self.totoro_picture)

        self.setLayout(layout)

    def showEvent(self, event):
        if self.main_window.ai_feedback == "":
            self.ai_output.setPlainText("No text is detected")
        else:
            self.ai_output.setPlainText(self.main_window.ai_feedback)
        event.accept()

    def back_page(self):
        # go back to the upload page
        self.main_window.stack.setCurrentIndex(0)
        # ensure the upload page inner stack is also set to the initial camera 
        self.main_window.upload_page.stack.setCurrentIndex(0)

# ----------------------------
# LOADING PAGE
# ----------------------------
class LoadingPage(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        layout = QVBoxLayout()

        # component
        # title
        title = QLabel("読み込み中...")
        title.setFont(self.main_window.title_font)
        title.setAlignment(Qt.AlignCenter)
        
        # create animation label
        animation_label = QLabel()
        animation_label.setAlignment(Qt.AlignCenter)

        # load the gif 
        self.movie = QMovie("graphics/spirited-away-soot.gif")
        animation_label.setMovie(self.movie)
       
        layout.setSpacing(0)
        layout.addStretch() 
        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(animation_label)
        layout.addStretch()

        self.setLayout(layout)

    def showEvent(self, event):
        self.movie.start()
        event.accept()

    def hideEvent(self, event):
        self.movie.stop()
        event.accept()
# ----------------------------
# ERROR PAGE
# ----------------------------
class ErrorPage(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        layout = QVBoxLayout()
        # title
        title = QLabel("あれれ？ 迷子になっちゃった")
        title.setFont(self.main_window.title_font)
        title.setContentsMargins(0, 0, 0, 0)
        # subtitle
        subtitle = QLabel("ドングリを追いかけすぎたかも～")
        subtitle.setFont(self.main_window.button_font)
        subtitle.setContentsMargins(0, 0, 0, 0)
        # return button
        return_button = QPushButton("お家にかえる")
        return_button.setFont(self.main_window.button_font)
        return_button.setFixedSize(300, 70)
        return_button.setStyleSheet("""
            QPushButton{
                border-radius: 30px;
                background-color: #c7c98a; 
            }
            QPushButton:hover{
                background-color: #b9bc7a;                               
            }
        """)
        # add functionality to button
        return_button.clicked.connect(self.return_page)
        # picture
        self.totoro_picture = QLabel()
        picture = QPixmap("graphics/totoro-leaf.png")
        picture = picture.scaled(400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.totoro_picture.setPixmap(picture)
        # error message
        self.error_message = QLabel("エラー：")
        self.error_message.setFont(self.main_window.body_font)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(15)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.addWidget(title, alignment=Qt.AlignHCenter)
        title_layout.addWidget(subtitle, alignment=Qt.AlignHCenter)
        layout.setSpacing(100)
        layout.addStretch()
        layout.addLayout(title_layout)
        layout.addWidget(self.error_message, alignment=Qt.AlignHCenter)
        layout.addWidget(self.totoro_picture, alignment=Qt.AlignHCenter)
        layout.addWidget(return_button, alignment=Qt.AlignHCenter)
        layout.addSpacing(70)

        self.setLayout(layout)

    def return_page(self):
        # go back to the upload page
        self.main_window.stack.setCurrentIndex(0)
        # ensure the upload page inner stack is set
        self.main_window.upload_page.stack.setCurrentIndex(0)

    def showEvent(self, event):
        self.error_message.setText(f"エラー: {self.main_window.error_message}")
        event.accept()
    
# ----------------------------
# MAIN WINDOW
# ----------------------------
# decided to keep it as a QWidget as I want to not have the pre-built main app window that QMainWindow comes with
class MainWindow(QWidget):
    def __init__(self):
        super().__init__() # this calls the parent class's __init__ which is QWidget

        self.setWindowTitle("Japanese Diary Assistant")
        self.setWindowIcon(QIcon("assets/leaf.ico"))
        self.setFixedSize(1440, 900)

        # styling/themes
        # font
        font_id = QFontDatabase.addApplicationFont(
            "C:/Users/m3sop/Documents/Engineering Projects/Japanese Diary Recognition/assets/fonts/KiwiMaru-Medium.ttf"
        )

        if font_id == -1:
            raise RuntimeError("Failed to load Kiwi Maru font")
        families = QFontDatabase.applicationFontFamilies(font_id)
        if not families:
            raise RuntimeError("No font families " \
            "found")
        
        base_font = QFont(families[0])
        base_font.setStyleStrategy(QFont.PreferAntialias)
        base_font.setLetterSpacing(QFont.PercentageSpacing, 115)
        
        self.title_font = QFont(base_font)
        self.title_font.setPointSize(20)
        self.body_font = QFont(base_font)
        self.body_font.setPointSize(12)
        self.button_font = QFont(base_font)
        self.button_font.setPointSize(12)

        # background colour
        self.setStyleSheet(f"""
            QWidget {{
                background-color: #FDFDFD;
                color: #524645;
            }}
        """)

        # Shared data between pages
        self.captured_frame = None
        self.extracted_text = ""
        self.ai_feedback = ""
        self.error_message = ""

        # create a stacked widget
        self.stack = QStackedWidget()

        # create pages
        self.upload_page = UploadPage(self)
        self.edit_page = EditPage(self)
        self.feedback_page = FeedbackPage(self)
        self.loading_page = LoadingPage(self)
        self.error_page = ErrorPage(self)

        # add pages to stack (order matters!)
        self.stack.addWidget(self.upload_page)   # index 0
        self.stack.addWidget(self.edit_page)     # index 1
        self.stack.addWidget(self.loading_page)  # index 2
        self.stack.addWidget(self.feedback_page) # index 3
        self.stack.addWidget(self.error_page)    # index 4

        self.stack.setCurrentIndex(0) # set to 0 for upload page

        # layout for main window
        layout = QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)

# ----------------------------
# only run the code below if this file is being run directly, NOT if its imported somewhere else - prevent things from running unintentionally
if __name__ == "__main__":
    # just pass an empty list for the sys.argv if I know I won't use command line argument to control Qt.
    app = QApplication([])

    window = MainWindow()
    window.show() # windows are hidden by default

    sys.exit(app.exec())
    # app.exec() start PyQt application's event loop
    # exits your entire program, the OS doesnt get proper exit status and may not exit cleanly.
