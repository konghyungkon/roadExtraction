import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QSizePolicy
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from tensorflow.keras.models import load_model
from PIL import Image
import os
from PyQt5 import uic

# --- 환경 변수 설정: OpenCV와 PyQt5의 Qt 플랫폼 충돌 방지 ---
try:
    if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
        del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
except Exception as e:
    print(f"환경 변수 설정 중 오류 발생: {e}")
# ------------------------------------------------------------------


# 이미지 전처리를 위한 상수 정의
IMG_SIZE = 64
MODEL_PATH = './cat_and_dog_binary_classfication_0.8644000291824341.h5'
UI_FILE_PATH = './cat_dog.ui'


# --- 1. 유틸리티 함수: OpenCV 이미지(NumPy)를 QPixmap으로 변환 ---

def convert_cv_to_qt(cv_img):
    """
    OpenCV (BGR, NumPy array) 이미지를 PyQt5에서 표시할 수 있는 QPixmap 객체로 변환합니다.
    """
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB).copy()
    height, width, channel = rgb_image.shape
    bytes_per_line = channel * width

    q_image = QImage(
        rgb_image.data,
        width,
        height,
        bytes_per_line,
        QImage.Format_RGB888
    )

    return QPixmap.fromImage(q_image)


# --- 2. QThread Worker: 비디오 캡처 및 프레임 전송 (I/O 분리) ---

class VideoWorker(QThread):
    """
    별도의 스레드에서 비디오 캡처를 수행하고 프레임을 메인 스레드로 보냅니다.
    """
    frame_ready = pyqtSignal(QPixmap)
    raw_frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, parent=None, cam_id=0):
        super().__init__(parent)
        self.cam_id = cam_id
        self.running = True
        self.capture = None

    def run(self):
        self.capture = cv2.VideoCapture(self.cam_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        if not self.capture.isOpened():
            print("카메라를 열 수 없습니다. ID를 확인하거나 다른 카메라를 시도하세요.")
            self.running = False
            return

        while self.running:
            ret, frame = self.capture.read()

            if ret:
                # 1. GUI 디스플레이용 프레임 전송
                pixmap = convert_cv_to_qt(frame)
                self.frame_ready.emit(pixmap)

                # 2. AI 예측용 원본 배열 전송
                self.raw_frame_ready.emit(frame)

            self.msleep(100)  # 약 30 FPS

        if self.capture:
            self.capture.release()

    def stop(self):
        """스레드를 안전하게 중지합니다."""
        self.running = False
        self.wait()

    # --- 3. 메인 애플리케이션 클래스 ---


# UI 파일을 로드합니다.
try:
    form_window = uic.loadUiType(UI_FILE_PATH)[0]
except FileNotFoundError:
    print(f"오류: UI 파일 '{UI_FILE_PATH}'을 찾을 수 없습니다. 기본 템플릿으로 대체합니다.")


    # UI 파일이 없을 경우 더미 클래스를 사용 (안정성 확보)
    class DummyForm:
        def setupUi(self, widget):
            widget.setWindowTitle("UI 파일 로드 실패")
            layout = QVBoxLayout(widget)
            # video_label 대신 self.label을 사용하도록 더미 클래스에 정의
            widget.label = QLabel("비디오 표시 영역 (self.label)")
            widget.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            widget.label_2 = QLabel("예측 결과 (self.label_2): 로드 실패")
            widget.pushButton = QPushButton("카메라 시작/중지")
            layout.addWidget(widget.label)
            layout.addWidget(widget.pushButton)
            layout.addWidget(widget.label_2)


    form_window = DummyForm


class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.path = None
        self.setupUi(self)
        self.setWindowTitle('실시간 AI 예측 (고양이/강아지)')

        # UI 파일에서 로드되는 위젯 이름을 확인해야 합니다.
        # 기존 코드에서 AttributeError가 발생했으므로,
        # video_label 대신 가장 일반적인 이름인 self.label을 비디오 표시 영역으로 가정합니다.

        # self.label (비디오 표시)이 QLabel이고, sizePolicy가 설정되어 있어야 안정적으로 보입니다.
        if hasattr(self, 'label'):
            self.label.setText("카메라 스트림 시작 대기 중...")
            self.label.setAlignment(Qt.AlignCenter)
        else:
            print("경고: UI에 'label'이라는 이름의 위젯이 없습니다. 비디오 표시가 불가능합니다.")

        self.model = None
        self.load_ai_model()
        self.categories = ['고양이!!', '강아지!!']
        self.video_thread = None

        # pushButton은 카메라 시작/중지 토글로 사용합니다.
        self.pushButton.clicked.connect(self.video_button_slot)
        self.pushButton_file.clicked.connect(self.button_slot)

        # 파일 이미지 예측용 버튼이 있다면 그 버튼에 self.button_slot을 연결해야 합니다.
        # 현재는 pushButton 하나만 있으므로 video_button_slot에 연결합니다.

    def load_ai_model(self):
        """AI 모델 로딩"""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = load_model(MODEL_PATH)
                print(f"AI 모델 로딩 성공: {MODEL_PATH}")
            except Exception as e:
                print(f"AI 모델 로딩 실패: {e}")
                self.label_2.setText("AI 모델 로딩 실패!")
        else:
            print(f"경고: 모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다. 예측 기능은 비활성화됩니다.")
            self.label_2.setText("예측 모델을 찾을 수 없습니다.")

    # --- 4. 비디오 시작/중지 슬롯 (이전 video_button_slot) ---
    def video_button_slot(self):
        """카메라 스트림 시작 또는 중지 버튼 클릭 처리."""
        if self.video_thread and self.video_thread.isRunning():
            # 중지 로직
            self.video_thread.stop()
            self.video_thread = None
            if hasattr(self, 'label'):
                self.label.setText("비디오 스트림 중지됨.")
            self.pushButton.setText("카메라 스트림 시작")
            self.label_2.setText("예측 결과: 비디오 스트림이 중지되었습니다.")
        else:
            # 시작 로직
            self.video_thread = VideoWorker()

            # 1. GUI 디스플레이 연결: self.label로 이름 수정
            self.video_thread.frame_ready.connect(self.update_frame)

            # 2. AI 예측 데이터 연결
            self.video_thread.raw_frame_ready.connect(self.process_prediction)

            self.video_thread.start()
            self.pushButton.setText("카메라 스트림 중지")
            self.label_2.setText("예측 결과: 카메라 연결 중...")

    def update_frame(self, pixmap):
        """Worker 스레드로부터 QPixmap을 받아 QLabel에 표시합니다. (GUI 스레드에서 실행)"""
        # self.video_label 대신 self.label 사용
        if not hasattr(self, 'label'):
            return  # self.label이 없으면 아무것도 할 수 없습니다.

        display_pixmap = pixmap.scaled(
            self.label.size(),  # self.video_label -> self.label 수정
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.label.setPixmap(display_pixmap)  # self.video_label -> self.label 수정

    def process_prediction(self, raw_frame):
        """
        raw NumPy 배열을 받아 AI 예측을 수행합니다. (메인 GUI 스레드 실행)
        """
        if self.model is None:
            return

        try:
            # 1. 비디오 프레임 전처리 (BGR -> RGB -> 리사이즈)
            data = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            data = cv2.resize(data, (IMG_SIZE, IMG_SIZE))

            # 2. 모델 입력 형태로 변환 및 정규화
            data = data / 255.0
            model_input = np.expand_dims(data, axis=0)  # (1, 64, 64, 3)

            # 3. 예측 수행
            predict_value = self.model.predict(model_input, verbose=0)
            probability = predict_value[0][0]
            predicted_index = 1 if probability > 0.5 else 0

            # 4. 결과 해석 및 출력
            if predicted_index == 1:  # 강아지
                prob_percent = int(probability * 100)
                result_text = f'강아지!! (확률: {prob_percent}%)'
            else:  # 고양이
                prob_percent = int((1 - probability) * 100)
                result_text = f'고양이!! (확률: {prob_percent}%)'

            self.label_2.setText(result_text)

        except Exception as e:
            error_text = f"예측 처리 오류: {e}"
            self.label_2.setText(error_text)
            print(error_text)

    # --- 5. 기존 파일 처리 슬롯 (이전 button_slot, 현재는 파일 처리용으로 사용) ---
    def button_slot(self):
        """
        UI 파일 로드 방식에서는 이 함수가 pushButton과 연결되었을 가능성이 낮습니다.
        만약 파일 선택 버튼이 있다면 이 함수를 연결하여 사용합니다.
        현재는 video_button_slot이 pushButton에 연결되어 있어 이 함수는 호출되지 않을 수 있습니다.
        """
        print("파일 선택 기능을 실행합니다.")

        file_dialog = QFileDialog()
        self.path, _ = file_dialog.getOpenFileName(self, '이미지 파일 선택', '', "Image Files (*.png *.jpg *.jpeg)")

        if self.path:
            try:
                # 1. 이미지 로드 및 전처리 (64x64)
                img = Image.open(self.path)
                img = img.convert('RGB')

                # 2. QLabel에 이미지 미리보기 표시: self.label 사용
                if hasattr(self, 'label'):
                    qt_img = QPixmap(self.path).scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.label.setPixmap(qt_img)

                # 3. 예측 로직 실행
                img_resized = img.resize((IMG_SIZE, IMG_SIZE))
                data = np.asarray(img_resized)
                self._run_prediction_logic_for_file(data)

            except Exception as e:
                self.label_2.setText(f'파일 처리 오류 발생: {e}')
                print(f'파일 처리 오류: {e}')
        else:
            self.label_2.setText('예측 결과: 파일 선택이 취소되었습니다.')

    def _run_prediction_logic_for_file(self, data):
        """파일 이미지 전용 예측 로직"""
        if self.model is None:
            return

        # 정규화 및 배치 차원 추가
        data = data / 255.0
        model_input = np.expand_dims(data, axis=0)

        predict_value = self.model.predict(model_input, verbose=0)
        probability = predict_value[0][0]
        predicted_index = 1 if probability > 0.5 else 0

        if predicted_index == 1:
            prob_percent = int(probability * 100)
            result_text = f'강아지!! (확률: {prob_percent}%)'
        else:
            prob_percent = int((1 - probability) * 100)
            result_text = f'고양이!! (확률: {prob_percent}%)'

        self.label_2.setText(result_text)
        print(f"file_predict: '{result_text}' 확률: {probability:.4f}")

    def closeEvent(self, event):
        """창이 닫힐 때 스레드를 안전하게 종료합니다."""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()


# --- 4. 애플리케이션 실행 ---

if __name__ == "__main__":
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()

    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())