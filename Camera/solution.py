from typing import Dict, Union, List
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import pandas as pd

from keras.models import load_model


class Solution:
    drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    def find(self, frame: np.ndarray) -> np.ndarray:
        pass

    def draw(self, landmark, frame) -> None:
        pass

    def get_vectors(self, landmarks) -> Dict[tuple, np.ndarray]:
        pass


class Pose(Solution):
    pose: mp.solutions.pose.Pose
    pose_connections = mp.solutions.pose.POSE_CONNECTIONS
    landmark_indices = mp.solutions.pose.PoseLandmark

    def __init__(self):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)

    def find(self, frame: np.ndarray, norm=False) -> np.ndarray:
        body_coords = np.zeros((33, 4))
        height, width, _ = frame.shape

        result = self.pose.process(frame)
        pose_landmarks = result.pose_landmarks

        if not pose_landmarks:
            return body_coords

        for ind in self.landmark_indices:
            try:
                landmark = pose_landmarks.landmark[ind]
                body_coords[ind] = np.array([landmark.x, landmark.y, landmark.z, landmark.visibility])
            except TypeError:
                pass
        self.draw(pose_landmarks, frame)

        if norm:
            ymax = body_coords[mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER][1]
            ymin = body_coords[mp.solutions.pose.PoseLandmark.LEFT_HEEL][1]
            body_coords[:, 0] = (body_coords[:, 0] - body_coords[mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER, 0]) / (
                    ymax - ymin)
            body_coords[:, 2] = (body_coords[:, 2] - body_coords[mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER, 2]) / (
                    ymax - ymin)
            y = (body_coords[:, 1] - ymin) / (ymax - ymin)
            body_coords[:, 1] = y

        return body_coords

    def draw(self, landmark, frame) -> None:
        self.drawing.draw_landmarks(
            frame,
            landmark,
            self.pose_connections,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )


class NeuronProcess(Pose):

    def __init__(self, model_speed, model_predict, model_process="models/test_CNN.h5"):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)
        self.nn_process = load_model(model_process)
        self.nn_predict = load_model(model_predict)
        self.nn_speed = load_model(model_speed)

    def nn_process(self, body):
        result = self.nn_process.predict(np.array([body]))
        return result[0]

    def predict_pose(self, body):
        result = self.nn_predict.predict(np.array([body]))
        return result[0]

    def predict_speed(self, body):
        result = self.nn_speed.predict(np.array([body]))
        return result[0]

    def get_final_speed_by_frame(self, frame):
        result = self.predict_speed(self.predict_pose(self.nn_process(self.find(frame))))
        return result

    def get_final_speed_(self, body):
        result = self.predict_speed(self.predict_pose(self.nn_process(body)))
        return result


class Camera:
    capture = None
    solution: Pose = None
    buffer: List[np.ndarray] = None

    def __init__(self, source: Union[str, int]):
        self.capture = cv2.VideoCapture(source)
        self.solution = Pose()
        self.buffer = []

    def process(self) -> None:

        sz = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mpeg')
        name = datetime.now().time()
        outvideo1 = cv2.VideoWriter()
        outvideo1.open(f'videos/road_2/{name}_simple.mp4', fourcc, 30, sz, True)

        outvideo2 = cv2.VideoWriter()
        outvideo2.open(f'videos/road_2/{name}_with_points.mp4', fourcc, 30, sz, True)

        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                print("Failed to read frame")
                break
            outvideo1.write(frame)

            points = self.solution.find(frame)
            self.buffer.append(points)
            outvideo2.write(frame)
            cv2.imshow('MediaPipe Pose', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # if press SPACE bar
                outvideo1.release()
                outvideo2.release()
                self.capture.release()
                cv2.destroyAllWindows()
                break

    def save_file(self, filename: str) -> None:
        np.save(filename, np.array(self.buffer))

    def load_points_from_videos(self) -> None:
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                print("Failed to read frame")
                break

            points = self.solution.find(frame)
            self.buffer.append(points)

    def get_points(self) -> np.array:
        return np.array(self.buffer)
