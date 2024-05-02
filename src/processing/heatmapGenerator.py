import logging

from torchvision.transforms.functional import to_pil_image
import matplotlib as plt
import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


class HeatmapGenerator:
    def __init__(self, heatmap_dir, model_path):
        # Initialiser le détecteur de visages avec MediaPipe
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Répertoire pour les heatmaps
        self.heatmap_dir = heatmap_dir
        if not os.path.exists(heatmap_dir):
            os.makedirs(heatmap_dir)

    def generate_heatmaps_for_directory(self,base_dir):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        for dirpath, dirnames, filenames in os.walk(base_dir):
            logging.info(f'Processing {filenames}')
            for filename in filenames:
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(dirpath, filename)
                    logging.info(f'Processing {image_path}')
                    self.process_image(image_path, filename)

    def process_image(self, image_path, original_filename):
        # Charger et prétraiter l'image
        image = mp.Image.create_from_file(image_path)
        detection_result = self.detector.detect(image)
        # Détecter les visages et annoter l'image
        annotated_image = self.draw_landmarks_on_image(image.numpy_view(),detection_result)

        # Enregistrer l'image annotée
        annotated_filename = os.path.join(self.heatmap_dir, os.path.splitext(original_filename)[0] + '_h.png')
        cv2.imwrite(annotated_filename, annotated_image)
        print("Image saved to", annotated_filename)

        return annotated_image

    def draw_landmarks_on_image(self, rgb_image,detection_result):
        face_landmarks_list = detection_result.face_landmarks
        height, width , _= rgb_image.shape
        #annotated_image = np.copy(rgb_image)
        annotated_image = np.zeros((height, width, 3), dtype=np.uint8)

        NOSE =[1,2, 4, 5, 6,19, 275, 278, 294, 168, 45, 48, 440, 64, 195, 197, 326, 327, 328, 331, 332, 344, 220, 94, 97, 98]
        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]
            selected_landmarks = [face_landmarks[i] for i in NOSE]
            for landmark in selected_landmarks:
                x, y = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(annotated_image, (x, y), 3, (255,255,255), thickness=-1)  # 255 for white points
            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in
                face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

        annotated_image = cv2.GaussianBlur(annotated_image, (0, 0), 1)
        return annotated_image


