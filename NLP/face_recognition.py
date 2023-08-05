#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Modify Time 2023/8/5 15:15
@Author maoalong  
@Version 1.0
@Desciption None
"""
import cv2
import numpy as np

IMAGE_DIR = "/data/along/dataset/FaceDataset"
# 图片向量缓存
image_map = {}


def load_face_model():
    # 加载人脸检测模型
    prototxt_path = "NLP/opencv_face_detector.pbtxt"
    model_path = "NLP/opencv_face_detector_uint8.pb"
    face_net = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)

    # 加载人脸识别模型
    face_rec_model = cv2.dnn.readNetFromTorch("NLP/nn4.small2.v1.t7")

    return face_net, face_rec_model


def detect_faces(image, face_net):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            faces.append(face)
    return faces


def get_face_embeddings(face, face_rec_model):
    face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    face_rec_model.setInput(face_blob)
    return face_rec_model.forward()


def calculate_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2.T)


def getEmbeddingFromCache(image_path, face_net, face_rec_model):
    if image_path not in image_map:
        imageCV2 = cv2.imread("{}{}".format(IMAGE_DIR, image_path))
        faces = detect_faces(imageCV2, face_net)
        if len(faces) != 1:
            return None
        embedding = get_face_embeddings(faces[0], face_rec_model)
        image_map[image_path] = embedding
    else:
        embedding = image_map[image_path]
    return embedding


def compare_faces(source, target, face_net, face_rec_model):
    if source is None or source == "" or target is None or target == "":
        return False, None

    # add cache
    source_embedding = getEmbeddingFromCache(source, face_net, face_rec_model)
    target_embedding = getEmbeddingFromCache(target, face_net, face_rec_model)

    if source_embedding is None or target_embedding is None:
        return False, None

    similarity = calculate_similarity(source_embedding, target_embedding)
    threshold = 0.5  # 调整此阈值以控制判断为同一个人的敏感度

    if similarity > threshold:
        return True, similarity
    else:
        return False, similarity


if __name__ == "__main__":
    image1 = "/lfw/Aaron_Peirsol/Aaron_Peirsol_0001.jpg"
    image2 = "/lfw/Aaron_Peirsol/Aaron_Peirsol_0002.jpg"

    # load model
    face_net, face_rec_model = load_face_model()

    is_same_person, similarity_score = compare_faces(image1, image2, face_net, face_rec_model)

    if is_same_person:
        print("是同一个人，相似度：", similarity_score)
    else:
        print("不是同一个人，相似度：", similarity_score)
