from PIL import Image
import cv2
import os
import numpy as np
import math

def read_csv(file_path):
    cnt = 0
    with open(file_path) as f:
        header = ''
        data = []
        for line in f:
            cnt += 1
            if cnt == 1:
                header = line.strip()
                continue
            str_list = line.strip().split(',')
            if len(str_list) != 9:
                print('len', len(str_list))
            str_list[5] = str_list[5].strip().split(';')
            str_list[5] = list(map(float, str_list[5]))
            str_list[5] = list(map(int, str_list[5]))
            str_list[5] = [(str_list[5][2 * i], str_list[5][2 * i + 1]) for i in range(68)]

            data.append(str_list)
        return data

def draw_landmarks(image_path, landmark):
    img = cv2.imread(image_path)
    for idx in range(68):
        cv2.circle(img, landmark[idx], 3, color = (0, 0, 0))
    # cv2.imwrite('/data/ljj/project/FRAN/imgs/with_landmark.jpg', img)
    return img

def get_face(image_path, face_pos):
    img = cv2.imread(image_path)
    # print('face_pos:', face_pos[0], face_pos[1], face_pos[2], face_pos[3])
    img = img[ int(face_pos[1]):int(face_pos[1]) + int(face_pos[2]) ][ int(face_pos[0]):int(face_pos[0]) + int(face_pos[3]) ][ : ]
    cv2.imwrite('/data/ljj/project/FRAN/imgs/face.jpg', img)
    return img

def get_mark(image_path, landmark):
    img = cv2.imread(image_path) 
    mask = np.zeros(img.shape[:2])
    p = [[[int(e[0]), int(e[1])]] for e in landmark]
    hull = cv2.convexHull(np.array(p), False)
    cv2.fillPoly(mask, [hull], 255)
    # cv2.imwrite('/data/ljj/project/FRAN/imgs/mask.jpg', mask)
    # print(mask)
    return mask

def single_face_alignment_with_crop(image_path, face_pos, landmarks):
    face = get_face(image_path, face_pos)
    eye_center = ((landmarks[36][0] - face_pos[0] + landmarks[45][0] - face_pos[0]) * 1. / 2,  # 计算两眼的中心坐标
                  (landmarks[36][1] - face_pos[1] + landmarks[45][1] - face_pos[1]) * 1. / 2)
    dx = (landmarks[45][0] - landmarks[36][0])  # note: right - right
    dy = (landmarks[45][1] - landmarks[36][1])

    angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度
    RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵
    align_face = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))  # 进行放射变换，即旋转
    # cv2.imwrite('/data/ljj/project/FRAN/imgs/align_face.jpg', align_face)
    return align_face

def single_face_alignment(image_path, landmarks):
    face = cv2.imread(image_path)
    eye_center = ((landmarks[36][0] + landmarks[45][0]) * 1. / 2,  # 计算两眼的中心坐标
                  (landmarks[36][1] + landmarks[45][1]) * 1. / 2)
    dx = (landmarks[45][0] - landmarks[36][0])  # note: right - right
    dy = (landmarks[45][1] - landmarks[36][1])

    angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度
    RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵
    align_face = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))  # 进行放射变换，即旋转
    cv2.imwrite('/data/ljj/project/FRAN/imgs/align_face.jpg', align_face)
    return align_face


if __name__ == '__main__':
    file_path = '/data/ljj/FER/AffectNet/Manually_Annotated_file_lists/validation.csv'
    data = read_csv(file_path)
    # print(data[0])

    IMAGE_BASE_PATH = '/data/ljj/FER/AffectNet/Manually_Annotated_compressed/Manually_Annotated_Images'
    idx = 232 
    image_path = os.path.join(IMAGE_BASE_PATH, data[idx][0])
    # print('idx', idx)
    # print('image_path', image_path)

    landmark = data[idx][5]
    face_pos = [data[idx][1], data[idx][2], data[idx][3], data[idx][4]]
    draw_landmarks(image_path, landmark)
    get_mark(image_path, landmark)
    single_face_alignment(image_path, landmark)
    get_face(image_path, face_pos)
