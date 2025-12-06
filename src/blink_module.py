import numpy as np

# MediaPipe FaceMesh eye landmark indices (approx)
# You can tweak later if needed
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_indices, image_width, image_height):
    """
    Calculate Eye Aspect Ratio (EAR) for given eye.
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * image_width, lm.y * image_height))

    if len(pts) != 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = pts

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4) + 1e-6

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def get_blink_status(face_landmarks, w, h, ear_thresh=0.22):
    """
    Returns:
        is_blink (bool): True if blink detected in this frame
        ear (float): average eye aspect ratio
    """
    if not face_landmarks:
        return False, 0.0

    # Take first detected face
    landmarks = face_landmarks[0].landmark

    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    ear = (left_ear + right_ear) / 2.0

    is_blink = ear < ear_thresh
    return is_blink, ear
