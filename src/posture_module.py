import math


def angle_between_points(a, b, c):
    """
    Returns angle at point b (in degrees) for triangle a-b-c.
    a, b, c are (x, y) tuples.
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot_prod = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2) + 1e-6

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    cos_angle = dot_prod / (mag_ba * mag_bc)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    angle = math.degrees(math.acos(cos_angle))
    return angle


def get_posture_status(pose_landmarks, w, h, slouch_thresh=150):
    """
    Very simple rule:
    - Use left hip, left shoulder, and a virtual vertical reference point.
    - If back angle < slouch_thresh => Bad posture (leaning forward).
    
    Returns:
        status (str): "Good Posture", "Bad Posture" or "No Person"
        angle_back (float or None)
    """
    if not pose_landmarks:
        return "No Person", None

    lm = pose_landmarks[0].landmark

    try:
        left_shoulder = (lm[11].x * w, lm[11].y * h)  # 11: left shoulder
        left_hip = (lm[23].x * w, lm[23].y * h)       # 23: left hip
    except IndexError:
        return "No Person", None

    # virtual vertical reference point above hip
    virtual_point = (left_hip[0], left_hip[1] - 0.25 * h)

    angle_back = angle_between_points(virtual_point, left_hip, left_shoulder)

    if angle_back < slouch_thresh:
        return "Bad Posture", angle_back
    else:
        return "Good Posture", angle_back
