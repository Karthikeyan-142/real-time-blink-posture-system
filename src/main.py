import cv2
import mediapipe as mp
import time

from blink_module import get_blink_status, LEFT_EYE, RIGHT_EYE
from posture_module import get_posture_status
from logger import save_session_stats

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose


# FULL face oval outline landmark indices
FULL_FACE_OUTLINE = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]


def main():

    # ================= Camera Setup ================
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # reduce camera resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "Blink Wise - Face Clean Mode 💙"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 600)  # window size fixed

    # =============================================
    session_id = int(time.time())
    start_time = time.time()

    blink_counter = 0
    last_blink_time = time.time()

    frame_count = 0
    good_frames = 0
    bad_frames = 0
    corrections = 0
    previous_posture = None

    BLINK_TIMEOUT = 10  # seconds

    # ====================== Mediapipe Init ======================
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh, mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_result = face_mesh.process(rgb)
            pose_result = pose.process(rgb)

            ear = 0.0
            # ================= FACE PROCESS ================
            if face_result.multi_face_landmarks:

                lm = face_result.multi_face_landmarks[0].landmark

                # BLINK detection
                isBlink, ear = get_blink_status(face_result.multi_face_landmarks, w, h)
                if isBlink:
                    blink_counter += 1
                    last_blink_time = time.time()

                face_color = (255, 0, 0)  # BLUE
                eye_color = (255, 0, 0)

                # Draw FULL FACE outline
                outline_pts = []
                for idx in FULL_FACE_OUTLINE:
                    x = int(lm[idx].x * w)
                    y = int(lm[idx].y * h)
                    outline_pts.append((x, y))

                    cv2.circle(frame, (x, y), 2, face_color, -1)

                for i in range(len(outline_pts)-1):
                    cv2.line(frame, outline_pts[i], outline_pts[i+1], face_color, 1)

                # close oval
                cv2.line(frame, outline_pts[-1], outline_pts[0], face_color, 1)

                # ===== EYE outline =====
                # LEFT EYE
                for i in range(len(LEFT_EYE)):
                    a = LEFT_EYE[i]
                    b = LEFT_EYE[(i+1) % len(LEFT_EYE)]
                    x1, y1 = int(lm[a].x*w), int(lm[a].y*h)
                    x2, y2 = int(lm[b].x*w), int(lm[b].y*h)

                    cv2.circle(frame, (x1, y1), 2, eye_color, -1)
                    cv2.line(frame, (x1, y1), (x2, y2), eye_color, 1)

                # RIGHT EYE
                for i in range(len(RIGHT_EYE)):
                    a = RIGHT_EYE[i]
                    b = RIGHT_EYE[(i+1) % len(RIGHT_EYE)]
                    x1, y1 = int(lm[a].x*w), int(lm[a].y*h)
                    x2, y2 = int(lm[b].x*w), int(lm[b].y*h)

                    cv2.circle(frame, (x1, y1), 2, eye_color, -1)
                    cv2.line(frame, (x1, y1), (x2, y2), eye_color, 1)

            # ==================== POSTURE PROCESS ==================
            pose_landmarks = pose_result.pose_landmarks and [pose_result.pose_landmarks]
            posture_status, _ = get_posture_status(pose_landmarks, w, h)

            if posture_status in ("Good Posture", "Bad Posture"):
                frame_count += 1

                if posture_status == "Good Posture":
                    good_frames += 1
                else:
                    bad_frames += 1

                if previous_posture == "Good Posture" and posture_status == "Bad Posture":
                    corrections += 1

                previous_posture = posture_status

            # ==================== UI Display ==================
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 200), 2)

            cv2.putText(frame, f"Blinks: {blink_counter}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            blink_alert = time.time() - last_blink_time > BLINK_TIMEOUT
            if blink_alert:
                cv2.putText(frame, "BLINK PLEASE 👁️", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            if posture_status:
                color = (0,255,0) if posture_status == "Good Posture" else (0,0,255)
                cv2.putText(frame, f"POSTURE: {posture_status}", (10, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            elapsed_min = (time.time() - start_time) / 60
            cv2.putText(frame, f"Time: {elapsed_min:.1f} Min", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,50), 2)

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Save log
    total_minutes = (time.time() - start_time) / 60
    total_frames = good_frames + bad_frames

    good_pct = (good_frames/total_frames)*100 if total_frames else 0

    save_session_stats(session_id, blink_counter, total_minutes, good_pct, corrections)
    print("Session Saved Successfully!")


if __name__ == "__main__":
    main()
