from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import time

from blink_module import get_blink_status, LEFT_EYE, RIGHT_EYE
from posture_module import get_posture_status

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

FULL_FACE_OUTLINE = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

# Per-person state
people = {}  # id -> dict
next_person_id = 1

# Global posture/time metrics
global_metrics = {
    "posture_status": "No Person",
    "good_posture_pct": 0.0,
    "posture_alert": False,
    "elapsed_min": 0.0,
}

start_time = time.time()

# Streaming flag (for ON/OFF)
is_streaming = True


def assign_ids(multi_faces, w, h):
    """
    Assign or keep a person ID for each detected face
    based on face center proximity.
    """
    global people, next_person_id

    now = time.time()
    face_entries = []

    for fl in multi_faces:
        lm = fl.landmark
        cx = int(lm[1].x * w)
        cy = int(lm[1].y * h)
        face_entries.append({"lm": lm, "raw": fl, "cx": cx, "cy": cy, "id": None})

    # Match each new face to existing
    for face in face_entries:
        best_id = None
        best_dist_sq = 999999999

        for pid, pdata in people.items():
            dx = face["cx"] - pdata["cx"]
            dy = face["cy"] - pdata["cy"]
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_id = pid

        # If far, create new ID
        if best_id is None or best_dist_sq > (80 * 80):
            pid = next_person_id
            next_person_id += 1
            people[pid] = {
                "cx": face["cx"],
                "cy": face["cy"],
                "blink_count": 0,
                "closed_frames": 0,
                "last_blink_time": now,
                "ear": 0.0,
                "blink_alert": False,
                "created_at": now,
                "last_seen": now,
            }
        else:
            pid = best_id
            people[pid]["cx"] = face["cx"]
            people[pid]["cy"] = face["cy"]
            people[pid]["last_seen"] = now

        face["id"] = pid

    # Remove stale persons
    remove_ids = []
    for pid, pdata in people.items():
        if now - pdata["last_seen"] > 10:
            remove_ids.append(pid)
    for pid in remove_ids:
        del people[pid]

    return face_entries


def gen_frames():
    """Camera + analysis + MJPEG stream with start/stop support."""
    global global_metrics, is_streaming

    print("gen_frames started...")

    # posture counters local
    good_frames = 0
    bad_frames = 0

    BLINK_TIMEOUT = 10
    MIN_CLOSED_FRAMES = 2

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    with mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh, mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while is_streaming:
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_result = face_mesh.process(rgb)
            pose_result = pose.process(rgb)

            # ------------- MULTI-FACE BLINK -------------
            if face_result.multi_face_landmarks:
                faces = assign_ids(face_result.multi_face_landmarks, w, h)

                for face in faces:
                    pid = face["id"]
                    lm = face["lm"]
                    pdata = people[pid]

                    is_closed, ear = get_blink_status([face["raw"]], w, h)
                    pdata["ear"] = ear

                    if is_closed:
                        pdata["closed_frames"] += 1
                    else:
                        if pdata["closed_frames"] >= MIN_CLOSED_FRAMES:
                            pdata["blink_count"] += 1
                            pdata["last_blink_time"] = time.time()
                        pdata["closed_frames"] = 0

                    pdata["blink_alert"] = (
                        time.time() - pdata["last_blink_time"] > BLINK_TIMEOUT
                    )

                    # Draw face outline & eyes
                    face_color = (255, 0, 0)
                    eye_color = (255, 0, 0)

                    outline_pts = []
                    for idx in FULL_FACE_OUTLINE:
                        x = int(lm[idx].x * w)
                        y = int(lm[idx].y * h)
                        outline_pts.append((x, y))
                        cv2.circle(frame, (x, y), 2, face_color, -1)

                    for i in range(len(outline_pts) - 1):
                        cv2.line(frame, outline_pts[i], outline_pts[i + 1], face_color, 1)
                    cv2.line(frame, outline_pts[-1], outline_pts[0], face_color, 1)

                    # left eye
                    for i in range(len(LEFT_EYE)):
                        a = LEFT_EYE[i]
                        b = LEFT_EYE[(i + 1) % len(LEFT_EYE)]
                        x1, y1 = int(lm[a].x * w), int(lm[a].y * h)
                        x2, y2 = int(lm[b].x * w), int(lm[b].y * h)
                        cv2.circle(frame, (x1, y1), 2, eye_color, -1)
                        cv2.line(frame, (x1, y1), (x2, y2), eye_color, 1)

                    # right eye
                    for i in range(len(RIGHT_EYE)):
                        a = RIGHT_EYE[i]
                        b = RIGHT_EYE[(i + 1) % len(RIGHT_EYE)]
                        x1, y1 = int(lm[a].x * w), int(lm[a].y * h)
                        x2, y2 = int(lm[b].x * w), int(lm[b].y * h)
                        cv2.circle(frame, (x1, y1), 2, eye_color, -1)
                        cv2.line(frame, (x1, y1), (x2, y2), eye_color, 1)

                    # show person id
                    cv2.putText(frame, f"ID {pid}",
                                (outline_pts[0][0], outline_pts[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # ------------- POSTURE (global) -------------
            posture_status = "No Person"
            pose_landmarks = pose_result.pose_landmarks and [pose_result.pose_landmarks]
            posture_status_tmp, _ = get_posture_status(pose_landmarks, w, h)
            if posture_status_tmp != "No Person":
                posture_status = posture_status_tmp

            if posture_status in ("Good Posture", "Bad Posture"):
                if posture_status == "Good Posture":
                    good_frames += 1
                else:
                    bad_frames += 1

            total_frames = good_frames + bad_frames
            if total_frames > 0:
                good_pct = (good_frames / total_frames) * 100.0
            else:
                good_pct = 0.0

            posture_alert = (posture_status == "Bad Posture")

            elapsed_min = (time.time() - start_time) / 60

            cv2.putText(frame, f"Posture: {posture_status}", (10, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if posture_status == "Good Posture" else (0, 0, 255),
                        2)
            cv2.putText(frame, f"Good Posture %: {good_pct:.1f}", (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

            global_metrics["posture_status"] = posture_status
            global_metrics["good_posture_pct"] = round(good_pct, 1)
            global_metrics["posture_alert"] = posture_alert
            global_metrics["elapsed_min"] = round(elapsed_min, 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    print("Camera released (stream stopped)")


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    global is_streaming
    is_streaming = True
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stop_stream")
def stop_stream():
    global is_streaming
    is_streaming = False
    return "STOPPED"


@app.route("/metrics")
def get_metrics():
    """Return list of persons + global posture metrics."""
    people_list = []
    now = time.time()
    for pid, p in people.items():
        people_list.append({
            "id": pid,
            "blink_count": p["blink_count"],
            "ear": round(p["ear"], 3),
            "blink_alert": p["blink_alert"],
            "last_seen_ago": round(now - p["last_seen"], 1),
        })

    return jsonify({
        "people": people_list,
        "posture_status": global_metrics["posture_status"],
        "good_posture_pct": global_metrics["good_posture_pct"],
        "posture_alert": global_metrics["posture_alert"],
        "elapsed_min": global_metrics["elapsed_min"],
    })


if __name__ == "__main__":
    app.run(debug=False)
