import csv
from pathlib import Path
from datetime import datetime

# sessions.csv path => project_root/logs/sessions.csv
LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "sessions.csv"


def init_log_file():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "session_id",
                "datetime",
                "blinks",
                "total_minutes",
                "blink_per_min",
                "good_posture_pct",
                "corrections"
            ])


def save_session_stats(session_id, blinks, total_minutes,
                       good_posture_pct, corrections):
    """
    Save one session summary into CSV.
    """
    init_log_file()

    if total_minutes <= 0:
        blink_per_min = 0.0
    else:
        blink_per_min = blinks / total_minutes

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            session_id,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            int(blinks),
            round(total_minutes, 2),
            round(blink_per_min, 2),
            round(good_posture_pct, 2),
            int(corrections)
        ])
