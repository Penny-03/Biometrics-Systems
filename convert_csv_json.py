import csv
import json
from datetime import datetime, UTC
import time

CSV_FILE = "DSL-StrongPasswordData.csv"
JSON_FILE = "output.json"

PASSWORD_ORDER = [
    ("period", ".", False),
    ("t", "t", False),
    ("i", "i", False),
    ("e", "e", False),
    ("five", "5", False),
    ("Shift.r", "R", True),     # R requires Shift
    ("o", "o", False),
    ("a", "a", False),
    ("n", "n", False),
    ("l", "l", False),
    ("Return", "Return", False)
]

def build_session_json(row):
    subject = row["subject"]
    sessionIndex = int(row["sessionIndex"])
    rep = int(row["rep"])

    sessionId = int(time.time() * 1000)

    keystrokes = []

    # Start everything at timestamp 0
    running_press_time = 0.0

    for i, (csv_key, output_key, is_shift) in enumerate(PASSWORD_ORDER):
        # Dwell time
        H_col = f"H.{csv_key}"
        dwell = float(row[H_col])

        # Press time
        press = running_press_time

        # Release time
        release = press + dwell

        # Flight times
        if i == 0:
            flight = 0.0
        else:
            UD_col = f"UD.{PASSWORD_ORDER[i-1][0]}.{csv_key}"
            flight = float(row[UD_col])

        keystrokes.append({
            "sessionId": sessionId,
            "userId": subject,
            "key": output_key,
            "pressTime": press,
            "releaseTime": release,
            "dwellTime": dwell,
            "flightTime": flight,
            "timestamp": datetime.now(UTC).isoformat(),
            "phraseNumber": rep
        })

        running_press_time = press + (0 if i == 0 else float(row[f"DD.{PASSWORD_ORDER[i-1][0]}.{csv_key}"]))

    session_json = {
        "sessionId": sessionId,
        "userId": subject,
        "keystrokeCount": len(keystrokes),
        "phrasesCompleted": 1,
        "data": keystrokes
    }

    return session_json


def main():
    sessions = []

    with open(CSV_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sessions.append(build_session_json(row))

    output = {
        "metadata": {
            "userId": "MULTI-SUBJECT",
            "totalSessions": len(sessions),
            "collectionDate": datetime.now(UTC).isoformat(),
            "textPrompt": ".tie5Roanl"
        },
        "sessions": sessions
    }

    with open(JSON_FILE, "w", encoding="utf-8") as out:
        json.dump(output, out, indent=2)

    print(f"✓ Converted {len(sessions)} rows into JSON → {JSON_FILE}")


if __name__ == "__main__":
    main()
