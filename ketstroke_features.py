from pynput import keyboard
import time
import csv

PRINT_KEYS = False  # set True if you want to see pressed keys

keystrokes = []
recording = True

# -----------------------------------
# Event handlers
# -----------------------------------

def on_press(key):
    global recording

    t = time.time()

    if key == keyboard.Key.enter:
        print("\nENTER detected — stopping recording.")
        recording = False
        return False  # this stops the listener

    try:
        k = key.char
    except:
        k = str(key)

    keystrokes.append(("PRESS", k, t))

    if PRINT_KEYS:
        print(f"P {k}")


def on_release(key):
    t = time.time()

    try:
        k = key.char
    except:
        k = str(key)

    keystrokes.append(("RELEASE", k, t))

    if PRINT_KEYS:
        print(f"R {k}")


# -----------------------------------
# Capture
# -----------------------------------

print("\nType the password/phrase and press ENTER to finish:")
print(">> ")

with keyboard.Listener(on_press=on_press,
                       on_release=on_release) as listener:
    listener.join()

print("Recording stopped.")

# -----------------------------------
# Feature extraction
# -----------------------------------

# Compute hold times (dwell)
hold_times = {}
dwell = []

for event, key, t in keystrokes:
    if event == "PRESS":
        hold_times[key] = t
    elif event == "RELEASE" and key in hold_times:
        dwell.append((key, t - hold_times[key]))
        del hold_times[key]

# Compute flight times
flight = []

for i in range(1, len(keystrokes)):
    prev = keystrokes[i - 1]
    curr = keystrokes[i]

    if prev[0] == "RELEASE" and curr[0] == "PRESS":
        flight.append((prev[1], curr[1], curr[2] - prev[2]))

# -----------------------------------
# Save results
# -----------------------------------

with open("keystroke_features.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Type", "Key1", "Key2", "Time"])

    for k, t in dwell:
        writer.writerow(["DWELL", k, "", round(t, 5)])

    for k1, k2, t in flight:
        writer.writerow(["FLIGHT", k1, k2, round(t, 5)])

print("\n✅ Data written to keystroke_features.csv")
print(f"Captured {len(dwell)} dwell events, {len(flight)} flight events")
