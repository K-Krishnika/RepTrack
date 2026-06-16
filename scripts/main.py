import subprocess
import sys

print("\nAvailable Exercises:")
print("1. push_up")
print("2. squat")
print("3. bicep_curl")

exercise = input(
    "\nChoose exercise: "
).strip().lower()

if exercise == "bicep_curl":

    subprocess.run(
        [sys.executable, "scripts/fp.py"]
    )

elif exercise in [
    "push_up",
    "squat"
]:

    reps = input(
        "Enter target reps: "
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/tc.py",
            exercise,
            reps
        ]
    )

else:

    print(
        "Invalid exercise."
    )