import subprocess
import sys

scripts = [
    "zeroShotTesting.py",
    "oneShotTesting.py",
    "twoShotTesting.py",
    "threeShotTesting.py",
    "evaluation/eval.py",
]

for script in scripts:
    print(f"\nStarting: {script}")

    result = subprocess.run([sys.executable, script])

    print(f"Finished: {script}")

print("\n All scripts completed!")