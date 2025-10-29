import os
import sys


def main():
    print("Running local mode tests...")
    print("Running generate task...")
    ret = os.system(
        "python generate.py --prompt 'a dog playing in the park' --seed 1 --output tests/outputs/generate_local.png --model-mode local"
    )
    if ret != 0:
        print(f"Error: generate task (local) failed with exit code {ret}")
        sys.exit(1)
    print("Running inspire task...")
    ret = os.system(
        "python generate.py --image-path assets/zebra_balloons.jpeg --seed 1 --output tests/outputs/inspire_local.png --model-mode local"
    )
    if ret != 0:
        print(f"Error: inspire task (local) failed with exit code {ret}")
        sys.exit(1)
    print("Running refine task...")
    ret = os.system(
        'python generate.py --structured-prompt default_json_caption.json --prompt "change the zebra to an elephant" --seed 1 --output tests/outputs/refine_local.png --model-mode local'
    )
    if ret != 0:
        print(f"Error: refine task (local) failed with exit code {ret}")
        sys.exit(1)
    print("Running refine on image task...")
    ret = os.system(
        'python generate.py --image-path assets/zebra_balloons.jpeg --prompt "change the zebra to a tiger" --seed 1 --output tests/outputs/refine_on_image_local.png --model-mode local'
    )
    if ret != 0:
        print(f"Error: refine on image task (local) failed with exit code {ret}")
        sys.exit(1)

    print("Running default task...")
    ret = os.system("python generate.py --seed 1 --output tests/outputs/default.png")
    if ret != 0:
        print(f"Error: default task (local) failed with exit code {ret}")
        sys.exit(1)

    print("Running gemini mode tests...")
    print("Running generate task...")
    ret = os.system(
        "python generate.py --model-mode gemini --prompt 'a dog playing in the park' --seed 1 --output tests/outputs/generate_gemini.png"
    )
    if ret != 0:
        print(f"Error: generate task (gemini) failed with exit code {ret}")
        sys.exit(1)
    print("Running inspire task...")
    ret = os.system(
        "python generate.py --model-mode gemini --image-path assets/zebra_balloons.jpeg --seed 1 --output tests/outputs/inspire_gemini.png"
    )
    if ret != 0:
        print(f"Error: inspire task (gemini) failed with exit code {ret}")
        sys.exit(1)
    print("Running refine task...")
    ret = os.system(
        'python generate.py --model-mode gemini --structured-prompt default_json_caption.json --prompt "change the zebra to an elephant" --seed 1 --output tests/outputs/refine_gemini.png'
    )
    if ret != 0:
        print(f"Error: refine task (gemini) failed with exit code {ret}")
        sys.exit(1)
    print("Running refine on image task...")
    ret = os.system(
        'python generate.py --model-mode gemini --image-path assets/zebra_balloons.jpeg --prompt "change the zebra to a tiger" --seed 1 --output tests/outputs/refine_on_image_gemini.png'
    )
    if ret != 0:
        print(f"Error: refine on image task (gemini) failed with exit code {ret}")
        sys.exit(1)


if __name__ == "__main__":
    main()
