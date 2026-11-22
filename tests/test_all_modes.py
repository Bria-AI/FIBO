import os
import sys


def main():
    print("Running local mode tests...")
    print("Running generate task...")
    ret = os.system(
        "python generate.py --fibo1 --prompt 'a dog playing in the park' --seed 1 --output tests/outputs/generate_local.png --model-mode local"
    )
    if ret != 0:
        print(f"Error: generate task (local) failed with exit code {ret}")
        sys.exit(1)
    print("Running inspire task...")
    ret = os.system(
        "python generate.py --fibo1 --image-path assets/zebra_balloons.jpeg --seed 1 --output tests/outputs/inspire_local.png --model-mode local"
    )
    if ret != 0:
        print(f"Error: inspire task (local) failed with exit code {ret}")
        sys.exit(1)
    print("Running refine task...")
    ret = os.system(
        'python generate.py --fibo1 --structured-prompt default_json_caption.json --prompt "change the zebra to an elephant" --seed 1 --output tests/outputs/refine_local.png --model-mode local'
    )
    if ret != 0:
        print(f"Error: refine task (local) failed with exit code {ret}")
        sys.exit(1)
    print("Running refine on image task...")
    ret = os.system(
        'python generate.py --fibo1 --image-path assets/zebra_balloons.jpeg --prompt "change the zebra to a tiger" --seed 1 --output tests/outputs/refine_on_image_local.png --model-mode local'
    )
    if ret != 0:
        print(f"Error: refine on image task (local) failed with exit code {ret}")
        sys.exit(1)

    print("Running default task...")
    ret = os.system("python generate.py --fibo1 --seed 1 --output tests/outputs/default.png")
    if ret != 0:
        print(f"Error: default task (local) failed with exit code {ret}")
        sys.exit(1)

    print("Running TeaCache test...")
    ret = os.system(
        "python generate.py --fibo1 --prompt 'a dog playing in the park' --seed 1 --output tests/outputs/generate_teacache.png --model-mode local --enable-teacache"
    )
    if ret != 0:
        print(f"Error: TeaCache test failed with exit code {ret}")
        sys.exit(1)

    print("Running gemini mode tests...")
    print("Running generate task...")
    ret = os.system(
        "python generate.py --fibo1 --model-mode gemini --prompt 'a dog playing in the park' --seed 1 --output tests/outputs/generate_gemini.png"
    )
    if ret != 0:
        print(f"Error: generate task (gemini) failed with exit code {ret}")
        sys.exit(1)
    print("Running inspire task...")
    ret = os.system(
        "python generate.py --fibo1 --model-mode gemini --image-path assets/zebra_balloons.jpeg --seed 1 --output tests/outputs/inspire_gemini.png"
    )
    if ret != 0:
        print(f"Error: inspire task (gemini) failed with exit code {ret}")
        sys.exit(1)
    print("Running refine task...")
    ret = os.system(
        'python generate.py --fibo1 --model-mode gemini --structured-prompt default_json_caption.json --prompt "change the zebra to an elephant" --seed 1 --output tests/outputs/refine_gemini.png'
    )
    if ret != 0:
        print(f"Error: refine task (gemini) failed with exit code {ret}")
        sys.exit(1)
    print("Running refine on image task...")
    ret = os.system(
        'python generate.py --fibo1 --model-mode gemini --image-path assets/zebra_balloons.jpeg --prompt "change the zebra to a tiger" --seed 1 --output tests/outputs/refine_on_image_gemini.png'
    )
    if ret != 0:
        print(f"Error: refine on image task (gemini) failed with exit code {ret}")
        sys.exit(1)
    print("Running FIBO-lite mode tests...")
    ret = os.system(
        "python generate.py --fibo1 --model-mode gemini --prompt 'a dog playing in the park' --seed 1 --output tests/outputs/generate_fibo_lite.png --fibo-lite"
    )
    if ret != 0:
        print(f"Error: generate task (FIBO-lite) failed with exit code {ret}")
        sys.exit(1)

    print("Running Fibo 2.0 mode tests...")
    fibo2_tasks = [
        ("generate", "--prompt 'a dog playing in the park' --output tests/outputs/generate_fibo2.png"),
        (
            "generate with style",
            "--prompt 'a dog running on a beach' --style kodak_film --output tests/outputs/generate_fibo2_style.png",
        ),
        (
            "refine",
            "--structured-prompt tests/outputs/generate_fibo2.json --prompt 'make the dog brown' "
            "--output tests/outputs/refine_fibo2.png",
        ),
        ("inspire", "--image-path assets/zebra_balloons.jpeg --output tests/outputs/inspire_fibo2.png"),
        (
            "inspire with prompt",
            "--image-path assets/zebra_balloons.jpeg --prompt 'make it futuristic' "
            "--output tests/outputs/inspire_prompt_fibo2.png",
        ),
        (
            "edit",
            "--edit-image assets/zebra_balloons.jpeg --prompt 'change the zebra to a tiger' "
            "--output tests/outputs/edit_fibo2.png",
        ),
        (
            "multi-reference edit",
            "--edit-image assets/face_portrait.jpeg --edit-image assets/owl.png "
            "--prompt 'the person in <image_1> holding the owl from <image_2>' "
            "--output tests/outputs/edit_multiref_fibo2.png",
        ),
    ]
    for name, task_args in fibo2_tasks:
        print(f"Running {name} task (Fibo 2.0)...")
        ret = os.system(f"python generate.py --seed 1 {task_args}")
        if ret != 0:
            print(f"Error: {name} task (Fibo 2.0) failed with exit code {ret}")
            sys.exit(1)


if __name__ == "__main__":
    main()
