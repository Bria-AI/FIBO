# FIBO Inference Examples

This directory contains examples of how to use the FIBO inference scripts for different tasks.

## Tasks

The FIBO inference pipeline supports several tasks for generating and manipulating images.

### Generate

The `generate` task creates a structured JSON prompt from a short natural-language prompt using a Vision-Language Model (VLM), and then generates an image.

**Example:**
```bash
python generate.py --prompt "a majestic lion in the savannah" --output examples/outputs/generate.png
```

### Inspire

The `inspire` task takes an input image and uses the VLM to generate a structured JSON prompt that describes it. This prompt is then used to generate a new image.

**Example:**
```bash
python generate.py --image-path assets/zebra_balloons.jpeg --output examples/outputs/inspire.png
```

### Refine

The `refine` task modifies an existing structured JSON prompt based on editing instructions. This is useful for iterating on an idea without starting from scratch.

**Example:**
First, generate an image and its structured prompt:
```bash
python generate.py --prompt "a cat sitting on a mat" --output examples/outputs/refine_original.png
```
Then, refine it with editing instructions:
```bash
python generate.py --structured-prompt examples/outputs/refine_original.json --prompt "make the cat a dog" --output examples/outputs/refine_edited.png
```

### Raw (JSON)

The `raw` task allows you to provide a detailed, structured JSON prompt directly to the image generation pipeline, bypassing the VLM. This gives you maximum control over the output.

**Example:**
```bash
python generate.py --json-prompt examples/outputs/generate.json --output examples/outputs/generate_from_raw.png
```
You can also pass the JSON as a string.

## JSON Input Schema

When using the `raw` task, you provide a JSON object with a specific structure. Below is an overview of the schema and an example.

The main keys in the JSON prompt are:
- `short_description`: A brief summary of the image.
- `objects`: A list of objects in the scene, each with properties like `description`, `location`, `shape_and_color`, etc.
- `background_setting`: A description of the background.
- `lighting`: Details about the lighting conditions, direction, and shadows.
- `aesthetics`: Information about composition, color scheme, and mood.
- `photographic_characteristics`: Camera-related details like depth of field, focus, and angle.
- `style_medium`: The artistic medium (e.g., "photograph", "oil painting").
- `text_render`: Any text to be rendered in the image.
- `context`: Additional context or conceptual information about the image.
- `artistic_style`: The artistic style (e.g., "Surreal, realistic").

### Example JSON Input

Here is an example of a JSON prompt for the `raw` task.

```json
{
  "short_description": "A realistic image features a zebra standing on a concrete sidewalk next to a red fire hydrant. The zebra is positioned prominently in the center-right of the frame, facing towards the right with its head slightly lowered. The fire hydrant is in the bottom-left foreground. The background consists of a plain, light-colored wall, suggesting an urban or industrial setting. The lighting is even, highlighting the zebra's distinctive black and white stripes and the vibrant red of the hydrant.",
  "objects": [
    {
      "description": "A full-grown zebra with distinct black and white stripes covering its entire body. Its mane is short and upright, and its tail is long and bushy at the end. The zebra appears healthy and well-fed.",
      "location": "center-right",
      "relationship": "The zebra is standing next to the fire hydrant, appearing to be observing it or simply pausing in its vicinity.",
      "relative_size": "large within frame",
      "shape_and_color": "Elongated, equine shape with alternating black and white stripes.",
      "texture": "The zebra's coat appears smooth and short, typical of a mammal's fur. End of texture answer.",
      "appearance_details": "The stripes are sharply defined and vary in width and pattern across its body. Its muzzle is dark, and its eyes are dark and alert.",
      "number_of_objects": null,
      "pose": "Standing upright on all four legs, with its head slightly lowered and turned to its right.",
      "expression": "Calm and observant.",
      "clothing": null,
      "action": "Standing still.",
      "gender": "Unidentifiable.",
      "skin_tone_and_texture": null,
      "orientation": "Facing right."
    },
    {
      "description": "A classic red fire hydrant, cylindrical in shape with various valves and caps. It has a chain connecting two of its components.",
      "location": "bottom-left foreground",
      "relationship": "The fire hydrant is situated on the sidewalk, directly in front of the zebra's left front leg.",
      "relative_size": "medium",
      "shape_and_color": "Cylindrical, bright red.",
      "texture": "The fire hydrant appears to have a smooth, painted metallic surface with some visible wear and tear. End of texture answer.",
      "appearance_details": "It has a slightly weathered appearance, with some dirt or grime near its base.",
      "number_of_objects": null,
      "pose": null,
      "expression": null,
      "clothing": null,
      "action": null,
      "gender": null,
      "skin_tone_and_texture": null,
      "orientation": "Upright."
    }
  ],
  "background_setting": "The background is a plain, light gray concrete wall, suggesting an urban environment. Below the wall, there is a narrow strip of what appears to be dry grass or dirt, indicating a small patch of nature in an otherwise man-made setting. The ground is a concrete sidewalk with a curb separating it from a darker asphalt road.",
  "lighting": {
    "conditions": "Bright daylight",
    "direction": "Evenly lit, possibly from above or slightly front-lit.",
    "shadows": "Subtle, soft shadows are visible beneath the zebra and the fire hydrant, indicating a clear day with diffused light."
  },
  "aesthetics": {
    "composition": "Centered, with the zebra occupying the majority of the frame and the fire hydrant providing a contrasting element in the foreground.",
    "color_scheme": "Monochromatic (black and white) for the zebra, contrasted with a vibrant red for the hydrant and neutral grays for the background.",
    "mood_atmosphere": "Surreal and intriguing, due to the unexpected presence of a zebra in an urban setting."
  },
  "photographic_characteristics": {
    "depth_of_field": "Shallow, with the zebra and fire hydrant in sharp focus and the background slightly blurred.",
    "focus": "Sharp focus on subject.",
    "camera_angle": "Eye-level.",
    "lens_focal_length": "Standard."
  },
  "style_medium": "photograph",
  "text_render": [],
  "context": "This is an art piece or conceptual photograph, likely created digitally, that plays on the juxtaposition of a wild animal in an unexpected urban environment. It could be used for advertising, editorial content, or as a standalone piece of art designed to provoke thought or amusement.",
  "artistic_style": "Surreal, realistic"
}
```
