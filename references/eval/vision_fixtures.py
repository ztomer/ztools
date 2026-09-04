"""Synthetic images with known contents, for measuring whether a model can SEE.

Generated rather than checked in, so the ground truth lives next to the drawing that
produces it and the two cannot drift. Each fixture is deliberately unmistakable and
mutually unrelated -- a model that gets one right by luck cannot get all three right
by luck.

DELIBERATELY NO TEXT IN THE IMAGES. `rn` reaches its vision path only when OCR found
nothing, and a fixture containing readable words could be passed by a model that reads
but cannot see. That is the same defect this task exists to close: `image_rename`
sends its prompt as TEXT, so ten models scoring 100 on it proved only that they can
emit a filename-shaped string.
"""

import base64
import io
from typing import Dict, List

#: Each fixture: how to draw it, and the words that prove the model saw it.
#: `accept` is generous about synonyms (a circle is legitimately "round", "dot",
#: "sphere") because the task measures SIGHT, not vocabulary. It is not generous
#: about the subject: nothing here overlaps between fixtures.
VISION_FIXTURES: List[Dict] = [
    {
        "name": "red_circle",
        "background": "white",
        "shapes": [("ellipse", (120, 120, 392, 392), (220, 30, 30))],
        "accept": ["red", "circle", "round", "dot", "sphere", "ball", "crimson"],
    },
    {
        "name": "green_triangle",
        "background": "white",
        "shapes": [("polygon", [(256, 110), (410, 400), (102, 400)], (30, 150, 60))],
        "accept": ["green", "triangle", "triangular", "cone", "arrow", "pyramid"],
    },
    {
        "name": "blue_square",
        "background": "white",
        "shapes": [("rectangle", (140, 140, 372, 372), (40, 70, 200))],
        "accept": ["blue", "square", "rectangle", "box", "cube", "navy"],
    },
]


def render(fixture: Dict) -> bytes:
    """Draw one fixture as PNG bytes. Deterministic: same spec, same pixels."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (512, 512), fixture["background"])
    draw = ImageDraw.Draw(img)
    for kind, geometry, colour in fixture["shapes"]:
        getattr(draw, kind)(geometry, fill=colour)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def data_uri(fixture: Dict) -> str:
    """The fixture as an OpenAI `image_url` payload.

    A data URI with content parts, NOT the Ollama-style `{"images": [b64]}` key --
    osaurus silently ignores that key and answers as though no image were attached,
    which is how `rn` came to rename every image from a hallucination.
    """
    return "data:image/png;base64," + base64.b64encode(render(fixture)).decode()


def image_message(prompt: str) -> List[Dict]:
    """One user message carrying the prompt and every fixture image.

    All fixtures in a single call on purpose. One image would make the task a coin
    flip that a blind model passes one time in three; three unrelated images make
    that essentially impossible while still costing one request.
    """
    parts: List[Dict] = [{"type": "text", "text": prompt}]
    for fixture in VISION_FIXTURES:
        parts.append({"type": "image_url", "image_url": {"url": data_uri(fixture)}})
    return [{"role": "user", "content": parts}]
