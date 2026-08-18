"""`rn` must send images in the format the server actually reads.

THE BUG THIS PINS: osaurus exposes an OpenAI-compatible endpoint and SILENTLY DROPS
the Ollama-style `{"images": [b64]}` key. It does not error. It answers as though no
image were attached, so the model invents a plausible description and `rn` writes that
invention into the filename.

Measured against the live server with a picture of a red circle:

    {"images": [b64]}          "Please provide the image you are referring to..."
    no image at all            "Please provide the image you are referring to..."
    content parts + image_url  "Red semi-circle."

The first two are identical, which is the whole proof: the key was ignored. Three
unmistakable, mutually unrelated fixtures went through the old path and produced
"large white building blue sky", "large brown dog" and "large brown bear forest" --
none of which appeared in any of them. After the fix the same three produce "red
curved shape", "small white circles" and "...tan rectangle, ...green rectangle".

A failure that returns confident, well-formed, wrong output is worse than one that
errors, because nothing downstream can tell it went wrong.
"""

import base64
from unittest.mock import patch

import pytest


def _png(tmp_path):
    from PIL import Image

    p = tmp_path / "probe.png"
    Image.new("RGB", (8, 8), "red").save(p)
    return p


def captured_messages(tmp_path, suffix=".png"):
    """Run the VLM path and return the messages it handed to the shared client."""
    import rename.llm as rl

    img = _png(tmp_path)
    target = img.with_suffix(suffix) if suffix != ".png" else img
    if target != img:
        target.write_bytes(img.read_bytes())
    seen = {}

    def _capture(model, messages, host, timeout, api_key=""):
        seen["messages"] = messages
        return {"content": "red circle", "error": ""}

    with patch("rename.llm._shared_call", _capture):
        rl.query_vlm_for_filename(target, "http://localhost:1337", "vision-model")
    return seen["messages"]


class TestThePayloadUsesOpenAiContentParts:
    def test_the_message_content_is_a_list_of_parts(self, tmp_path):
        msgs = captured_messages(tmp_path)
        assert isinstance(msgs[0]["content"], list), (
            "content must be OpenAI parts; a bare string cannot carry an image"
        )

    def test_it_carries_an_image_url_part(self, tmp_path):
        parts = captured_messages(tmp_path)[0]["content"]
        assert any(p.get("type") == "image_url" for p in parts)

    def test_the_image_is_a_base64_data_uri(self, tmp_path):
        parts = captured_messages(tmp_path)[0]["content"]
        url = next(p["image_url"]["url"] for p in parts if p.get("type") == "image_url")
        assert url.startswith("data:image/"), url[:40]
        assert ";base64," in url
        payload = url.split(";base64,", 1)[1]
        assert base64.b64decode(payload)[:4] == b"\x89PNG", "not a real PNG payload"

    def test_the_prompt_still_travels_with_it(self, tmp_path):
        parts = captured_messages(tmp_path)[0]["content"]
        assert any(p.get("type") == "text" and p.get("text") for p in parts)

    def test_the_ollama_images_key_is_not_used(self, tmp_path):
        """The exact shape that was silently ignored. If this comes back, `rn` goes
        blind again and says nothing."""
        msgs = captured_messages(tmp_path)
        assert "images" not in msgs[0], "the ignored Ollama key is back"

    def test_a_jpeg_declares_its_own_mime_type(self, tmp_path):
        parts = captured_messages(tmp_path, suffix=".jpg")[0]["content"]
        url = next(p["image_url"]["url"] for p in parts if p.get("type") == "image_url")
        assert url.startswith("data:image/jpeg")


class TestQuirksLeaveMultimodalMessagesAlone:
    """Every quirk rewrite is a string operation. The first version of the vision fix
    raised `AttributeError: 'list' object has no attribute 'lower'` the moment a
    multimodal message reached them."""

    PARTS = [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]

    @pytest.mark.parametrize("model", ["qwen3.8-27b-mxfp8", "gemma-4-12b-it-mxfp8", "foundation"])
    def test_a_list_content_message_survives_untouched(self, model):
        from lib.llm.quirks import apply_model_quirks

        out = apply_model_quirks([{"role": "user", "content": self.PARTS}], model)
        assert out[0]["content"] == self.PARTS

    def test_string_messages_are_still_processed(self, model="qwen3.8-27b-mxfp8"):
        """Calibration: the guard must not disable quirks for ordinary prompts."""
        from lib.llm.quirks import apply_model_quirks

        out = apply_model_quirks(
            [{"role": "system", "content": "Extract the venues as JSON"}], model
        )
        assert isinstance(out[0]["content"], str)

    def test_a_mixed_conversation_keeps_both_shapes(self):
        from lib.llm.quirks import apply_model_quirks

        out = apply_model_quirks(
            [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": self.PARTS},
            ],
            "gemma-4-12b-it-mxfp8",
        )
        assert isinstance(out[0]["content"], str)
        assert out[1]["content"] == self.PARTS
