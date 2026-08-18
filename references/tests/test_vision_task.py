"""`image_real` measures whether a model can SEE, and is calibrated to prove it.

Every other image task in this suite sends its prompt as TEXT. Ten models scored 100
on `image_rename` while none of them had been shown an image, which is why
`best_models.vlm` had to be marked UNMEASURED -- and why, when the vision transport
turned out to be silently dropping the image entirely, nothing noticed.

The calibration that makes this task trustworthy, run live against gemma-4-12b:

    same prompt, images ATTACHED    "1. Red semicircle 2. Green triangle 3. Blue
                                     rectangle"                          -> 100
    same prompt, images STRIPPED    "Please provide the images you would like me
                                     to describe..."                     ->   0

Same model, same words, only the images differ. A task that cannot produce that gap
is measuring vocabulary, not sight.
"""

import pytest
from eval.vision_fixtures import VISION_FIXTURES, data_uri, image_message, render
from lib.validators.vision_validator import matched_fixtures, validate_image_description


class TestTheFixturesCannotBePassedBlind:
    """Three unmistakable, mutually unrelated subjects. A guess gets one at best."""

    def test_there_are_several_fixtures(self):
        assert len(VISION_FIXTURES) >= 3, "one image makes the task a coin flip"

    def test_no_two_fixtures_share_an_accepted_word(self):
        """If two fixtures both accepted "circle", one description could satisfy two
        of them and a blind model would score higher than it earned."""
        seen = {}
        for fixture in VISION_FIXTURES:
            for word in fixture["accept"]:
                assert word not in seen, (
                    f"{word!r} accepted by both {seen.get(word)} and {fixture['name']}"
                )
                seen[word] = fixture["name"]

    def test_the_images_contain_no_text(self):
        """rn reaches its vision path only when OCR found NOTHING. A fixture with
        readable words could be passed by a model that reads but cannot see."""
        for fixture in VISION_FIXTURES:
            assert "text" not in {kind for kind, _g, _c in fixture["shapes"]}

    def test_rendering_is_deterministic(self):
        """Same spec, same pixels -- otherwise the ground truth drifts from the image."""
        for fixture in VISION_FIXTURES:
            assert render(fixture) == render(fixture)

    def test_each_fixture_renders_a_real_png(self):
        for fixture in VISION_FIXTURES:
            assert render(fixture)[:4] == b"\x89PNG"


class TestTheScoreTracksWhatWasSeen:
    def test_naming_every_subject_scores_full(self):
        assert validate_image_description("red circle, green triangle, blue square")[0] == 100

    def test_naming_two_scores_two_thirds(self):
        score, msg = validate_image_description("a red circle beside a green triangle")
        assert score == 67
        assert "blue_square" in msg

    def test_a_hallucinated_answer_scores_zero(self):
        """The literal output rn produced while its images were being dropped."""
        score, msg = validate_image_description("large brown dog running through a forest")
        assert score == 0
        assert "no image content recognised" in msg

    def test_zero_says_the_payload_may_not_be_arriving(self):
        """The failure mode is a silently-ignored image, so the message has to point
        at the transport rather than at the model's eyesight."""
        assert "payload" in validate_image_description("nothing relevant here")[1]

    def test_an_empty_answer_is_not_scored_as_a_miss(self):
        assert validate_image_description("")[1] == "empty response"

    def test_no_ground_truth_scores_zero_rather_than_inventing_one(self):
        assert validate_image_description("anything at all", fixtures=[])[0] == 0


class TestTheCrossImageControl:
    """Scoring one image's description against a DIFFERENT image's key must fail.

    Without this, a validator that merely liked colour-and-shape words would pass
    every fixture on any plausible answer, and would have scored the hallucinating
    transport a comfortable pass.
    """

    @pytest.mark.parametrize("i", range(len(VISION_FIXTURES)))
    def test_a_description_matches_only_its_own_fixture(self, i):
        own = VISION_FIXTURES[i]
        description = " ".join(own["accept"])
        results = dict(matched_fixtures(description, VISION_FIXTURES))
        assert results[own["name"]] is True, "a fixture must match its own words"
        others = [name for name, ok in results.items() if ok and name != own["name"]]
        assert others == [], f"{own['name']}'s words also matched {others}"


class TestThePayloadIsTheOneRnUses:
    """A task that proved something about a DIFFERENT transport would prove nothing
    about the tool that ships -- the exact class backlog item 9 describes."""

    def test_the_message_carries_openai_image_parts(self):
        parts = image_message("describe")[0]["content"]
        kinds = [p["type"] for p in parts]
        assert kinds[0] == "text"
        assert kinds.count("image_url") == len(VISION_FIXTURES)

    def test_every_image_is_a_base64_png_data_uri(self):
        for fixture in VISION_FIXTURES:
            uri = data_uri(fixture)
            assert uri.startswith("data:image/png;base64,")

    def test_the_ollama_key_is_not_used(self):
        """The shape osaurus silently ignores."""
        assert "images" not in image_message("describe")[0]

    def test_the_task_is_registered_with_this_validator(self):
        from eval.tasks_core import TASKS

        assert TASKS["image_real"]["validator"] is validate_image_description
