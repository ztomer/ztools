
from eval.tasks_core import _extract_items_from_text


class TestExtractItemsFromText:
    def test_empty_text(self):
        assert _extract_items_from_text("") == []

    def test_markdown_table(self):
        text = "| Name | Location |\n| --- | --- |\n| Park | Toronto |\n| Zoo | Vaughan |"
        items = _extract_items_from_text(text)
        assert len(items) == 2
        assert items[0]["name"] == "Park"
        assert items[1]["name"] == "Zoo"

    def test_markdown_table_with_custom_header(self):
        text = "| Event | Venue |\n| --- | --- |\n| Concert | Hall |\n| Play | Theatre |"
        items = _extract_items_from_text(text)
        assert len(items) == 2
        assert items[0]["name"] == "Concert"

    def test_bullet_list_with_colons(self):
        text = "- name: Park\n- name: Zoo\n- name: Museum"
        items = _extract_items_from_text(text)
        assert len(items) == 3
        assert all(item.get("name") for item in items)

    def test_bullet_with_location_colon(self):
        text = "- Park: Toronto\n- Zoo: Vaughan"
        items = _extract_items_from_text(text)
        assert len(items) == 2
        assert items[0].get("park") == "Toronto" or items[0].get("location") == "Toronto"

    def test_bullet_with_name_location_colon(self):
        text = "- name: Park, location: Toronto"
        items = _extract_items_from_text(text)
        # "name:" is parsed as field marker, but comma on the same line keeps the rest as the name value
        assert items == [{"name": "Park, location: Toronto"}]

    def test_bullet_with_separator(self):
        text = "- Park - Toronto\n- Zoo - Vaughan"
        items = _extract_items_from_text(text)
        assert items[0]["name"] == "Park"
        assert items[0]["location"] == "Toronto"

    def test_simple_bullets_without_detail(self):
        text = "- just a line\n- another line"
        items = _extract_items_from_text(text)
        # 2 bullet lines → 2 single-name items
        assert items == [{"name": "just a line"}, {"name": "another line"}]

    def test_table_header_row_skipped(self):
        text = "| ---- | ---- |\n| Data | Point |"
        items = _extract_items_from_text(text)
        assert len(items) == 1

    def test_table_duplicate_header_in_rows(self):
        text = "| Name | Location |\n| Name | Location |\n| Park | Toronto |"
        items = _extract_items_from_text(text)
        assert len(items) == 1

    def test_no_parseable_content(self):
        text = "Just plain prose with no structure at all\nand nothing to extract"
        items = _extract_items_from_text(text)
        assert items == []


class TestTaxesTasksAreRunnable:
    """README advertised three taxes tasks that nothing wired up.

    The snapshots, the rubric and the validators all shipped, but no TASKS
    entry loaded them, so `ev --task taxes_synthesis` answered "Unknown task"
    and the documented GT-leak checks never ran.
    """

    NAMES = ("taxes_anomalies", "taxes_audit_readiness", "taxes_synthesis")

    def test_registered_with_their_own_prompts_and_validators(self):
        from eval.tasks_core import TASKS

        for name in self.NAMES:
            task = TASKS[name]
            roles = [m["role"] for m in task["messages"]]
            assert roles == ["system", "user"]
            assert len(task["messages"][1]["content"]) > 500, "snapshot prompt is substantial"
            assert task["validator"].__name__ == f"validate_{name}"
            # The validators consume raw text, so pre-parsing would break them.
            assert task["parse_json"] is False

    def test_readme_task_list_matches_reality(self):
        import re

        from eval.tasks_core import TASKS
        from lib.paths import repo_path

        readme = repo_path("README.md")
        if readme is None:
            import pytest

            pytest.skip("not running from a checkout")
        line = next(
            line for line in readme.read_text().splitlines() if line.startswith("**Tasks:**")
        )
        advertised = re.findall(r"`([a-z_]+)`", line)
        assert advertised, "expected a backticked task list"
        assert [t for t in advertised if t not in TASKS] == []

    def test_scored_through_the_real_eval_path(self):
        """Drive the task the way `ev` does, not by calling the validator directly.

        eval/run.py calls every validator as `validator(content, source_text=...)`;
        the taxes validators took only `(output)`, so each run raised a TypeError
        that run.py's blanket except turned into a permanent score of 0 tagged
        as an infra failure — the tasks ran the model but could never be scored.
        """
        from eval.run import _validate_result
        from eval.tasks_core import TASKS

        content = (
            "**1. Cross-border summary**\n"
            "T1135 filing with box 38 amounts of $12,000 and Form 106 details. " * 12
        )
        score, failure, _ = _validate_result(
            {"content": content}, TASKS["taxes_synthesis"], "taxes_synthesis"
        )
        assert score > 0, failure
        assert "grounding=" in failure, "expected the rubric's own reason string"

    def test_validators_score_the_real_snapshot_text(self):
        from eval.tasks_core import TASKS
        from lib.validators.taxes_validator import validate_taxes_synthesis

        prompt = TASKS["taxes_synthesis"]["messages"][1]["content"]
        # Echoing the prompt back is not a valid answer: it leaks GT-flavored
        # terms or misses the required sections, so it must not score full marks.
        score, _ = validate_taxes_synthesis(prompt)
        assert score < 100
