import pytest

from eval_tasks_core import _extract_items_from_text


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
        assert len(items) == 1
        assert items[0].get("name") is not None

    def test_bullet_with_separator(self):
        text = "- Park - Toronto\n- Zoo - Vaughan"
        items = _extract_items_from_text(text)
        assert items[0]["name"] == "Park"
        assert items[0]["location"] == "Toronto"

    def test_simple_bullets_without_detail(self):
        text = "- just a line\n- another line"
        items = _extract_items_from_text(text)
        assert len(items) >= 2

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
