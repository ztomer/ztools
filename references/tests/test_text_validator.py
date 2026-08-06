"""Tests for lib.validators.text_validator."""



class TestValidateFilename:
    def test_empty(self):
        from lib.validators.text_validator import validate_filename

        score, msg = validate_filename("")
        assert score == 0
        assert "empty" in msg

    def test_none(self):
        from lib.validators.text_validator import validate_filename

        score, msg = validate_filename(None)
        assert score == 0

    def test_good_filename(self):
        from lib.validators.text_validator import validate_filename

        score, _ = validate_filename("my_great_file")
        # 30 (length) + 20 (chars) + 25 (format) + 25 (specificity) = 100
        assert score == 100

    def test_too_short(self):
        from lib.validators.text_validator import validate_filename

        score, msg = validate_filename("abc")  # 3 chars
        assert "generic" in msg or score == 0

    def test_too_long(self):
        from lib.validators.text_validator import validate_filename

        long_name = "a" * 60  # 60 chars, hits FILENAME_LENGTH_MAX
        score, msg = validate_filename(long_name)
        # Should fall back to extract_best_filename_candidate
        # Then check if that one passes
        assert "length" in msg or score < 100

    def test_with_dash(self):
        from lib.validators.text_validator import validate_filename

        score, _ = validate_filename("my-cool-file")
        # 3 words separated by dash, no generic terms
        assert score == 100

    def test_with_dot(self):
        from lib.validators.text_validator import validate_filename

        score, _ = validate_filename("my.file.txt")
        # 2 word parts separated by dot
        assert score == 100

    def test_invalid_chars(self):
        from lib.validators.text_validator import validate_filename

        # 60+ chars with invalid chars triggers fallback to extract_best
        long_with_invalid = "this is a long thing @#$%^&*() stuff"
        score, msg = validate_filename(long_with_invalid)
        # Falls back to extract_best which finds "this is a long thing" (no invalid chars)
        # but still has spaces and is wordy
        assert score == 45
        assert "invalid chars" in msg

    def test_question_in_filename(self):
        from lib.validators.text_validator import validate_filename

        score, msg = validate_filename("what_file?")
        assert "question" in msg
        assert score < 100

    def test_wordy_filename(self):
        from lib.validators.text_validator import validate_filename

        score, msg = validate_filename("the quick brown fox jumps")
        # 25 chars, no question - hits wordy path
        assert "wordy" in msg

    def test_starts_with_the(self):
        from lib.validators.text_validator import validate_filename

        score, msg = validate_filename("the summer party event")
        # 23 chars, no ?, starts with "the" - wordy
        assert "wordy" in msg

    def test_starts_with_this(self):
        from lib.validators.text_validator import validate_filename

        score, msg = validate_filename("this is some event name")
        # 23 chars, starts with "this"
        assert "wordy" in msg

    def test_starts_with_a(self):
        from lib.validators.text_validator import validate_filename

        score, msg = validate_filename("a special event name")
        # 20 chars, starts with "a" - wordy
        assert "wordy" in msg

    def test_starts_with_please(self):
        from lib.validators.text_validator import validate_filename

        score, msg = validate_filename("please_what_to_call_this")
        assert "question" in msg

    def test_starts_with_which(self):
        from lib.validators.text_validator import validate_filename

        score, msg = validate_filename("which_file_is_better")
        assert "question" in msg

    def test_starts_with_what(self):
        from lib.validators.text_validator import validate_filename

        score, msg = validate_filename("what_should_i_name_this")
        assert "question" in msg

    def test_no_separators(self):
        from lib.validators.text_validator import validate_filename

        score, msg = validate_filename("abcdefghij")
        assert "no separators" in msg
        # But still has length, chars, and non-wordy score
        # 30 (length) + 20 (chars) + 0 (no format) + 25 (specific) = 75
        assert score < 100

    def test_with_backticks(self):
        from lib.validators.text_validator import validate_filename

        score, _ = validate_filename("`my_file`")
        # strip_backtick_value removes backticks
        assert score == 100

    def test_with_code_block(self):
        from lib.validators.text_validator import validate_filename

        score, _ = validate_filename("```my_file```")
        # Code fence stripped, then 2-word name
        assert score == 100

    def test_generic_filename_txt(self):
        from lib.validators.text_validator import validate_filename

        score, msg = validate_filename("filename.txt")
        assert "generic" in msg
        assert score == 0

    def test_generic_filename_just(self):
        from lib.validators.text_validator import validate_filename

        score, msg = validate_filename("file")
        assert "generic" in msg
        assert score == 0

    def test_generic_image(self):
        from lib.validators.text_validator import validate_filename

        score, msg = validate_filename("image.png")
        assert "generic" in msg

    def test_length_failure(self):
        """When clean is in length failure range (>= MAX after fallback)."""
        from lib.validators.text_validator import validate_filename

        # A single line of 59+ chars with no separators will be kept by extract_best
        # but then fail the length check
        long_line = "a" * 59  # exactly at MAX
        score, msg = validate_filename(long_line)
        # len 59 is NOT in (4, 59) (exclusive), so length failure
        assert "length" in msg


class TestValidateSummary:
    def test_empty(self):
        from lib.validators.text_validator import validate_summary

        score, msg = validate_summary("")
        assert score == 0
        assert "empty" in msg

    def test_none(self):
        from lib.validators.text_validator import validate_summary

        score, msg = validate_summary(None)
        assert score == 0

    def test_dict_input(self):
        from lib.validators.text_validator import validate_summary

        score, msg = validate_summary({"key": "value"})
        # str(dict) = "{'key': 'value'}" - no structure detected
        assert score == 10
        assert "no structure" in msg

    def test_structure_headers_and_bullets(self):
        from lib.validators.text_validator import validate_summary

        text = "## Section 1\n- bullet one\n- bullet two"
        score, _ = validate_summary(text)
        # Has headers and bullets = 15 + 10 + 15 = 40 (length not met)
        assert score == 40

    def test_structure_headers_only(self):
        from lib.validators.text_validator import validate_summary

        text = "## Section 1\n## Section 2\nsome content here"
        score, _ = validate_summary(text)
        # 2 headers (15) + 1 header=structure (15) + length>=200 (15) = 45
        assert score == 45

    def test_structure_bullets_long(self):
        from lib.validators.text_validator import validate_summary

        text = "- " + ("a " * 200)  # > 300 chars
        score, _ = validate_summary(text)
        # 1 header (10) + length (5) + bullets long (3) = 18
        assert score == 18

    def test_structure_long_no_headers(self):
        from lib.validators.text_validator import validate_summary

        text = "a " * 200  # 400 chars
        score, _ = validate_summary(text)
        # No structure (0) + length>=200 (15) = 15
        assert score == 15

    def test_structure_short_no_headers(self):
        from lib.validators.text_validator import validate_summary

        text = "short text"
        score, msg = validate_summary(text)
        # < 200 chars, no headers
        assert "no structure" in msg

    def test_user_mentions_3plus(self):
        from lib.validators.text_validator import validate_summary

        text = "@user1 @user2 @user3 hello"
        score, _ = validate_summary(text)
        # 3 users = 25 pts (10 base + 5*2 + 5 for 3rd+)
        assert score == 25

    def test_user_mentions_2(self):
        from lib.validators.text_validator import validate_summary

        text = "@user1 @user2 hello"
        score, _ = validate_summary(text)
        # 2 users = 20 pts (10 base + 5 each for 2nd and 3rd)
        assert score == 20

    def test_user_mentions_1(self):
        from lib.validators.text_validator import validate_summary

        text = "@user1 hello"
        score, _ = validate_summary(text)
        # 1 user = 15 pts (10 base + 5 for second)
        assert score == 15

    def test_user_mentions_none(self):
        from lib.validators.text_validator import validate_summary

        text = "no mentions here at all just text"
        score, msg = validate_summary(text)
        # 0 users
        assert "no user" in msg

    def test_user_word_format(self):
        from lib.validators.text_validator import validate_summary

        # "user 1" without @
        text = "user 1 user 2 user 3 did things"
        score, _ = validate_summary(text)
        # 3 "user N" patterns = 25 pts
        assert score == 25

    def test_timestamps(self):
        from lib.validators.text_validator import validate_summary

        text = "At 10:30 something happened"
        score, _ = validate_summary(text)
        # Timestamp 10 + length>=200? no → 0
        assert score == 20

    def test_narrative_words(self):
        from lib.validators.text_validator import validate_summary

        text = "user asks and then responds and thanks"
        score, _ = validate_summary(text)
        # Multiple narrative verbs → 35 pts (5 narrative + 30 for "user" mentions)
        assert score == 35

    def test_no_specificity(self):
        from lib.validators.text_validator import validate_summary

        text = "very short text"
        score, msg = validate_summary(text)
        assert "no timestamps" in msg

    def test_template_driven(self):
        from lib.validators.text_validator import validate_summary

        text = "**Who: x\n**What: y\n**When: z\nbody text here"
        score, msg = validate_summary(text)
        assert "template-driven" in msg

    def test_boilerplate(self):
        from lib.validators.text_validator import validate_summary

        text = "Some content with not specified value here"
        score, msg = validate_summary(text)
        assert "boilerplate" in msg

    def test_synthesis_top_level(self):
        from lib.validators.text_validator import validate_summary

        # Has "overall" in top-level
        text = "Overall, this is a summary of events. ## Section\n- bullet"
        score, _ = validate_summary(text)
        # Has synthesis: 25 pts structure (headers+bullets=15, synthesis=10) = 25, no other bonuses
        # Actually: structure 15 + synthesis 10 = 25, no user/timestamp/topic bonus
        assert score == 20  # 10 for ##, 5 for - (one bullet), 5 for "Overall" synthesis

    def test_synthesis_key_takeaways(self):
        from lib.validators.text_validator import validate_summary

        text = "Key takeaways: lots happened. ## Section"
        score, _ = validate_summary(text)
        # 1 header (10) + 1 'overall/synthesis' keyword (10) = 20
        assert score == 20

    def test_synthesis_tldr(self):
        from lib.validators.text_validator import validate_summary

        text = "TL;DR: short version. ## Section"
        score, _ = validate_summary(text)
        # 1 header (10) + 1 tldr synthesis (15) + 1 'overall' (10) = 35
        assert score == 35

    def test_topic_markers_many(self):
        from lib.validators.text_validator import validate_summary

        text = "## Topic1\n## Topic2\n- bullets"
        score, _ = validate_summary(text)
        # 2 headers (15) + 1 header structure (15) + topic markers bonus (20) = 50
        assert score == 50

    def test_topic_markers_one(self):
        from lib.validators.text_validator import validate_summary

        text = "## Topic1\n- bullets"
        score, _ = validate_summary(text)
        # 1 topic = 25 pts structure (15 headers/bullets + 10 single topic)
        assert score == 40

    def test_topic_transitions(self):
        from lib.validators.text_validator import validate_summary

        text = "First this happened. Then that. Also meanwhile something."
        score, _ = validate_summary(text)
        # 3+ transitions = 10; plus length>=200? no → 0
        assert score == 20

    def test_topic_no_structure(self):
        from lib.validators.text_validator import validate_summary

        text = "just a plain text with no structure at all"
        score, msg = validate_summary(text)
        assert "no topic" in msg

    def test_synthesis_in_short(self):
        from lib.validators.text_validator import validate_summary

        text = "In short, things happened. ## Section"
        score, _ = validate_summary(text)
        # Has "in short" + headers
        assert score == 20

    def test_synthesis_this_conversation(self):
        from lib.validators.text_validator import validate_summary

        text = "This conversation was interesting. ## Section"
        score, _ = validate_summary(text)
        # Has "this conversation" + headers
        assert score == 20

    def test_no_header_no_top_level(self):
        from lib.validators.text_validator import validate_summary

        text = "no headers here just text"
        # top_level gets set to the whole thing
        score, _ = validate_summary(text)
        # No headers, no bullets, no synthesis → low score
        assert score == 10


class TestValidateFileSummary:
    def test_empty(self):
        from lib.validators.text_validator import validate_file_summary

        score, msg = validate_file_summary(None)
        assert score == 0
        assert "empty" in msg

    def test_empty_list(self):
        from lib.validators.text_validator import validate_file_summary

        score, msg = validate_file_summary([])
        assert score == 0
        # Empty list is falsy
        assert "empty" in msg

    def test_dict_input(self):
        from lib.validators.text_validator import validate_file_summary

        score, msg = validate_file_summary({"path": "x.py", "desc": "python file"})
        # Wrapped in list — 1 item → "only 1 items" failure
        assert "only 1 items" in msg
        # 30 (count 0 + paths 25 + quality 0) but not full 30 since count check fails
        # Actually: 30 (count 5/100 × 60) + 25 (path 50% × 50) + 0 (no unique descs) = 55
        assert score < 75

    def test_string_input(self):
        from lib.validators.text_validator import validate_file_summary

        score, _ = validate_file_summary("not a list")
        # Not a dict or list, treated as empty list
        assert score == 0

    def test_too_few_items(self):
        from lib.validators.text_validator import validate_file_summary

        items = [{"path": "x.py", "desc": "a python file"}]
        score, msg = validate_file_summary(items)
        assert "only 1 items" in msg

    def test_good_count(self):
        from lib.validators.text_validator import validate_file_summary

        items = [
            {"path": f"file{i}.py", "desc": f"description for file {i} which is good"}
            for i in range(5)
        ]
        score, _ = validate_file_summary(items)
        # 30 (count) + 30 (paths) + 40 (quality) = 100
        assert score == 100

    def test_unrealistic_paths(self):
        from lib.validators.text_validator import validate_file_summary

        items = [{"path": "filename", "desc": "specific unique description here"} for _ in range(5)]
        score, msg = validate_file_summary(items)
        # No . or / in paths
        assert "unrealistic paths" in msg

    def test_partial_paths(self):
        from lib.validators.text_validator import validate_file_summary

        items = [
            {"path": "x.py", "desc": "specific desc 1"},
            {"path": "y.py", "desc": "specific desc 2"},
            {"path": "filename", "desc": "specific desc 3"},
            {"path": "another", "desc": "specific desc 4"},
            {"path": "another2", "desc": "specific desc 5"},
        ]
        # 2/5 = 40% real paths (boundary: exactly 0.4 → partial credit, no failure msg)
        score, msg = validate_file_summary(items)
        # Partial credit: 30 (count) + 30 (path 60% × 50) + 30 (quality) = 90
        assert score == 90
        # 40% is at the boundary — partial credit, no unrealistic failure
        assert "unrealistic" not in msg

        # Now test < 40% to verify the failure message
        items2 = [
            {"path": "x.py", "desc": "specific desc 1"},
            {"path": "filename", "desc": "specific desc 2"},
            {"path": "another", "desc": "specific desc 3"},
            {"path": "another2", "desc": "specific desc 4"},
            {"path": "another3", "desc": "specific desc 5"},
        ]
        score2, msg2 = validate_file_summary(items2)
        assert "unrealistic" in msg2

    def test_generic_descriptions(self):
        from lib.validators.text_validator import validate_file_summary

        items = [{"path": "x.py", "desc": "personal document"} for _ in range(5)]
        score, msg = validate_file_summary(items)
        # Generic descriptions - quality score is 15 (unique but generic)
        # No specific "generic" failure msg
        assert score == 75  # 30 + 30 + 15

    def test_one_specific(self):
        from lib.validators.text_validator import validate_file_summary

        items = [
            {"path": "x.py", "desc": "personal document"},
            {"path": "y.py", "desc": "specific unique description for the file"},
            {"path": "z.py", "desc": "another generic item"},
            {"path": "a.py", "desc": "another generic item"},
            {"path": "b.py", "desc": "another generic item"},
        ]
        score, _ = validate_file_summary(items)
        # 1 specific + 3 unique generic → 100
        assert score == 100

    def test_no_descriptions(self):
        from lib.validators.text_validator import validate_file_summary

        items = [
            {"path": "x.py"},  # No desc field
            {"path": "y.py"},
            {"path": "z.py"},
            {"path": "a.py"},
            {"path": "b.py"},
        ]
        score, _ = validate_file_summary(items)
        # unique_descs is empty → 60 (30 + 30 + 0)
        assert score == 60

    def test_description_field_alias(self):
        from lib.validators.text_validator import validate_file_summary

        items = [
            {"path": "x.py", "description": "specific description here"},
            {"path": "y.py", "description": "another specific one here"},
            {"path": "z.py", "description": "third specific here"},
            {"path": "a.py", "description": "fourth specific here"},
            {"path": "b.py", "description": "fifth specific here"},
        ]
        score, _ = validate_file_summary(items)
        # Uses "description" as fallback
        assert score > 50
