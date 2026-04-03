from pathlib import Path

from storygen.parser import parse_story_file


def test_parse_story_file_extracts_scene_and_entity_metadata() -> None:
    story = parse_story_file(Path("test_set/01.txt"))

    assert len(story.scenes) == 3
    assert story.scenes[0].scene_id == "SCENE-1"
    assert story.scenes[0].raw_text == "<Lily> makes breakfast in the kitchen."
    assert story.scenes[0].clean_text == "Lily makes breakfast in the kitchen."
    assert story.scenes[0].entities == ["Lily"]
    assert story.recurring_entities == []
    assert story.entity_to_scene_ids == {"Lily": ["SCENE-1"]}
