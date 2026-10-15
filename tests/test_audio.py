import json

from modules.audio.audio_understanding import (
    clean_transcript,
    extract_requirements,
    save_requirements,
)


def test_clean_transcript():

    text = "  Create   an advertisement   for students.  "

    cleaned = clean_transcript(text)

    assert cleaned == (
        "Create an advertisement for students."
    )


def test_requirement_extraction():

    transcript = (
        "Create an energetic Instagram advertisement "
        "for college students. Make it 15 seconds long "
        "in English. Include a discount and subtitles."
    )

    result = extract_requirements(
        transcript=transcript,
        detected_language="en"
    )

    assert result["audience"] == "college students"

    assert result["tone"] == "energetic"

    assert result["platform"] == "Instagram"

    assert result["duration_seconds"] == 15

    assert result["language"] == "en"

    assert "discount" in result["extra_requirements"]

    assert "subtitles" in result["extra_requirements"]


def test_save_requirements(tmp_path):

    requirements = {
        "transcript": "Create an advertisement.",
        "audience": "general audience",
        "tone": "professional",
        "platform": "Instagram",
        "duration_seconds": 15,
        "language": "en",
        "extra_requirements": [],
    }

    output_file = tmp_path / "requirements.json"

    saved_path = save_requirements(
        requirements,
        str(output_file)
    )

    assert saved_path.exists()

    with saved_path.open(
        encoding="utf-8"
    ) as file:

        saved_data = json.load(file)

    assert saved_data == requirements