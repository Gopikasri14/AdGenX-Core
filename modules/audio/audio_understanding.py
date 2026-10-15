import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import whisper


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "requirements.json"


# ---------------------------------------------------------
# Whisper
# ---------------------------------------------------------

def load_whisper_model(model_name: str = "base"):
    """
    Load the local Whisper speech-to-text model.

    Available models include:
        tiny, base, small, medium, large
    """

    print(f"Loading Whisper model: {model_name}")

    model = whisper.load_model(model_name)

    print("Whisper model loaded successfully.")

    return model


# ---------------------------------------------------------
# Audio transcription
# ---------------------------------------------------------

def transcribe_audio(
    audio_path: str,
    model,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Convert an audio file into text using Whisper.

    Returns:
        transcript
        detected language
        complete Whisper result
    """

    audio_file = Path(audio_path)

    if not audio_file.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_file}"
        )

    if not audio_file.is_file():
        raise ValueError(
            f"Audio path is not a file: {audio_file}"
        )

    print(f"Transcribing: {audio_file}")

    result = model.transcribe(
        str(audio_file),
        fp16=False,
        verbose=False
    )

    transcript = result.get("text", "").strip()

    detected_language = result.get(
        "language",
        "unknown"
    )

    return transcript, detected_language, result


# ---------------------------------------------------------
# Transcript cleaning
# ---------------------------------------------------------

def clean_transcript(text: str) -> str:
    """
    Clean Whisper's transcript without changing its meaning.
    """

    if not text:
        return ""

    # Replace multiple spaces/newlines with one space
    text = re.sub(r"\s+", " ", text)

    # Remove unnecessary spaces before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


# ---------------------------------------------------------
# Requirement extraction helpers
# ---------------------------------------------------------

def _extract_audience(text: str) -> str:
    """
    Extract the intended advertisement audience.
    """

    patterns = [
        r"\bfor\s+(college students)\b",
        r"\bfor\s+(students)\b",
        r"\bfor\s+(teenagers)\b",
        r"\bfor\s+(young adults)\b",
        r"\bfor\s+(professionals)\b",
        r"\bfor\s+(working professionals)\b",
        r"\bfor\s+(children)\b",
        r"\bfor\s+(kids)\b",
        r"\bfor\s+(parents)\b",
        r"\bfor\s+(women)\b",
        r"\bfor\s+(men)\b",
        r"\btarget(?:ed)?\s+(?:at|towards)\s+([^.,!?]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1).strip()

    return "general audience"


def _extract_tone(text: str) -> str:
    """
    Extract advertisement tone.
    """

    tones = [
        "energetic",
        "professional",
        "fun",
        "funny",
        "humorous",
        "emotional",
        "inspiring",
        "inspirational",
        "friendly",
        "exciting",
        "premium",
        "luxurious",
        "dramatic",
        "calm",
        "modern",
        "bold",
    ]

    lower_text = text.lower()

    for tone in tones:
        if tone in lower_text:
            return tone

    return "professional"


def _extract_platform(text: str) -> str:
    """
    Extract the target advertising platform.
    """

    platforms = {
        "instagram": "Instagram",
        "youtube": "YouTube",
        "facebook": "Facebook",
        "linkedin": "LinkedIn",
        "tiktok": "TikTok",
        "twitter": "Twitter",
        "x": "X",
        "whatsapp": "WhatsApp",
        "website": "Website",
    }

    lower_text = text.lower()

    for keyword, platform in platforms.items():

        if re.search(
            rf"\b{re.escape(keyword)}\b",
            lower_text
        ):
            return platform

    return "Instagram"


def _extract_duration(text: str) -> int:
    """
    Extract advertisement duration in seconds.
    """

    patterns = [
        r"(\d+)\s*seconds?",
        r"(\d+)\s*secs?",
        r"(\d+)\s*second",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return int(match.group(1))

    # Project specification defaults to 15 seconds
    return 15


def _extract_language(
    text: str,
    detected_language: str
) -> str:
    """
    Extract language requirement.

    Prefer an explicit language mentioned in the transcript.
    Otherwise use Whisper's detected language.
    """

    language_map = {
        "english": "en",
        "tamil": "ta",
        "hindi": "hi",
        "telugu": "te",
        "malayalam": "ml",
        "kannada": "kn",
        "french": "fr",
        "german": "de",
        "spanish": "es",
        "japanese": "ja",
        "korean": "ko",
    }

    lower_text = text.lower()

    for language_name, language_code in language_map.items():

        if re.search(
            rf"\b{re.escape(language_name)}\b",
            lower_text
        ):
            return language_code

    if detected_language:
        return detected_language

    return "unknown"


def _extract_extra_requirements(
    text: str
) -> list:
    """
    Extract additional advertisement requirements.
    """

    requirements = []

    lower_text = text.lower()

    requirement_keywords = {
        "discount": "discount",
        "offer": "special offer",
        "coupon": "coupon",
        "subtitles": "subtitles",
        "subtitle": "subtitles",
        "call to action": "call to action",
        "cta": "call to action",
        "logo": "product logo",
        "music": "background music",
        "background music": "background music",
        "contact details": "contact details",
        "website": "website details",
        "phone number": "phone number",
    }

    for keyword, requirement in requirement_keywords.items():

        if keyword in lower_text:
            if requirement not in requirements:
                requirements.append(requirement)

    return requirements


# ---------------------------------------------------------
# Main requirement extraction
# ---------------------------------------------------------

def extract_requirements(
    transcript: str,
    detected_language: str = "en"
) -> Dict[str, Any]:
    """
    Convert a cleaned advertisement instruction into
    the JSON contract expected by Member 3.
    """

    cleaned_transcript = clean_transcript(transcript)

    requirements = {
        "transcript": cleaned_transcript,
        "audience": _extract_audience(
            cleaned_transcript
        ),
        "tone": _extract_tone(
            cleaned_transcript
        ),
        "platform": _extract_platform(
            cleaned_transcript
        ),
        "duration_seconds": _extract_duration(
            cleaned_transcript
        ),
        "language": _extract_language(
            cleaned_transcript,
            detected_language
        ),
        "extra_requirements": _extract_extra_requirements(
            cleaned_transcript
        ),
    }

    return requirements


# ---------------------------------------------------------
# Save requirements
# ---------------------------------------------------------

def save_requirements(
    requirements: Dict[str, Any],
    output_path: Optional[str] = None
) -> Path:
    """
    Save Member 2 output as JSON.
    """

    if output_path is None:
        output_file = DEFAULT_OUTPUT
    else:
        output_file = Path(output_path)

        if not output_file.is_absolute():
            output_file = PROJECT_ROOT / output_file

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            requirements,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"Requirements saved to: {output_file}")

    return output_file


# ---------------------------------------------------------
# Complete audio pipeline
# ---------------------------------------------------------

def process_audio(
    audio_path: str,
    output_path: Optional[str] = None,
    model_name: str = "base"
) -> Dict[str, Any]:
    """
    Complete Member 2 pipeline:

        Audio
          ↓
        Whisper
          ↓
        Transcript cleaning
          ↓
        Requirement extraction
          ↓
        requirements.json
    """

    model = load_whisper_model(model_name)

    transcript, detected_language, _ = transcribe_audio(
        audio_path,
        model
    )

    print("\nTranscript:")
    print(transcript)

    requirements = extract_requirements(
        transcript,
        detected_language
    )

    save_requirements(
        requirements,
        output_path
    )

    return requirements


# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AdGenX Member 2 - Audio Understanding"
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="Path to the input audio file"
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path for requirements JSON"
    )

    parser.add_argument(
        "--model",
        default="base",
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large"
        ],
        help="Whisper model size"
    )

    args = parser.parse_args()

    requirements = process_audio(
        audio_path=args.audio,
        output_path=args.output,
        model_name=args.model
    )

    print("\nFinal requirements:")
    print(
        json.dumps(
            requirements,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()