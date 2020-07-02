"""Rhasspy utility functions."""
import io
import logging
import re
import subprocess
import typing
import wave
from pathlib import Path

import rhasspynlu

WHITESPACE_PATTERN = re.compile(r"\s+")
_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class FunctionLoggingHandler(logging.Handler):
    """Calls a function for each logging message."""

    def __init__(
        self,
        func,
        log_format: str = "[%(levelname)s:%(asctime)s] %(name)s: %(message)s",
    ):
        logging.Handler.__init__(self)
        self.func = func
        self.formatter = logging.Formatter(log_format)

    def handle(self, record):
        self.func(self.formatter.format(record))


# -----------------------------------------------------------------------------


def read_dict(
    dict_file: typing.Iterable[str],
    word_dict: typing.Optional[typing.Dict[str, typing.List[str]]] = None,
    transform: typing.Optional[typing.Callable[[str], str]] = None,
    silence_words: typing.Optional[typing.Set[str]] = None,
) -> typing.Dict[str, typing.List[str]]:
    """
    Loads a CMU/Julius word dictionary, optionally into an existing Python dictionary.
    """
    if word_dict is None:
        word_dict = {}

    for i, line in enumerate(dict_file):
        line = line.strip()
        if not line:
            continue

        try:
            # Use explicit whitespace (avoid 0xA0)
            word, *parts = re.split(r"[ \t]+", line)

            # Skip Julius extras
            pronounce = " ".join(p for p in parts if p[0] not in {"[", "@"})

            word = word.split("(")[0]
            # Julius format word1+word2
            words = word.split("+")

            for word in words:
                # Don't transform silence words
                if transform and (
                    (silence_words is None) or (word not in silence_words)
                ):
                    word = transform(word)

                if word in word_dict:
                    word_dict[word].append(pronounce)
                else:
                    word_dict[word] = [pronounce]
        except Exception as e:
            _LOGGER.warning("read_dict: %s (line %s)", e, i + 1)

    return word_dict


# -----------------------------------------------------------------------------


def recursive_remove(
    base_dict: typing.Dict[typing.Any, typing.Any],
    new_dict: typing.Dict[typing.Any, typing.Any],
) -> None:
    """Recursively removes values from new dictionary that are already in base dictionary"""
    for k, v in list(new_dict.items()):
        if k in base_dict:
            if isinstance(v, dict):
                recursive_remove(base_dict[k], v)
                if not v:
                    del new_dict[k]
            elif v == base_dict[k]:
                del new_dict[k]


# -----------------------------------------------------------------------------


def buffer_to_wav(buffer: bytes) -> bytes:
    """Wraps a buffer of raw audio data (16-bit, 16Khz mono) in a WAV"""
    with io.BytesIO() as wav_buffer:
        wav_file: wave.Wave_write = wave.open(wav_buffer, mode="wb")
        with wav_file:
            wav_file.setframerate(16000)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
            wav_file.writeframes(buffer)

        return wav_buffer.getvalue()


def get_wav_duration(wav_bytes: bytes) -> float:
    """Return the real-time duration of a WAV file"""
    with io.BytesIO(wav_bytes) as wav_buffer:
        wav_file: wave.Wave_read = wave.open(wav_buffer, "rb")
        with wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate)


def wav_to_buffer(wav_bytes: bytes) -> bytes:
    """Return the raw audio of a WAV file"""
    with io.BytesIO(wav_bytes) as wav_buffer:
        wav_file: wave.Wave_read = wave.open(wav_buffer, "rb")
        with wav_file:
            frames = wav_file.getnframes()
            return wav_file.readframes(frames)


# -----------------------------------------------------------------------------


def load_phoneme_examples(
    path: typing.Union[str, Path]
) -> typing.Dict[str, typing.Dict[str, str]]:
    """Loads example words and pronunciations for each phoneme."""
    examples = {}
    with open(path, "r") as example_file:
        for line in example_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip blanks and comments

            parts = split_whitespace(line)
            examples[parts[0]] = {"word": parts[1], "phonemes": " ".join(parts[2:])}

    return examples


def load_phoneme_map(path: typing.Union[str, Path]) -> typing.Dict[str, str]:
    """Load phoneme map from CMU (Sphinx) phonemes to eSpeak phonemes."""
    phonemes = {}
    with open(path, "r") as phoneme_file:
        for line in phoneme_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip blanks and comments

            parts = split_whitespace(line, maxsplit=1)
            phonemes[parts[0]] = parts[1]

    return phonemes


# -----------------------------------------------------------------------------


def get_ini_paths(
    sentences_ini: Path, sentences_dir: typing.Optional[Path] = None
) -> typing.List[Path]:
    """Get paths to all .ini files in profile."""
    ini_paths: typing.List[Path] = []
    if sentences_ini.is_file():
        ini_paths = [sentences_ini]

    # Add .ini files from intents directory
    if sentences_dir and sentences_dir.is_dir():
        ini_paths.extend(sentences_dir.rglob("*.ini"))

    return ini_paths


def get_all_intents(ini_paths: typing.List[Path]) -> typing.Dict[str, typing.Any]:
    """Get intents from all .ini files in profile."""
    try:
        with io.StringIO() as combined_ini_file:
            for ini_path in ini_paths:
                combined_ini_file.write(ini_path.read_text())
                print("", file=combined_ini_file)

            return rhasspynlu.parse_ini(combined_ini_file.getvalue())
    except Exception:
        _LOGGER.exception("Failed to parse %s", ini_paths)

    return {}


# -----------------------------------------------------------------------------


def split_whitespace(s: str, **kwargs):
    """Split a string by whitespace of any type/length."""
    return WHITESPACE_PATTERN.split(s, **kwargs)


# -----------------------------------------------------------------------------


def get_espeak_wav(word: str, voice: typing.Optional[str] = None) -> bytes:
    """Get eSpeak WAV pronunciation for a word."""
    try:
        espeak_command = ["espeak", "--stdout", "-s", "80"]

        if voice:
            espeak_command.extend(["-v", str(voice)])

        espeak_command.append(word)

        _LOGGER.debug(espeak_command)
        return subprocess.check_output(espeak_command)
    except Exception:
        _LOGGER.exception("get_espeak_wav")

    return bytes()


def get_espeak_phonemes(word: str) -> str:
    """Get eSpeak phonemes for a word."""
    try:
        espeak_command = ["espeak", "-q", "-x", word]

        _LOGGER.debug(espeak_command)
        return subprocess.check_output(espeak_command, universal_newlines=True).strip()
    except Exception:
        _LOGGER.exception("get_espeak_phonemes")

    return ""
