"""Rhasspy web application server."""
import argparse
import asyncio
import atexit
import json
import logging
import os
import re
import subprocess
import sys
import time
import typing
from collections import defaultdict
from functools import wraps
from pathlib import Path
from uuid import uuid4

import attr
import jinja2
import rhasspyprofile
import rhasspysupervisor
from paho.mqtt.matcher import MQTTMatcher
from quart import (
    Quart,
    Response,
    jsonify,
    redirect,
    request,
    render_template,
    safe_join,
    send_file,
    send_from_directory,
    websocket,
)
from quart_cors import cors
from rhasspyhermes.asr import AsrStartListening, AsrStopListening, AsrTextCaptured
from rhasspyhermes.handle import HandleToggleOff, HandleToggleOn
from rhasspyhermes.nlu import NluIntent, NluIntentNotRecognized
from rhasspyhermes.wake import HotwordDetected, HotwordToggleOff, HotwordToggleOn
from rhasspyprofile import Profile
from swagger_ui import quart_api_doc

from . import RhasspyCore
from .utils import (
    FunctionLoggingHandler,
    buffer_to_wav,
    get_all_intents,
    get_espeak_phonemes,
    get_espeak_wav,
    get_ini_paths,
    get_wav_duration,
    load_phoneme_examples,
    read_dict,
    recursive_remove,
)

_LOGGER = logging.getLogger(__name__)
loop = asyncio.get_event_loop()

# -----------------------------------------------------------------------------
# Parse Arguments
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser("Rhasspy")
parser.add_argument(
    "--profile", "-p", required=True, type=str, help="Name of profile to load"
)
parser.add_argument("--host", type=str, help="Host for web server", default="0.0.0.0")
parser.add_argument("--port", type=int, help="Port for web server", default=12101)
parser.add_argument("--mqtt-host", type=str, help="Host for MQTT broker", default=None)
parser.add_argument("--mqtt-port", type=int, help="Port for MQTT broker", default=None)
parser.add_argument(
    "--local-mqtt-port",
    type=int,
    default=12183,
    help="Port to use for internal MQTT broker (default: 12183)",
)
parser.add_argument(
    "--system-profiles", type=str, help="Directory with base profile files (read only)"
)
parser.add_argument(
    "--user-profiles", type=str, help="Directory with user profile files (read/write)"
)
parser.add_argument(
    "--set",
    "-s",
    nargs=2,
    action="append",
    help="Set a profile setting value",
    default=[],
)
parser.add_argument(
    "--ssl", nargs=2, help="Use SSL with <CERT_FILE <KEY_FILE>", default=None
)
parser.add_argument("--log-level", default="DEBUG", help="Set logging level")
parser.add_argument(
    "--web-dir", default="web", help="Directory with compiled Vue site (default: web)"
)

args = parser.parse_args()

# Set log level
log_level = getattr(logging, args.log_level.upper())
logging.basicConfig(level=log_level)


_LOGGER.debug(args)

# Convert profile directories to Paths
if args.system_profiles:
    system_profiles_dir = Path(args.system_profiles)
else:
    # Bundled
    system_profiles_dir = None

if args.user_profiles:
    user_profiles_dir = Path(args.user_profiles)
else:
    user_profiles_dir = Path("~/.config/rhasspy/profiles").expanduser()

# -----------------------------------------------------------------------------
# Quart Web App Setup
# -----------------------------------------------------------------------------

web_dir = Path(args.web_dir)
assert web_dir.is_dir(), f"Missing web directory {web_dir}"
template_dir = web_dir.parent / "templates"

app = Quart("rhasspy", template_folder=template_dir)
app.secret_key = str(uuid4())
app = cors(app)

# -----------------------------------------------------------------------------
# Template Functions
# -----------------------------------------------------------------------------


def get_version() -> str:
    return Path("VERSION").read_text().strip()


def get_template_args():
    return {
        "profile": core.profile,
        "profile_json": json.dumps(core.profile.json, indent=4),
        "version": get_version(),
    }


def save_profile(profile_json: typing.Dict[str, typing.Any]) -> str:
    # Ensure that JSON is valid
    recursive_remove(core.profile.system_json, profile_json)

    profile_path = Path(core.profile.write_path("profile.json"))
    with open(profile_path, "w") as profile_file:
        json.dump(profile_json, profile_file, indent=4)

    msg = f"Wrote profile to {str(profile_path)}"
    _LOGGER.debug(msg)

    # Re-generate supevisord config
    new_profile = Profile(args.profile, system_profiles_dir, user_profiles_dir)
    supervisor_conf_path = new_profile.write_path("supervisord.conf")
    _LOGGER.debug("Re-generating %s", str(supervisor_conf_path))

    with open(supervisor_conf_path, "w") as supervisor_conf_file:
        rhasspysupervisor.profile_to_conf(
            new_profile, supervisor_conf_file, local_mqtt_port=args.local_mqtt_port
        )

    # Re-genenerate Docker compose YAML
    docker_compose_path = new_profile.write_path("docker-compose.yml")
    _LOGGER.debug("Re-generating %s", str(docker_compose_path))

    with open(docker_compose_path, "w") as docker_compose_file:
        rhasspysupervisor.profile_to_docker(
            new_profile, docker_compose_file, local_mqtt_port=args.local_mqtt_port
        )

    return msg


def read_sentences(sentences_path: Path, sentences_dir: Path):
    sentences_dict = {}
    if sentences_path.is_file():
        try:
            # Try user profile dir first
            profile_dir = Path(core.profile.user_profiles_dir) / core.profile.name
            key = str(sentences_path.relative_to(profile_dir))
        except Exception:
            # Fall back to system profile dir
            profile_dir = Path(core.profile.system_profiles_dir) / core.profile.name
            key = str(sentences_path.relative_to(profile_dir))

        sentences_dict[key] = sentences_path.read_text()

    # Add all .ini files from sentences_dir
    if sentences_dir.is_dir():
        for ini_path in sentences_dir.glob("*.ini"):
            key = str(ini_path.relative_to(core.profile.read_path()))
            sentences_dict[key] = ini_path.read_text()

    return sentences_dict


def save_sentences(sentences_dict):
    # Update multiple ini files at once. Paths as keys (relative to
    # profile directory), sentences as values.
    num_chars = 0
    paths_written = []

    new_sentences = {}
    for sentences_key, sentences_text in sentences_dict.items():
        # Path is relative to profile directory
        sentences_path = core.profile.write_path(sentences_key)

        if sentences_text.strip():
            # Overwrite file
            _LOGGER.debug("Writing %s", sentences_path)

            sentences_path.parent.mkdir(parents=True, exist_ok=True)
            sentences_path.write_text(sentences_text)

            num_chars += len(sentences_text)
            paths_written.append(sentences_path)
            new_sentences[sentences_key] = sentences_text
        elif sentences_path.is_file():
            # Remove file
            _LOGGER.debug("Removing %s", sentences_path)
            sentences_path.unlink()

    return new_sentences


def get_phonemes():
    speech_system = core.profile.get("speech_to_text.system", "pocketsphinx")
    examples_path = core.profile.read_path(
        core.profile.get(
            f"speech_to_text.{speech_system}.phoneme_examples", "phoneme_examples.txt"
        )
    )

    # phoneme -> { word, phonemes }
    _LOGGER.debug("Loading phoneme examples from %s", examples_path)
    return load_phoneme_examples(examples_path)


def read_slots(slots_dir: Path) -> typing.Dict[str, str]:
    slots_dict: typing.Dict[st, str] = {}

    if slots_dir.is_dir():
        # Load slot values
        for slot_file_path in slots_dir.glob("*"):
            if slot_file_path.is_file():
                slot_name = slot_file_path.name
                slots_dict[slot_name] = [
                    line.strip() for line in slot_file_path.read_text().splitlines()
                ]

    return slots_dict


def save_slots(
    slots_dir: Path, new_slot_values: typing.Dict[str, str], overwrite_all=True
):
    # Write slots on POST
    if overwrite_all:
        # Remote existing values first
        for name in new_slot_values:
            slots_path = safe_join(slots_dir, f"{name}")
            if slots_path.is_file():
                try:
                    slots_path.unlink()
                except Exception:
                    _LOGGER.exception("api_slots")

    # Write new values
    slots = defaultdict(list)
    for name, values in new_slot_values.items():
        if not values:
            continue

        if isinstance(values, str):
            values = [values]

        slots_path = slots_dir / name

        # Create directories
        slots_path.parent.mkdir(parents=True, exist_ok=True)

        # Merge with existing values
        values = set(values)
        if slots_path.is_file():
            values.update(slots_path.read_text().splitlines())

        # Write merged values
        if values:
            with open(slots_path, "w") as slots_file:
                for value in values:
                    value = value.strip()
                    if value:
                        print(value, file=slots_file)
                        slots[name].append(value)

    return slots


# -----------------------------------------------------------------------------
# Dialogue Manager Setup
# -----------------------------------------------------------------------------

core = None

# We really, *really* want shutdown to be called
@atexit.register
def shutdown(*_args: typing.Any, **kwargs: typing.Any) -> None:
    """Ensure Rhasspy core is stopped."""
    global core
    if core is not None:
        loop.run_until_complete(loop.create_task(core.shutdown()))
        core = None


async def start_rhasspy() -> None:
    """Create actor system and Rhasspy core."""
    global core

    # Load core
    core = RhasspyCore(
        args.profile,
        system_profiles_dir,
        user_profiles_dir,
        host=args.mqtt_host,
        port=args.mqtt_port,
        local_mqtt_port=args.local_mqtt_port,
    )

    # Set environment variables
    os.environ["RHASSPY_BASE_DIR"] = os.getcwd()
    os.environ["RHASSPY_PROFILE"] = core.profile.name
    os.environ["RHASSPY_PROFILE_DIR"] = str(core.profile.write_dir())

    # Add profile settings from the command line
    extra_settings = {}
    for key, value in args.set:
        try:
            value = json.loads(value)
        except Exception:
            pass

        _LOGGER.debug("Profile: %s=%s", key, value)
        extra_settings[key] = value
        core.profile.set(key, value)

    await core.start()
    _LOGGER.info("Started")


# -----------------------------------------------------------------------------
# HTTP API
# -----------------------------------------------------------------------------


@app.route("/api/version")
async def api_version() -> str:
    """Get Rhasspy version."""
    return get_version()


# -----------------------------------------------------------------------------


@app.route("/api/profiles")
async def api_profiles() -> Response:
    """Get list of available profiles and verify necessary files."""
    assert core is not None
    profile_names: typing.Set[str] = set()

    for profiles_dir in [
        core.profile.system_profiles_dir,
        core.profile.user_profiles_dir,
    ]:
        if not profiles_dir.is_dir():
            continue

        for name in profiles_dir.glob("*"):
            profile_dir = profiles_dir / name
            if profile_dir.is_dir():
                profile_names.add(str(name))

    # TODO: Check for missing profile files
    missing_files = rhasspyprofile.get_missing_files(core.profile)
    downloaded = len(missing_files) == 0
    return jsonify(
        {
            "default_profile": core.profile.name,
            "profiles": sorted(profile_names),
            "downloaded": downloaded,
            "missing_files": {
                str(missing.file_path): (missing.setting_name, missing.setting_value)
                for missing in missing_files
            },
        }
    )


# -----------------------------------------------------------------------------


@app.route("/api/download-profile", methods=["POST"])
async def api_download_profile() -> str:
    """Downloads the current profile."""
    assert core is not None

    # TODO: Report status
    await rhasspyprofile.download_files(core.profile)

    return "OK"


@app.route("/api/download-status", methods=["GET"])
async def api_download_status() -> str:
    """Get status of profile download"""
    assert core is not None

    return ""


# -----------------------------------------------------------------------------


@app.route("/api/problems", methods=["GET"])
async def api_problems() -> Response:
    """Returns any problems Rhasspy has found."""
    assert core is not None

    # TODO: Detect problems at startup
    # return jsonify(await core.get_problems())

    return jsonify({})


# -----------------------------------------------------------------------------


@app.route("/api/microphones", methods=["GET"])
async def api_microphones() -> Response:
    """Get a dictionary of available recording devices"""
    assert core is not None
    # TODO: Add for PyAudio
    microphones = await core.get_microphones()

    return jsonify({mic.name: mic.description for mic in microphones.devices})


# -----------------------------------------------------------------------------


@app.route("/api/test-microphones", methods=["GET"])
async def api_test_microphones() -> Response:
    """Get a dictionary of available, functioning recording devices"""
    assert core is not None
    microphones = await core.get_microphones(test=True)

    return jsonify(
        {
            mic.name: mic.description + (" (working!)" if mic.working else "")
            for mic in microphones.devices
        }
    )


# -----------------------------------------------------------------------------


@app.route("/api/speakers", methods=["GET"])
async def api_speakers() -> Response:
    """Get a dictionary of available playback devices"""
    assert core is not None
    speakers = await core.get_speakers()

    return jsonify({speaker.name: speaker.description for speaker in speakers.devices})


# -----------------------------------------------------------------------------


@app.route("/api/listen-for-wake", methods=["POST"])
async def api_listen_for_wake() -> str:
    """Make Rhasspy listen for a wake word"""
    assert core is not None
    toggle_off = (await request.data).decode().lower() in ["false", "off"]

    if toggle_off:
        # Disable
        core.publish(HotwordToggleOff(siteId=core.siteId))
        return "off"

    # Enable
    core.publish(HotwordToggleOn(siteId=core.siteId))
    return "on"


# -----------------------------------------------------------------------------


@app.route("/api/listen-for-command", methods=["POST"])
async def api_listen_for_command() -> Response:
    """Wake Rhasspy up and listen for a voice command"""
    assert core is not None
    no_hass = request.args.get("nohass", "false").lower() == "true"
    wakewordId = request.args.get("wakewordId", "default")
    sessionId = str(uuid4())

    if no_hass:
        # Temporarily disable intent handling
        core.publish(HandleToggleOff(siteId=core.siteId))

    try:
        # Fake a hotword detected event and wait for intent/notRecognized
        def handle_intent():
            while True:
                _, message = yield

                if isinstance(message, (NluIntent, NluIntentNotRecognized)) and (
                    message.sessionId == sessionId
                ):
                    return message

        topics = [NluIntent.topic(intentName="#"), NluIntentNotRecognized.topic()]
        messages = [
            (
                HotwordDetected(
                    modelId=wakewordId,
                    modelVersion="",
                    modelType="personal",
                    currentSensitivity=1,
                    siteId=core.siteId,
                    sessionId=sessionId,
                    sendAudioCaptured=True,
                ),
                {"wakewordId": wakewordId},
            )
        ]

        # Expecting only a single result
        _LOGGER.debug("Waiting for intent (sessionId=%s)", sessionId)
        result = None
        async for response in core.publish_wait(handle_intent(), messages, topics):
            result = response

        if isinstance(result, (NluIntent, NluIntentNotRecognized)):
            # Convert to dict
            intent_dict = result.to_rhasspy_dict()

        return jsonify(intent_dict)
    finally:
        # Re-enable intent handling
        core.publish(HandleToggleOn(siteId=core.siteId))


# -----------------------------------------------------------------------------


@app.route("/api/profile", methods=["GET", "POST"])
async def api_profile() -> typing.Union[str, Response]:
    """Read or write profile JSON directly"""
    assert core is not None
    layers = request.args.get("layers", "all")

    # Write profile on POST
    if request.method == "POST":
        # Ensure that JSON is valid
        profile_json = await request.json
        return save_profile(profile_json)

    if layers == "defaults":
        # Read default settings
        return jsonify(core.defaults)

    if layers == "profile":
        # Local settings only
        profile_path = Path(core.profile.read_path("profile.json"))
        return await send_file(profile_path)

    return jsonify(core.profile.json)


# -----------------------------------------------------------------------------


@app.route("/api/lookup", methods=["POST"])
async def api_lookup() -> Response:
    """Get CMU phonemes from dictionary or guessed pronunciation(s)"""
    assert core is not None
    n = int(request.args.get("n", 5))
    assert n > 0, "No pronunciations requested"

    data = await request.data
    word = data.decode().strip().lower()
    assert word, "No word to look up"

    result = await core.get_word_pronunciations([word], n)

    # Convert to Rhasspy format
    pronunciations = {
        word: {
            "in_dictionary": any(
                not word_pron.guessed for word_pron in result.wordPhonemes[word]
            ),
            "pronunciations": [
                " ".join(word_pron.phonemes) for word_pron in result.wordPhonemes[word]
            ],
            "phonemes": get_espeak_phonemes(word),
        }
        for word in result.wordPhonemes
    }

    return jsonify(pronunciations[word])


# -----------------------------------------------------------------------------


@app.route("/api/pronounce", methods=["POST"])
async def api_pronounce() -> typing.Union[Response, str]:
    """Pronounce CMU phonemes or word using eSpeak"""
    assert core is not None
    download = request.args.get("download", "false").lower() == "true"

    data = await request.data
    pronounce_str = data.decode().strip()
    assert pronounce_str, "No string to pronounce"

    # phonemes or word
    pronounce_type = request.args.get("type", "phonemes")

    # TODO: Get phonemes
    if pronounce_type == "phonemes":
        # Convert from Sphinx to espeak phonemes
        speech_system = core.profile.get("speech_to_text.system", "pocketsphinx")
        phoneme_map_path = core.profile.read_path(
            core.profile.get(
                f"speech_to_text.{speech_system}.phoneme_map", "espeak_phonemes.txt"
            )
        )

        assert phoneme_map_path.is_file(), f"Missing phoneme map at {phoneme_map_path}"

        # Load map from dictionary phonemes to eSpeak phonemes
        phoneme_map: typing.Dict[str, str] = {}
        with open(phoneme_map_path, "r") as phoneme_map_file:
            for line in phoneme_map_file:
                line = line.strip()
                if line and not line.startswith("#"):
                    # P1 E1
                    # P2 E2
                    parts = line.split(maxsplit=1)
                    phoneme_map[parts[0]] = parts[1]

        # Map to eSpeak phonemes
        espeak_phonemes = "".join(phoneme_map.get(p, p) for p in pronounce_str.split())
        espeak_str = f"[[{espeak_phonemes}]]"
    else:
        # Speak word directly
        espeak_str = pronounce_str

    # Convert to WAV
    wav_bytes = get_espeak_wav(espeak_str, voice=core.profile.get("language"))

    if download:
        # Return WAV
        return Response(wav_bytes, mimetype="audio/wav")

    # Play through speakers
    await core.play_wav_data(wav_bytes)

    return "OK"


# -----------------------------------------------------------------------------


@app.route("/api/play-wav", methods=["POST"])
async def api_play_wav() -> str:
    """Play WAV data through the configured audio output system"""
    assert core is not None

    if request.content_type == "audio/wav":
        wav_bytes = await request.data
    else:
        # Interpret as URL
        data = await request.data
        url = data.decode()
        _LOGGER.debug("Loading WAV data from %s", url)

        async with core.http_session.get(url) as response:
            wav_bytes = await response.read()

    # Play through speakers
    _LOGGER.debug("Playing %s byte(s)", len(wav_bytes))
    await core.play_wav_data(wav_bytes)

    return "OK"


# -----------------------------------------------------------------------------


@app.route("/api/phonemes")
def api_phonemes():
    """Get phonemes and example words for a profile"""
    assert core is not None
    return jsonify(get_phonemes())


# -----------------------------------------------------------------------------


@app.route("/api/sentences", methods=["GET", "POST"])
async def api_sentences():
    """Read or write sentences for a profile"""
    assert core is not None

    if request.method == "POST":
        # Write sentences on POST
        if request.mimetype == "application/json":
            # Update multiple ini files at once. Paths as keys (relative to
            # profile directory), sentences as values.
            num_chars = 0
            paths_written = []

            sentences_dict = await request.json
            for sentences_path, sentences_text in sentences_dict.items():
                # Path is relative to profile directory
                sentences_path = Path(core.profile.write_path(sentences_path))

                if sentences_text.strip():
                    # Overwrite file
                    _LOGGER.debug("Writing %s", sentences_path)

                    sentences_path.parent.mkdir(parents=True, exist_ok=True)
                    sentences_path.write_text(sentences_text)

                    num_chars += len(sentences_text)
                    paths_written.append(sentences_path)
                elif sentences_path.is_file():
                    # Remove file
                    _LOGGER.debug("Removing %s", sentences_path)
                    sentences_path.unlink()

            return "Wrote {} char(s) to {}".format(
                num_chars, [str(p) for p in paths_written]
            )

        # Update sentences.ini only
        sentences_path = Path(
            core.profile.write_path(core.profile.get("speech_to_text.sentences_ini"))
        )

        data = await request.data
        with open(sentences_path, "wb") as sentences_file:
            sentences_file.write(data)
            return "Wrote {} byte(s) to {}".format(len(data), sentences_path)

    # GET
    sentences_path_rel = core.profile.read_path(
        core.profile.get("speech_to_text.sentences_ini")
    )
    sentences_path = Path(sentences_path_rel)

    if prefers_json():
        # Return multiple .ini files, keyed by path relative to profile
        # directory.
        sentences_dict = {}
        if sentences_path.is_file():
            try:
                # Try user profile dir first
                profile_dir = Path(core.profile.user_profiles_dir) / core.profile.name
                key = str(sentences_path.relative_to(profile_dir))
            except Exception:
                # Fall back to system profile dir
                profile_dir = Path(core.profile.system_profiles_dir) / core.profile.name
                key = str(sentences_path.relative_to(profile_dir))

            sentences_dict[key] = sentences_path.read_text()

        ini_dir = Path(
            core.profile.read_path(core.profile.get("speech_to_text.sentences_dir"))
        )

        # Add all .ini files from sentences_dir
        if ini_dir.is_dir():
            for ini_path in ini_dir.glob("*.ini"):
                key = str(ini_path.relative_to(core.profile.read_path()))
                sentences_dict[key] = ini_path.read_text()

        return jsonify(sentences_dict)

    # Return sentences.ini contents only
    if not sentences_path.is_file():
        return ""  # no sentences yet

    # Return file contents
    return await send_file(sentences_path)


# -----------------------------------------------------------------------------


@app.route("/api/custom-words", methods=["GET", "POST"])
async def api_custom_words():
    """Read or write custom word dictionary for a profile"""
    assert core is not None
    speech_system = core.profile.get("speech_to_text.system", "pocketsphinx")

    if request.method == "POST":
        # Write custom words on POST
        custom_words_path = Path(
            core.profile.write_path(
                core.profile.get(
                    f"speech_to_text.{speech_system}.custom_words", "custom_words.txt"
                )
            )
        )

        # Update custom words
        lines_written = 0
        with open(custom_words_path, "w") as custom_words_file:
            data = await request.data
            lines = data.decode().splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                print(line, file=custom_words_file)
                lines_written += 1

            return f"Wrote {lines_written} line(s) to {custom_words_path}"

    custom_words_path = Path(
        core.profile.read_path(
            core.profile.get(
                f"speech_to_text.{speech_system}.custom_words", "custom_words.txt"
            )
        )
    )

    # Return custom_words
    if prefers_json():
        # Return JSON instead of plain text
        if not custom_words_path.is_file():
            return jsonify({})  # no custom_words yet

        with open(custom_words_path, "r") as words_file:
            return jsonify(read_dict(words_file))
    else:
        # Return plain text
        if not custom_words_path.is_file():
            return ""  # no custom_words yet

        # Return file contents
        return await send_file(custom_words_path)

    return ""


# -----------------------------------------------------------------------------


@app.route("/api/train", methods=["POST"])
async def api_train() -> str:
    """Generate speech/intent artifacts for profile."""
    assert core is not None

    start_time = time.time()
    _LOGGER.info("Starting training")

    result = await core.train()
    _LOGGER.debug(result)

    end_time = time.time()

    return "Training completed in %0.2f second(s)" % (end_time - start_time)


# -----------------------------------------------------------------------------


@app.route("/api/restart", methods=["POST"])
async def api_restart() -> str:
    """Restart Rhasspy actors."""
    assert core is not None
    _LOGGER.debug("Restarting Rhasspy")
    in_docker = True

    # Check for PID file from supervisord.
    # If present, send a SIGHUP to restart it and re-read the configuration file,
    # which should have be re-written with a POST to /api/profile.
    pid_path = core.profile.read_path("supervisord.pid")
    if pid_path.is_file():
        in_docker = False
        pid = pid_path.read_text().strip()
        assert pid, f"No PID in {pid_path}"
        restart_command = ["kill", "-SIGHUP", pid]
        _LOGGER.debug(restart_command)

        subprocess.check_call(restart_command)

    if in_docker:
        # Signal Docker orchestration script to restart.
        # Wait for file to be deleted as a signal that restart is complete.
        restart_path = core.profile.write_path(".restart_docker")
        restart_path.write_text("")

    # Shut down core
    await core.shutdown()

    # Start core back up
    await start_rhasspy()

    if in_docker:
        seconds_left = 30
        wait_seconds = 0.5
        while restart_path.is_file():
            await asyncio.sleep(wait_seconds)
            seconds_left -= wait_seconds
            if seconds_left <= 0:
                raise RuntimeError("Did not receive shutdown signal within timeout")

    return "Restarted Rhasspy"


# -----------------------------------------------------------------------------


@app.route("/api/speech-to-text", methods=["POST"])
async def api_speech_to_text() -> str:
    """Transcribe speech from WAV file."""
    no_header = request.args.get("noheader", "false").lower() == "true"
    assert core is not None

    # Prefer 16-bit 16Khz mono, but will convert with sox if needed
    wav_bytes = await request.data
    if no_header:
        # Wrap in WAV
        wav_bytes = buffer_to_wav(wav_bytes)

    start_time = time.perf_counter()
    result = await core.transcribe_wav(wav_bytes)
    end_time = time.perf_counter()

    if prefers_json():
        # Return extra info in JSON
        return jsonify(
            {
                "text": result.text,
                "likelihood": result.likelihood,
                "transcribe_seconds": (end_time - start_time),
                "wav_seconds": get_wav_duration(wav_bytes),
            }
        )

    return result.text


# -----------------------------------------------------------------------------


@app.route("/api/text-to-intent", methods=["POST"])
async def api_text_to_intent():
    """Recgonize intent from text and optionally handle."""
    assert core is not None
    data = await request.data
    text = data.decode()
    no_hass = request.args.get("nohass", "false").lower() == "true"

    if no_hass:
        # Temporarily disable intent handling
        core.publish(HandleToggleOff(siteId=core.siteId))

    try:
        # Convert text to intent
        start_time = time.time()
        intent = (await core.recognize_intent(text)).to_rhasspy_dict()
        intent["raw_text"] = text
        intent["speech_confidence"] = 1

        intent_sec = time.time() - start_time
        intent["time_sec"] = intent_sec

        return jsonify(intent)
    finally:
        # Re-enable intent handling
        core.publish(HandleToggleOn(siteId=core.siteId))


# -----------------------------------------------------------------------------


@app.route("/api/speech-to-intent", methods=["POST"])
async def api_speech_to_intent() -> Response:
    """Transcribe speech, recognize intent, and optionally handle."""
    assert core is not None
    no_hass = request.args.get("nohass", "false").lower() == "true"

    if no_hass:
        # Temporarily disable intent handling
        core.publish(HandleToggleOff(siteId=core.siteId))

    try:
        # Prefer 16-bit 16Khz mono, but will convert with sox if needed
        wav_bytes = await request.data

        # speech -> text
        start_time = time.time()
        transcription = await core.transcribe_wav(wav_bytes)
        text = transcription.text
        _LOGGER.debug(text)

        # text -> intent
        intent = (await core.recognize_intent(text)).to_rhasspy_dict()
        intent["raw_text"] = transcription.text
        intent["speech_confidence"] = transcription.likelihood

        intent_sec = time.time() - start_time
        intent["time_sec"] = intent_sec

        _LOGGER.debug(intent)
    finally:
        # Re-enable intent handling
        core.publish(HandleToggleOn(siteId=core.siteId))

    return jsonify(intent)


# -----------------------------------------------------------------------------

session_names: typing.Dict[str, str] = {}


@app.route("/api/start-recording", methods=["POST"])
async def api_start_recording() -> str:
    """Begin recording voice command."""
    assert core is not None
    name = str(request.args.get("name", ""))

    # Start new ASR session
    sessionId = str(uuid4())
    core.publish(
        AsrStartListening(
            siteId=core.siteId,
            sessionId=sessionId,
            stopOnSilence=False,
            sendAudioCaptured=True,
        )
    )

    # Map to user provided name
    session_names[name] = sessionId

    return sessionId


@app.route("/api/stop-recording", methods=["POST"])
async def api_stop_recording() -> Response:
    """End recording voice command. Transcribe and handle."""
    assert core is not None
    no_hass = request.args.get("nohass", "false").lower() == "true"

    if no_hass:
        # Temporarily disable intent handling
        core.publish(HandleToggleOff(siteId=core.siteId))

    try:
        # End session
        # TODO: Handle AsrAudioCaptured
        name = request.args.get("name", "")
        assert name in session_names, f"No session for name: {name}"
        sessionId = session_names.pop(name)

        def handle_captured():
            while True:
                _, message = yield

                if isinstance(message, AsrTextCaptured) and (
                    message.sessionId == sessionId
                ):
                    return message

        topics = [AsrTextCaptured.topic()]
        messages = [AsrStopListening(siteId=core.siteId, sessionId=sessionId)]

        # Expecting only a single result
        _LOGGER.debug("Waiting for text captured (sessionId=%s)", sessionId)
        text_captured = None
        async for response in core.publish_wait(handle_captured(), messages, topics):
            text_captured = response

        assert isinstance(text_captured, AsrTextCaptured)
        text = text_captured.text
        result = await core.recognize_intent(text)

        if isinstance(result, (NluIntent, NluIntentNotRecognized)):
            # Convert to dict
            intent = result.to_rhasspy_dict()

        return jsonify(intent)
    finally:
        # Re-enable intent handling
        core.publish(HandleToggleOn(siteId=core.siteId))


@app.route("/api/play-recording", methods=["POST"])
async def api_play_recording() -> str:
    """Play last recorded voice command through the configured audio output system"""
    assert core is not None

    if core.last_audio_captured:
        # Play through speakers
        wav_bytes = core.last_audio_captured.wav_bytes
        _LOGGER.debug("Playing %s byte(s)", len(wav_bytes))
        await core.play_wav_data(wav_bytes)

    return "OK"


# -----------------------------------------------------------------------------


@app.route("/api/unknown-words", methods=["GET"])
async def api_unknown_words() -> Response:
    """Get list of unknown words."""
    assert core is not None
    speech_system = core.profile.get("speech_to_text.system", "pocketsphinx")
    unknown_words = {}
    unknown_path = Path(
        core.profile.read_path(
            core.profile.get(
                f"speech_to_text.{speech_system}.unknown_words", "unknown_words.txt"
            )
        )
    )

    if unknown_path.is_file():
        # Load dictionary of unknown words
        for line in open(unknown_path, "r"):
            line = line.strip()
            if line:
                word, pronunciation = re.split(r"[ ]+", line, maxsplit=1)
                unknown_words[word] = pronunciation

    return jsonify(unknown_words)


# -----------------------------------------------------------------------------

last_sentence = ""


@app.route("/api/text-to-speech", methods=["POST"])
async def api_text_to_speech() -> typing.Union[bytes, str]:
    """Speak a sentence with text to speech system."""
    global last_sentence
    repeat = request.args.get("repeat", "false").strip().lower() == "true"
    language = request.args.get("language")
    voice = request.args.get("voice")
    data = await request.data

    # Repeat last sentence or use incoming plain text
    sentence = last_sentence if repeat else data.decode().strip()

    assert core is not None
    await core.speak_sentence(sentence, language=(language or voice))

    # Cache last sentence spoken
    last_sentence = sentence

    return sentence


# -----------------------------------------------------------------------------


@app.route("/api/slots", methods=["GET", "POST"])
async def api_slots() -> typing.Union[str, Response]:
    """Get the values of all slots."""
    assert core is not None

    if request.method == "POST":
        # Write slots on POST
        overwrite_all = request.args.get("overwrite_all", "false").lower() == "true"
        new_slot_values = await request.json

        slots_dir = Path(
            core.profile.write_path(
                core.profile.get("speech_to_text.slots_dir", "slots")
            )
        )

        if overwrite_all:
            # Remote existing values first
            for name in new_slot_values:
                slots_path = safe_join(slots_dir, f"{name}")
                if slots_path.is_file():
                    try:
                        slots_path.unlink()
                    except Exception:
                        _LOGGER.exception("api_slots")

        # Write new values
        for name, values in new_slot_values.items():
            if isinstance(values, str):
                values = [values]

            slots_path = slots_dir / name

            # Create directories
            slots_path.parent.mkdir(parents=True, exist_ok=True)

            # Merge with existing values
            values = set(values)
            if slots_path.is_file():
                values.update(slots_path.read_text().splitlines())

            # Write merged values
            if values:
                with open(slots_path, "w") as slots_file:
                    for value in values:
                        value = value.strip()
                        if value:
                            print(value, file=slots_file)

        return "OK"

    # Read slots into dictionary
    slots_dir = Path(
        core.profile.read_path(core.profile.get("speech_to_text.slots_dir", "slots"))
    )

    slots_dict = {}

    if slots_dir.is_dir():
        # Load slot values
        for slot_file_path in slots_dir.glob("*"):
            if slot_file_path.is_file():
                slot_name = slot_file_path.name
                slots_dict[slot_name] = [
                    line.strip() for line in slot_file_path.read_text().splitlines()
                ]

    return jsonify(slots_dict)


@app.route("/api/slots/<name>", methods=["GET", "POST"])
def api_slots_by_name(name: str) -> typing.Union[str, Response]:
    """Get or set the values of a slot list."""
    assert core is not None
    overwrite_all = request.args.get("overwrite_all", "false").lower() == "true"

    slots_dir = Path(
        core.profile.read_path(core.profile.get("speech_to_text.slots_dir", "slots"))
    )

    if request.method == "POST":
        if overwrite_all:
            # Remote existing values first
            slots_path = safe_join(slots_dir, f"{name}")
            if slots_path.is_file():
                try:
                    slots_path.unlink()
                except Exception:
                    _LOGGER.exception("api_slots_by_name")

        slots_path = Path(
            core.profile.write_path(
                core.profile.get("speech_to_text.slots_dir", "slots"), f"{name}"
            )
        )

        # Create directories
        slots_path.parent.mkdir(parents=True, exist_ok=True)

        # Write data
        with open(slots_path, "wb") as slots_file:
            slots_file.write(request.data)

        return f"Wrote {len(request.data)} byte(s) to {slots_path}"

    # Load slots values
    slot_values: typing.List[str] = []
    slot_file_path = slots_dir / name
    if slot_file_path.is_file():
        slot_values = [line.strip() for line in slot_file_path.read_text().splitlines()]

    return jsonify(slot_values)


# -----------------------------------------------------------------------------


@app.route("/api/intents")
def api_intents():
    """Return JSON with information about intents."""
    assert core is not None

    sentences_ini = core.profile.read_path(
        core.profile.get("speech_to_text.sentences_ini")
    )
    sentences_dir = core.profile.read_path(
        core.profile.get("speech_to_text.sentences_dir")
    )

    # Load all .ini files and parse
    ini_paths: typing.List[Path] = get_ini_paths(sentences_ini, sentences_dir)
    intents: typing.Dict[str, typing.Any] = get_all_intents(ini_paths)

    def add_type(item, item_dict: typing.Dict[str, typing.Any]):
        """Add item_type to expression dictionary."""
        item_dict["item_type"] = type(item).__name__
        if hasattr(item, "items"):
            # Group, alternative, etc.
            for sub_item, sub_item_dict in zip(item.items, item_dict["items"]):
                add_type(sub_item, sub_item_dict)
        elif hasattr(item, "rule_body"):
            # Rule
            add_type(item.rule_body, item_dict["rule_body"])

    # Convert to dictionary
    intents_dict = {}
    for intent_name, intent_sentences in intents.items():
        sentence_dicts = []
        for sentence in intent_sentences:
            sentence_dict = attr.asdict(sentence)

            # Add item_type field
            add_type(sentence, sentence_dict)
            sentence_dicts.append(sentence_dict)

        intents_dict[intent_name] = sentence_dicts

    # Convert to JSON
    return jsonify(intents_dict)


# -----------------------------------------------------------------------------


@app.route("/api/mqtt/<path:topic>", methods=["POST"])
async def api_mqtt(topic: str):
    """POST an MQTT message to a topic."""
    assert core is not None
    assert core.client is not None
    payload = await request.data

    # Publish directly to MQTT broker
    core.client.publish(topic, payload)

    return f"Published {len(payload)} byte(s) to {topic}"


# -----------------------------------------------------------------------------
# WebSocket API
# -----------------------------------------------------------------------------

# Logging
logging_websockets: typing.Set[asyncio.Queue] = set()


def logging_websocket(func):
    """Wraps a websocket route to use the logging_websockets queue"""

    @wraps(func)
    async def wrapper(*_args, **kwargs):
        global logging_websockets
        queue = asyncio.Queue()
        logging_websockets.add(queue)
        try:
            return await func(queue, *_args, **kwargs)
        finally:
            logging_websockets.remove(queue)

    return wrapper


async def broadcast_logging(message):
    """Broadcasts a logging message to all logging websockets."""
    for queue in logging_websockets:
        await queue.put(message)


logging.root.addHandler(
    FunctionLoggingHandler(
        lambda message: asyncio.run_coroutine_threadsafe(
            broadcast_logging(message), loop
        )
    )
)


@app.websocket("/api/events/log")
@logging_websocket
async def api_events_log(queue) -> None:
    """Websocket endpoint to send out logging messages as text."""
    while True:
        try:
            message = await queue.get()
            await websocket.send(message)
        except Exception:
            break


# -----------------------------------------------------------------------------

# MQTT
mqtt_websockets: typing.Set[asyncio.Queue] = set()


def mqtt_websocket(func):
    """Wraps a websocket route to use the mqtt_websockets queue"""

    @wraps(func)
    async def wrapper(*_args, **kwargs):
        global mqtt_websockets
        queue = asyncio.Queue()
        mqtt_websockets.add(queue)

        assert core
        core.message_queues.add(queue)

        try:
            return await func(queue, *_args, **kwargs)
        finally:
            mqtt_websockets.remove(queue)
            core.message_queues.remove(queue)

    return wrapper


async def broadcast_mqtt(topic: str, payload: typing.Union[str, bytes]):
    """Broadcasts an MQTT topic/payload to all MQTT websockets."""
    for queue in mqtt_websockets:
        await queue.put((topic, payload))


def handle_ws_mqtt(message: typing.Union[str, bytes], matcher: MQTTMatcher):
    """Handle subscribe/publish MQTT requests from a websocket."""
    assert core
    assert core.client

    message_dict = json.loads(message)
    message_type = message_dict["type"]
    assert message_type in [
        "subscribe",
        "publish",
    ], f'Invalid message type "{message_type}" (must be "subscribe" or "publish")'

    topic = message_dict["topic"]

    if message_type == "subscribe":
        core.client.subscribe(topic)
        matcher[topic] = message_dict
        _LOGGER.debug("Subscribed to %s", topic)
    elif message_type == "publish":
        payload = json.dumps(message_dict["payload"])
        core.client.publish(topic, payload)
        _LOGGER.debug("Published %s char(s) to %s", len(payload), topic)


@app.websocket("/api/mqtt")
@mqtt_websocket
async def api_ws_mqtt(queue) -> None:
    """Websocket endpoint to send/receive MQTT messages."""
    topic_matcher = MQTTMatcher()
    receive_task = loop.create_task(websocket.receive())
    send_task = loop.create_task(queue.get())
    pending: typing.Set[
        typing.Union[asyncio.Future[typing.Any], asyncio.Task[typing.Any]]
    ] = {receive_task, send_task}

    while True:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task is receive_task:
                try:
                    # Process received message
                    message = task.result()
                    handle_ws_mqtt(message, topic_matcher)
                except Exception:
                    _LOGGER.debug(message)
                    _LOGGER.exception("api_ws_mqtt (receive)")

                # Schedule another receive
                receive_task = loop.create_task(websocket.receive())
                pending.add(receive_task)
            elif task is send_task:
                try:
                    # Send out queued message (if matches topic)
                    message = task.result()
                    if message[0] == "mqtt":
                        topic, payload = message[1], message[2]
                        mqtt_message = json.dumps(
                            {"topic": topic, "payload": json.loads(payload)}
                        )

                        for _ in topic_matcher.iter_match(topic):
                            await websocket.send(mqtt_message)
                            _LOGGER.debug(
                                "Sent %s char(s) to websocket", len(mqtt_message)
                            )
                            break
                except Exception:
                    _LOGGER.debug(topic)
                    _LOGGER.exception("api_ws_mqtt (send)")

                # Schedule another send
                send_task = loop.create_task(queue.get())
                pending.add(send_task)


@app.websocket("/api/events/intent")
@mqtt_websocket
async def api_ws_intent(queue) -> None:
    """Websocket endpoint to send intents in Rhasspy JSON format."""
    while True:
        message = await queue.get()
        if message[0] == "intent":
            try:
                intent = typing.cast(
                    typing.Union[NluIntent, NluIntentNotRecognized], message[1]
                )
                intent_dict = intent.to_rhasspy_dict()

                ws_message = json.dumps(intent_dict)
                await websocket.send(ws_message)
                _LOGGER.debug("Sent %s char(s) to websocket", len(ws_message))
            except Exception:
                _LOGGER.exception("api_ws_intent")


@app.websocket("/api/events/wake")
@mqtt_websocket
async def api_ws_wake(queue) -> None:
    """Websocket endpoint to notify clients on wake up."""
    while True:
        message = await queue.get()
        if message[0] == "wake":
            try:
                hotword_detected: HotwordDetected = message[1]
                wakewordId: str = message[2]

                ws_message = json.dumps(
                    {"wakewordId": wakewordId, "siteId": hotword_detected.siteId}
                )
                await websocket.send(ws_message)
                _LOGGER.debug("Sent %s char(s) to websocket", len(ws_message))
            except Exception:
                _LOGGER.exception("api_ws_wake")


@app.websocket("/api/events/text")
@mqtt_websocket
async def api_ws_text(queue) -> None:
    """Websocket endpoint to notify clients when speech is transcribed."""
    while True:
        message = await queue.get()
        if message[0] == "text":
            try:
                text_captured: AsrTextCaptured = message[1]
                wakewordId: str = message[2]

                ws_message = json.dumps(
                    {
                        "text": text_captured.text,
                        "siteId": text_captured.siteId,
                        "wakewordId": wakewordId,
                    }
                )
                await websocket.send(ws_message)
                _LOGGER.debug("Sent %s char(s) to websocket", len(ws_message))
            except Exception:
                _LOGGER.exception("api_ws_wake")


# -----------------------------------------------------------------------------
# MaryTTS
# -----------------------------------------------------------------------------

# @app.route("/process", methods=["GET"])
# async def marytts_process():
#     """Emulate MaryTTS /process API"""
#     global last_sentence

#     assert core is not None
#     sentence = request.args.get("INPUT_TEXT", "")
#     voice = request.args.get("VOICE")
#     locale = request.args.get("LOCALE")
#     spoken = await core.speak_sentence(
#         sentence, play=False, voice=voice, language=locale
#     )

#     return spoken.wav_bytes


# -----------------------------------------------------------------------------


@app.errorhandler(Exception)
async def handle_error(err) -> typing.Tuple[str, int]:
    """Return error as text."""
    _LOGGER.exception(err)
    return (str(err), 500)


# ---------------------------------------------------------------------
# Static Routes
# ---------------------------------------------------------------------

css_dir = web_dir / "css"
js_dir = web_dir / "js"
img_dir = web_dir / "img"
webfonts_dir = web_dir / "webfonts"
docs_dir = web_dir / "docs"


@app.route("/css/<path:filename>", methods=["GET"])
async def css(filename) -> Response:
    """CSS static endpoint."""
    return await send_from_directory(css_dir, filename)


@app.route("/js/<path:filename>", methods=["GET"])
async def js(filename) -> Response:
    """Javascript static endpoint."""
    return await send_from_directory(js_dir, filename)


@app.route("/img/<path:filename>", methods=["GET"])
async def img(filename) -> Response:
    """Image static endpoint."""
    return await send_from_directory(img_dir, filename)


@app.route("/webfonts/<path:filename>", methods=["GET"])
async def webfonts(filename) -> Response:
    """Web font static endpoint."""
    return await send_from_directory(webfonts_dir, filename)


# @app.route("/docs", methods=["GET"])
# @app.route("/docs/", methods=["GET"])
# async def docs_index() -> Response:
#     """Documentation index static endpoint."""
#     return await send_from_directory(docs_dir, "index.html")


# @app.route("/docs/<path:filename>", methods=["GET"])
# async def docs(filename) -> Response:
#     """Documentation static endpoint."""
#     doc_file = Path(safe_join(docs_dir, filename))
#     if doc_file.is_dir():
#         doc_file = doc_file / "index.html"

#     return await send_file(doc_file)


# ----------------------------------------------------------------------------
# HTML Page Routes
# ----------------------------------------------------------------------------


@app.route("/", methods=["GET"])
async def page_index() -> Response:
    """Render main web page."""
    return await render_template("index.html", page="Home", **get_template_args())


@app.route("/sentences", methods=["GET", "POST"])
async def page_sentences() -> Response:
    """Render sentences web page."""
    sentences_path = core.profile.read_path(
        core.profile.get("speech_to_text.sentences_ini", "sentences.ini")
    )
    sentences_dir = core.profile.write_path(
        core.profile.get("speech_to_text.sentences_dir", "intents")
    )

    sentences = read_sentences(sentences_path, sentences_dir)

    if request.method == "POST":
        form = await request.form
        new_sentences = json.loads(form["sentences"])
        sentences = save_sentences(new_sentences)

    sentences_json = json.dumps(sentences, indent=4)

    return await render_template(
        "sentences.html",
        page="Sentences",
        sentences=sentences,
        sentences_json=sentences_json,
        **get_template_args(),
    )


@app.route("/words", methods=["GET", "POST"])
async def page_words() -> Response:
    """Render words web page."""
    custom_words = ""
    speech_system = core.profile.get("speech_to_text.system", "pocketsphinx")
    words_path = core.profile.get(
        f"speech_to_text.{speech_system}.custom_words", "custom_words.txt"
    )

    words_write_path = core.profile.write_path(words_path)

    if request.method == "POST":
        form = await request.form
        custom_words = form["customWords"]
        words_write_path.write_text(custom_words)
    elif words_write_path.is_file():
        custom_words = words_write_path.read_text()

    # Load phoneme examples
    phonemes = get_phonemes()

    return await render_template(
        "words.html",
        page="Words",
        custom_words=custom_words,
        phonemes=phonemes,
        **get_template_args(),
    )


@app.route("/slots", methods=["GET", "POST"])
async def page_slots() -> Response:
    """Render slots web page."""
    slots_dir = core.profile.read_path(
        core.profile.get("speech_to_text.slots_dir", "slots")
    )
    slots: typing.Dict[str, str] = {}

    if request.method == "POST":
        form = await request.form
        newline_slots = json.loads(form["slots"])
        new_slot_values = {
            name: value.splitlines() for name, value in newline_slots.items()
        }
        slots = save_slots(slots_dir, new_slot_values)
    elif slots_dir.is_dir():
        slots = read_slots(slots_dir)

    slots_json = json.dumps(
        {name: "\n".join(values) for name, values in slots.items()}, indent=4
    )
    return await render_template(
        "slots.html",
        page="Slots",
        slots=slots,
        slots_json=slots_json,
        sorted=sorted,
        **get_template_args(),
    )


@app.route("/settings", methods=["GET"])
async def page_settings() -> Response:
    """Render settings web page."""
    return await render_template(
        "settings.html", page="Settings", **get_template_args()
    )


@app.route("/advanced", methods=["GET", "POST"])
async def page_advanced() -> Response:
    """Render advanced web page."""
    if request.method == "POST":
        form = await request.form
        profile_json = json.loads(form["profileJson"])
        save_profile(profile_json)

    profile_path = Path(core.profile.read_path("profile.json"))
    local_profile_json = profile_path.read_text()

    return await render_template(
        "advanced.html",
        page="Advanced",
        local_profile_json=local_profile_json,
        **get_template_args(),
    )


@app.route("/swagger.yaml", methods=["GET"])
async def swagger_yaml() -> Response:
    """OpenAPI static endpoint."""
    return await send_file(web_dir / "swagger.yaml")


# -----------------------------------------------------------------------------

# Swagger UI
quart_api_doc(
    app, config_path=(web_dir / "swagger.yaml"), url_prefix="/api", title="Rhasspy API"
)

# -----------------------------------------------------------------------------


def prefers_json() -> bool:
    """True if client prefers JSON over plain text."""
    return quality(request.accept_mimetypes, "application/json") > quality(
        request.accept_mimetypes, "text/plain"
    )


def quality(accept, key: str) -> float:
    """Return Accept quality for media type."""
    for option in accept.options:
        # pylint: disable=W0212
        if accept._values_match(key, option.value):
            return option.quality
    return 0.0


# -----------------------------------------------------------------------------

# Start Rhasspy actors
loop.run_until_complete(start_rhasspy())

# -----------------------------------------------------------------------------

# Disable useless logging messages
logging.getLogger("wsproto").setLevel(logging.CRITICAL)

# Start web server
if args.ssl is not None:
    _LOGGER.debug("Using SSL with certfile, keyfile = %s", args.ssl)
    certfile = args.ssl[0]
    keyfile = args.ssl[1]
    protocol = "https"
else:
    certfile = None
    keyfile = None
    protocol = "http"

_LOGGER.debug("Starting web server at %s://%s:%s", protocol, args.host, args.port)

try:
    app.run(host=args.host, port=args.port, certfile=certfile, keyfile=keyfile)
except KeyboardInterrupt:
    pass
