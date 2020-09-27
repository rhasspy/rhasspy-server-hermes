"""Hermes implementation of RhasspyCore"""
import asyncio
import io
import logging
import os
import ssl
import sys
import threading
import time
import typing
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import paho.mqtt.client as mqtt
from paho.mqtt.matcher import MQTTMatcher

import rhasspynlu
from rhasspyhermes.asr import (
    AsrAudioCaptured,
    AsrError,
    AsrStartListening,
    AsrStopListening,
    AsrTextCaptured,
    AsrToggleOff,
    AsrToggleOn,
    AsrToggleReason,
    AsrTrain,
    AsrTrainSuccess,
)
from rhasspyhermes.audioserver import (
    AudioDeviceMode,
    AudioDevices,
    AudioFrame,
    AudioGetDevices,
    AudioPlayBytes,
    AudioPlayError,
    AudioPlayFinished,
    AudioRecordError,
    AudioSessionFrame,
    AudioSummary,
)
from rhasspyhermes.base import Message
from rhasspyhermes.client import HermesClient
from rhasspyhermes.dialogue import DialogueSessionStarted
from rhasspyhermes.g2p import G2pError, G2pPhonemes, G2pPronounce
from rhasspyhermes.nlu import (
    NluError,
    NluIntent,
    NluIntentNotRecognized,
    NluQuery,
    NluTrain,
    NluTrainSuccess,
)
from rhasspyhermes.tts import GetVoices, TtsError, TtsSay, TtsSayFinished, Voices
from rhasspyhermes.wake import (
    GetHotwords,
    HotwordDetected,
    HotwordError,
    HotwordExampleRecorded,
    Hotwords,
    HotwordToggleOff,
    HotwordToggleOn,
    HotwordToggleReason,
)
from rhasspyprofile import Profile

from .train import sentences_to_graph
from .utils import get_ini_paths, wav_to_buffer

_LOGGER = logging.getLogger("rhasspyserver_hermes")

# -----------------------------------------------------------------------------


@dataclass
class TrainingFailedException(Exception):
    """Raised when training fails."""

    reason: str

    def __str__(self):
        return self.reason


@dataclass
class AudioEnergies:
    """Summary of energy from recorded audio."""

    current_energy: float = float("nan")
    max_energy: float = float("nan")
    min_energy: float = float("nan")
    max_current_ratio: float = float("nan")


# -----------------------------------------------------------------------------


class RhasspyCore:
    """Core Rhasspy functionality, abstracted over MQTT Hermes services"""

    def __init__(
        self,
        profile_name: str,
        system_profiles_dir: typing.Optional[typing.Union[str, Path]],
        user_profiles_dir: typing.Union[str, Path],
        host: typing.Optional[str] = None,
        port: typing.Optional[int] = None,
        username: typing.Optional[str] = None,
        password: typing.Optional[str] = None,
        local_mqtt_port: int = 12183,
        connection_retries: int = 10,
        retry_seconds: float = 1,
        message_timeout_seconds: float = 30,
        training_timeout_seconds: float = 600,
        certfile: typing.Optional[str] = None,
        keyfile: typing.Optional[str] = None,
        loop: typing.Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.loop = loop or asyncio.get_event_loop()

        # Load profile
        self.profile_name = profile_name

        if system_profiles_dir:
            system_profiles_dir = Path(system_profiles_dir)

        self.user_profiles_dir = Path(user_profiles_dir)

        self.profile = Profile(
            self.profile_name, system_profiles_dir, self.user_profiles_dir
        )

        self.defaults = Profile.load_defaults(self.profile.system_profiles_dir)

        # Look up site id(s) in profile
        site_ids = str(self.profile.get("mqtt.site_id", "")).split(",")
        self.site_ids = set(site_ids)

        if site_ids:
            self.site_id = site_ids[0]
        else:
            self.site_id = "default"

        # Satellite site ids
        self.satellite_site_ids = defaultdict(
            set,
            {
                "wake": self.parse_satellite_site_ids("wake"),
                "intent": self.parse_satellite_site_ids("intent"),
                "speech_to_text": self.parse_satellite_site_ids("speech_to_text"),
                "text_to_speech": self.parse_satellite_site_ids("text_to_speech"),
            },
        )

        # MQTT client
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message

        self.is_connected = False
        self.connected_event = threading.Event()

        # Message types that are we are subscribed to
        self.subscribed_types: typing.Set[typing.Type[Message]] = set()
        self.subscribed_topics: typing.Set[str] = set()

        # Cache of all MQTT topics in case we get disconnected
        self.all_mqtt_topics: typing.Set[str] = set()

        # Default timeout for response messages
        self.message_timeout_seconds = message_timeout_seconds
        self.training_timeout_seconds = training_timeout_seconds

        remote_mqtt = str(self.profile.get("mqtt.enabled", False)).lower() == "true"
        if remote_mqtt:
            # External broker
            self.host = host or str(self.profile.get("mqtt.host", "localhost"))
            self.port = port or int(self.profile.get("mqtt.port", 1883))
            self.username = username or str(self.profile.get("mqtt.username", ""))
            self.password = password or str(self.profile.get("mqtt.password", ""))
        else:
            # Internal broker
            self.host = host or "localhost"
            self.port = port or local_mqtt_port
            self.username = ""
            self.password = ""

        if self.username:
            # MQTT username/password
            self.client.username_pw_set(self.username, self.password)

        # MQTT TLS
        tls_enabled = self.profile.get("mqtt.tls.enabled", False)
        if tls_enabled:
            _LOGGER.debug("Loading MQTT TLS settings")

            # Certificate Authority certs
            tls_ca_certs = self.profile.get("mqtt.tls.ca_certs")

            # CERT_REQUIRED, CERT_OPTIONAL, CERT_NONE
            tls_cert_reqs = self.profile.get("mqtt.tls.cert_reqs")

            # PEM
            tls_certfile = self.profile.get("mqtt.tls.certfile")
            if tls_certfile == "":
                tls_certfile = None
            if tls_certfile is not None:
                tls_certfile = os.path.expandvars(tls_certfile)

            tls_keyfile = self.profile.get("mqtt.tls.keyfile")
            if tls_keyfile == "":
                tls_keyfile = None
            if tls_keyfile is not None:
                tls_keyfile = os.path.expandvars(tls_keyfile)

            # Cipers/version
            tls_ciphers = self.profile.get("mqtt.tls.ciphers")
            tls_version = self.profile.get("mqtt.tls.version")

            if tls_version is None:
                # Use highest TLS version
                tls_version = ssl.PROTOCOL_TLS

            tls_args = {
                "ca_certs": tls_ca_certs,
                "certfile": tls_certfile,
                "keyfile": tls_keyfile,
                "cert_reqs": getattr(ssl, tls_cert_reqs),
                "tls_version": tls_version,
                "ciphers": (tls_ciphers or None),
            }
            _LOGGER.debug("MQTT TLS enabled: %s", tls_args)
            self.client.tls_set(**tls_args)

        # MQTT retries
        self.connection_retries = connection_retries
        self.retry_seconds = retry_seconds

        # id -> matcher
        self.handler_matchers: typing.Dict[str, MQTTMatcher] = {}

        # id -> async queue
        self.handler_queues: typing.Dict[str, asyncio.Queue] = {}

        # External message queues
        self.message_queues: typing.Set[asyncio.Queue] = set()

        # Cached wake word IDs for sessions
        # Holds last voice command
        self.last_audio_captured: typing.Optional[AsrAudioCaptured] = None

        # Enable/disable playing sounds on events
        self.sounds_enabled = True

        self.asr_system = self.profile.get("speech_to_text.system", "dummy")
        self.nlu_system = self.profile.get("intent.system", "dummy")
        self.dialogue_system = self.profile.get("dialogue.system", "dummy")
        self.sound_system = self.profile.get("sounds.system", "dummy")

        # Audio summaries
        self.audio_summaries = False
        self.audio_energies = AudioEnergies()

    # -------------------------------------------------------------------------

    def start(self):
        """Connect to MQTT broker"""
        _LOGGER.debug("Starting core")

        # Connect to MQTT broker
        retries = 0
        while (not self.is_connected) and (retries < self.connection_retries):
            try:
                _LOGGER.debug(
                    "Connecting to %s:%s (retries: %s/%s)",
                    self.host,
                    self.port,
                    retries,
                    self.connection_retries,
                )

                self.client.connect(self.host, self.port)
                self.client.loop_start()
                self.connected_event.wait(timeout=5)
            except Exception:
                _LOGGER.exception("mqtt connect")
                retries += 1
                time.sleep(self.retry_seconds)

        if not self.is_connected:
            _LOGGER.fatal(
                "Failed to connect to MQTT broker (%s:%s)", self.host, self.port
            )
            raise RuntimeError("Failed to connect to MQTT broker")

    def shutdown(self):
        """Disconnect from MQTT broker"""
        print("Shutting down core", file=sys.stderr)

        # Shut down MQTT client
        self.client.loop_stop()

    # -------------------------------------------------------------------------

    async def train(self):
        """Send an NLU query and wait for intent or not recognized"""

        # Load sentences.ini files
        sentences_ini = self.profile.read_path(
            self.profile.get("speech_to_text.sentences_ini", "sentences.ini")
        )
        sentences_dir: typing.Optional[Path] = self.profile.read_path(
            self.profile.get("speech_to_text.sentences_dir", "intents")
        )

        assert sentences_dir is not None
        if not sentences_dir.is_dir():
            sentences_dir = None

        ini_paths = get_ini_paths(sentences_ini, sentences_dir)
        _LOGGER.debug("Loading sentences from %s", ini_paths)

        sentences_dict = {str(p): p.read_text() for p in ini_paths}

        # Load settings
        language = self.profile.get("language", "en")
        dictionary_casing = self.profile.get(
            "speech_to_text.dictionary_casing", "ignore"
        ).lower()
        word_transform = None
        if dictionary_casing == "upper":
            word_transform = str.upper
        elif dictionary_casing == "lower":
            word_transform = str.lower

        slots_dir = self.profile.write_path(
            self.profile.get("speech_to_text.slots_dir", "slots")
        )
        system_slots_dir = (
            self.profile.system_profiles_dir / self.profile.name / "slots"
        )
        slot_programs_dir = self.profile.write_path(
            self.profile.get("speech_to_text.slot_programs_dir", "slot_programs")
        )
        system_slot_programs_dir = (
            self.profile.system_profiles_dir / self.profile.name / "slot_programs"
        )

        # Convert to graph
        _LOGGER.debug("Generating intent graph")

        intent_graph, slot_replacements = sentences_to_graph(
            sentences_dict,
            slots_dirs=[slots_dir, system_slots_dir],
            slot_programs_dirs=[slot_programs_dir, system_slot_programs_dir],
            language=language,
            word_transform=word_transform,
        )

        # Convert to dict for train messages
        slots_dict = {
            slot_name: [value.text for value in values]
            for slot_name, values in slot_replacements.items()
        }

        # Convert to gzipped pickle
        graph_path = self.profile.write_path(
            self.profile.get("intent.fsticuffs.intent_json", "intent_graph.pickle.gz")
        )
        _LOGGER.debug("Writing %s", graph_path)
        with open(graph_path, mode="wb") as graph_file:
            rhasspynlu.graph_to_gzip_pickle(intent_graph, graph_file)

        _LOGGER.debug("Finished writing %s", graph_path)

        # Send to ASR/NLU systems
        has_speech = self.asr_system != "dummy"
        has_intent = self.nlu_system != "dummy"

        if has_speech or has_intent:
            request_id = str(uuid4())

            def handle_train():
                asr_response = None if has_speech else True
                nlu_response = None if has_intent else True

                while True:
                    _, message = yield

                    if isinstance(message, NluTrainSuccess) and (
                        message.id == request_id
                    ):
                        nlu_response = message
                    elif isinstance(message, AsrTrainSuccess) and (
                        message.id == request_id
                    ):
                        asr_response = message
                    if isinstance(message, NluError) and (
                        message.session_id == request_id
                    ):
                        nlu_response = message
                    elif isinstance(message, AsrError) and (
                        message.session_id == request_id
                    ):
                        asr_response = message

                    if asr_response and nlu_response:
                        return [asr_response, nlu_response]

            messages: typing.List[
                typing.Tuple[
                    typing.Union[NluTrain, AsrTrain], typing.Dict[str, typing.Any]
                ],
            ] = []

            message_types: typing.List[typing.Type[Message]] = []

            if has_speech:
                # Request ASR training
                messages.append(
                    (
                        AsrTrain(
                            id=request_id,
                            graph_path=str(graph_path.absolute()),
                            sentences=sentences_dict,
                            slots=slots_dict,
                        ),
                        {"site_id": self.site_id},
                    )
                )
                message_types.extend([AsrTrainSuccess, AsrError])

            if has_intent:
                # Request NLU training
                messages.append(
                    (
                        NluTrain(
                            id=request_id,
                            graph_path=str(graph_path.absolute()),
                            sentences=sentences_dict,
                            slots=slots_dict,
                        ),
                        {"site_id": self.site_id},
                    )
                )
                message_types.extend([NluTrainSuccess, NluError])

            # Expecting only a single result
            result = None
            async for response in self.publish_wait(
                handle_train(),
                messages,
                message_types,
                timeout_seconds=self.training_timeout_seconds,
            ):
                result = response

            # Check result
            assert isinstance(result, list), f"Expected list, got {result}"
            asr_response, nlu_response = result

            if isinstance(asr_response, AsrError):
                _LOGGER.error(asr_response)
                raise TrainingFailedException(reason=asr_response.error)

            if isinstance(nlu_response, NluError):
                _LOGGER.error(nlu_response)
                raise TrainingFailedException(reason=nlu_response.error)

            return result

        return None

    # -------------------------------------------------------------------------

    async def recognize_intent(
        self, text: str, intent_filter: typing.Optional[typing.List[str]] = None
    ) -> typing.Union[NluIntent, NluIntentNotRecognized]:
        """Send an NLU query and wait for intent or not recognized"""
        if self.nlu_system == "dummy":
            raise NluException("No intent system configured")

        nlu_id = str(uuid4())
        query = NluQuery(
            id=nlu_id,
            input=text,
            intent_filter=intent_filter,
            site_id=self.site_id,
            session_id=nlu_id,
        )

        def handle_intent():
            while True:
                _, message = yield

                if isinstance(
                    message, (NluIntent, NluIntentNotRecognized, NluError)
                ) and (message.session_id == nlu_id):
                    return message

        messages = [query]
        message_types: typing.List[typing.Type[Message]] = [
            NluIntent,
            NluIntentNotRecognized,
            NluError,
        ]

        # Expecting only a single result
        result = None
        async for response in self.publish_wait(
            handle_intent(), messages, message_types
        ):
            result = response

        if isinstance(result, NluError):
            _LOGGER.error(result)
            raise NluException(result.error)

        assert isinstance(result, (NluIntent, NluIntentNotRecognized)), result
        return result

    # -------------------------------------------------------------------------

    async def speak_sentence(
        self,
        sentence: str,
        language: typing.Optional[str] = None,
        capture_audio: bool = False,
        wait_play_finished: bool = True,
        site_id: typing.Optional[str] = None,
        session_id: str = "",
    ) -> typing.Tuple[TtsSayFinished, typing.Optional[AudioPlayBytes]]:
        """Speak a sentence using text to speech."""
        if (self.sound_system == "dummy") and (
            not self.satellite_site_ids["text_to_speech"]
        ):
            raise TtsException("No text to speech system configured")

        site_id = site_id or self.site_id
        tts_id = str(uuid4())

        def handle_finished():
            say_finished: typing.Optional[TtsSayFinished] = None
            play_bytes: typing.Optional[
                AudioPlayBytes
            ] = None if capture_audio else True
            play_finished = not wait_play_finished

            while True:
                topic, message = yield

                if isinstance(message, TtsSayFinished) and (message.id == tts_id):
                    say_finished = message
                    play_finished = True
                elif isinstance(message, TtsError):
                    # Assume audio playback didn't happen
                    say_finished = message
                    play_bytes = True
                    play_finished = True
                elif isinstance(message, AudioPlayBytes):
                    request_id = AudioPlayBytes.get_request_id(topic)
                    if request_id == tts_id:
                        play_bytes = message
                elif isinstance(message, AudioPlayError):
                    play_bytes = message

                if say_finished and play_bytes and play_finished:
                    return (say_finished, play_bytes)

        say = TtsSay(id=tts_id, text=sentence, site_id=site_id, session_id=session_id)
        if language:
            say.lang = language

        messages = [say]
        message_types: typing.List[typing.Type[Message]] = [
            TtsSayFinished,
            TtsError,
            AudioPlayBytes,
            AudioPlayError,
        ]

        # Expecting only a single result
        result = None
        async for response in self.publish_wait(
            handle_finished(), messages, message_types
        ):
            result = response

        assert isinstance(result, tuple), f"Expected tuple, got {result}"
        say_response, play_response = result

        if isinstance(say_response, TtsError):
            _LOGGER.error(say_response)
            raise TtsException(say_response.error)

        assert isinstance(say_response, TtsSayFinished), say_response

        if isinstance(play_response, AudioPlayError):
            _LOGGER.error(play_response)
            raise AudioServerException(play_response.error)

        if capture_audio:
            assert isinstance(play_response, AudioPlayBytes), play_response

        return typing.cast(
            typing.Tuple[TtsSayFinished, typing.Optional[AudioPlayBytes]], result
        )

    # -------------------------------------------------------------------------

    async def transcribe_wav(
        self,
        wav_bytes: bytes,
        frames_per_chunk: int = 4096,
        send_audio_captured=True,
        stop_on_silence=False,
        intent_filter: typing.Optional[typing.List[str]] = None,
    ) -> AsrTextCaptured:
        """Transcribe WAV data"""
        if self.asr_system == "dummy":
            raise AsrException("No speech to text system configured")

        session_id = str(uuid4())

        def handle_captured():
            while True:
                _, message = yield

                if isinstance(message, (AsrTextCaptured, AsrError)) and (
                    message.session_id == session_id
                ):
                    return message

        def messages():
            yield AsrStartListening(
                site_id=self.site_id,
                session_id=session_id,
                stop_on_silence=stop_on_silence,
                send_audio_captured=send_audio_captured,
                intent_filter=intent_filter,
            )

            # Break WAV into chunks
            num_bytes_sent: int = 0
            with io.BytesIO(wav_bytes) as wav_buffer:
                for wav_chunk in AudioFrame.iter_wav_chunked(
                    wav_buffer, frames_per_chunk
                ):
                    num_bytes_sent += len(wav_chunk)
                    yield (
                        AudioSessionFrame(wav_bytes=wav_chunk),
                        {"site_id": self.site_id, "session_id": session_id},
                    )

            _LOGGER.debug("Sent %s byte(s) of WAV data", num_bytes_sent)
            yield AsrStopListening(site_id=self.site_id, session_id=session_id)

        message_types: typing.List[typing.Type[Message]] = [AsrTextCaptured, AsrError]

        # Expecting only a single result
        result = None
        async for response in self.publish_wait(
            handle_captured(), messages(), message_types
        ):
            result = response

        if isinstance(result, AsrError):
            _LOGGER.error(result)
            raise AsrException(result.error)

        assert isinstance(result, AsrTextCaptured), result
        return result

    # -------------------------------------------------------------------------

    async def play_wav_data(
        self, wav_bytes: bytes, site_id: typing.Optional[str] = None
    ) -> AudioPlayFinished:
        """Play WAV data through speakers."""
        if self.sound_system == "dummy":
            raise RuntimeError("No audio output system configured")

        site_id = site_id or self.site_id
        request_id = str(uuid4())

        def handle_finished():
            while True:
                _, message = yield

                if (
                    isinstance(message, AudioPlayFinished)
                    and (message.id == request_id)
                ) or isinstance(message, AudioPlayError):
                    return message

        def messages():
            yield (
                AudioPlayBytes(wav_bytes=wav_bytes),
                {"site_id": site_id, "request_id": request_id},
            )

        message_types: typing.List[typing.Type[Message]] = [
            AudioPlayFinished,
            AudioPlayError,
        ]

        # Disable hotword/ASR
        self.publish(
            HotwordToggleOff(site_id=site_id, reason=HotwordToggleReason.PLAY_AUDIO)
        )
        self.publish(AsrToggleOff(site_id=site_id, reason=AsrToggleReason.PLAY_AUDIO))

        try:
            # Expecting only a single result
            result = None
            async for response in self.publish_wait(
                handle_finished(), messages(), message_types
            ):
                result = response

            if isinstance(result, AudioPlayError):
                _LOGGER.error(result)
                raise RuntimeError(result.error)

            assert isinstance(result, AudioPlayFinished)
            return result
        finally:
            # Enable hotword/ASR
            self.publish(
                HotwordToggleOn(site_id=site_id, reason=HotwordToggleReason.PLAY_AUDIO)
            )
            self.publish(
                AsrToggleOn(site_id=site_id, reason=AsrToggleReason.PLAY_AUDIO)
            )

    # -------------------------------------------------------------------------

    async def get_word_pronunciations(
        self, words: typing.Iterable[str], num_guesses: int = 5
    ) -> G2pPhonemes:
        """Look up or guess word phonetic pronunciations."""
        if self.asr_system == "dummy":
            raise G2pException("No speech to text system configured")

        request_id = str(uuid4())

        def handle_finished():
            while True:
                _, message = yield

                if (
                    isinstance(message, G2pPhonemes) and (message.id == request_id)
                ) or isinstance(message, G2pError):
                    return message

        messages = [
            G2pPronounce(
                words=list(words),
                num_guesses=num_guesses,
                id=request_id,
                site_id=self.site_id,
            )
        ]
        message_types: typing.List[typing.Type[Message]] = [G2pPhonemes, G2pError]

        # Expecting only a single result
        result = None
        async for response in self.publish_wait(
            handle_finished(), messages, message_types
        ):
            result = response

        if isinstance(result, G2pError):
            _LOGGER.error(result)
            raise G2pException(result.error)

        assert isinstance(result, G2pPhonemes), result
        return result

    # -------------------------------------------------------------------------

    async def get_microphones(self, test: bool = False) -> AudioDevices:
        """Get available microphones and optionally test them."""
        if self.profile.get("microphone.system", "dummy") == "dummy":
            raise AudioServerException(
                "Microphone disabled. Make sure to save settings and restart."
            )

        request_id = str(uuid4())

        def handle_finished():
            while True:
                _, message = yield

                if isinstance(message, AudioDevices) and (message.id == request_id):
                    return message

        messages = [
            AudioGetDevices(
                id=request_id,
                site_id=self.site_id,
                modes=[AudioDeviceMode.INPUT],
                test=test,
            )
        ]
        message_types: typing.List[typing.Type[Message]] = [AudioDevices]

        # Expecting only a single result
        result = None
        async for response in self.publish_wait(
            handle_finished(), messages, message_types
        ):
            result = response

        assert isinstance(result, AudioDevices), result
        return result

    async def get_speakers(self) -> AudioDevices:
        """Get available speakers."""
        if self.profile.get("sounds.system", "dummy") == "dummy":
            raise AudioServerException(
                "Audio output disabled. Make sure to save settings and restart."
            )

        request_id = str(uuid4())

        def handle_finished():
            while True:
                _, message = yield

                if isinstance(message, AudioDevices) and (message.id == request_id):
                    return message

        messages = [
            AudioGetDevices(
                id=request_id, site_id=self.site_id, modes=[AudioDeviceMode.OUTPUT]
            )
        ]
        message_types: typing.List[typing.Type[Message]] = [AudioDevices]

        # Expecting only a single result
        result = None
        async for response in self.publish_wait(
            handle_finished(), messages, message_types
        ):
            result = response

        assert isinstance(result, AudioDevices), result
        return result

    async def get_hotwords(self) -> Hotwords:
        """Get available hotwords."""
        if self.profile.get("wake.system", "dummy") == "dummy":
            raise HotwordException(
                "Wake word detection disabled. Make sure to save settings and restart."
            )

        request_id = str(uuid4())

        def handle_finished():
            while True:
                _, message = yield

                if isinstance(message, Hotwords) and (message.id == request_id):
                    return message

        messages = [GetHotwords(id=request_id, site_id=self.site_id)]
        message_types: typing.List[typing.Type[Message]] = [Hotwords]

        # Expecting only a single result
        result = None
        async for response in self.publish_wait(
            handle_finished(), messages, message_types
        ):
            result = response

        assert isinstance(result, Hotwords), result
        return result

    async def get_voices(self) -> Voices:
        """Get available voices for text to speech system."""
        if self.profile.get("text_to_speech.system", "dummy") == "dummy":
            raise TtsException(
                "Text to speech disabled. Make sure to save settings and restart."
            )

        request_id = str(uuid4())

        def handle_finished():
            while True:
                _, message = yield

                if isinstance(message, Voices) and (message.id == request_id):
                    return message

        messages = [GetVoices(id=request_id, site_id=self.site_id)]
        message_types: typing.List[typing.Type[Message]] = [Voices]

        # Expecting only a single result
        result = None
        async for response in self.publish_wait(
            handle_finished(), messages, message_types
        ):
            result = response

        assert isinstance(result, Voices), result
        return result

    # -------------------------------------------------------------------------
    # Supporting Functions
    # -------------------------------------------------------------------------

    async def publish_wait(
        self,
        handler,
        messages: typing.Sequence[
            typing.Union[Message, typing.Tuple[Message, typing.Dict[str, typing.Any]]]
        ],
        message_types: typing.List[typing.Type[Message]],
        timeout_seconds: typing.Optional[float] = None,
    ):
        """Publish messages and wait for responses."""
        timeout_seconds = timeout_seconds or self.message_timeout_seconds

        # Start generator
        handler.send(None)

        handler_id = str(uuid4())
        handler_matcher = MQTTMatcher()

        self.handler_matchers[handler_id] = handler_matcher
        self.handler_queues[handler_id] = asyncio.Queue()

        # Subscribe to requested message types
        self.subscribe(*message_types)
        for message_type in message_types:
            # Register handler for each message topic
            handler_matcher[message_type.topic()] = handler

        # Publish messages
        for maybe_message in messages:
            if isinstance(maybe_message, Message):
                # Just a message
                self.publish(maybe_message)
            else:
                # Message and keyword arguments
                message, kwargs = maybe_message
                self.publish(message, **kwargs)

        try:
            # Wait for response or timeout
            result_awaitable = self.handler_queues[handler_id].get()

            if timeout_seconds > 0:
                # With timeout
                _, result = await asyncio.wait_for(
                    result_awaitable, timeout=timeout_seconds
                )
            else:
                # No timeout
                _, result = await result_awaitable

            yield result
        finally:
            # Remove queue
            self.handler_queues.pop(handler_id)

    # -------------------------------------------------------------------------

    def handle_message(self, topic: str, message: Message):
        """Send matching messages to waiting handlers."""
        for handler_id in list(self.handler_matchers):
            handler_matcher = self.handler_matchers[handler_id]

            for handler in handler_matcher.iter_match(topic):
                try:
                    # Run handler
                    _LOGGER.debug(
                        "Handling %s (topic=%s, id=%s)",
                        message.__class__.__name__,
                        topic,
                        handler_id,
                    )

                    try:
                        result = handler.send((topic, message))
                        done = False
                    except StopIteration as e:
                        result = e.value
                        done = True

                    if done:
                        # Message has satistfied handler
                        self.handler_matchers.pop(handler_id)

                    # Signal other thread
                    if done or (result is not None):
                        if handler_id in self.handler_queues:
                            self.loop.call_soon_threadsafe(
                                self.handler_queues[handler_id].put_nowait,
                                (done, result),
                            )
                        else:
                            _LOGGER.warning(
                                "Message queue missing (topic=%s, id=%s)",
                                topic,
                                handler_id,
                            )
                except Exception:
                    _LOGGER.exception("handle_message")

    # -------------------------------------------------------------------------

    def enable_audio_summaries(self):
        """Enable audio summaries on websocket /api/events/audiosummary."""
        self.audio_energies = AudioEnergies()
        self.audio_summaries = True
        self.subscribe(AudioFrame)

    def disable_audio_summaries(self):
        """Disable audio summaries on websocket /api/events/audiosummary."""
        self.audio_summaries = False
        self.unsubscribe(AudioFrame)

    def compute_audio_energies(self, wav_bytes: bytes):
        """Compute energy statistics from audio frame."""
        audio_data = wav_to_buffer(wav_bytes)
        energy = AudioSummary.get_debiased_energy(audio_data)
        self.audio_energies.current_energy = energy
        self.audio_energies.max_energy = max(energy, self.audio_energies.max_energy)
        self.audio_energies.min_energy = min(energy, self.audio_energies.min_energy)
        self.audio_energies.max_current_ratio = self.audio_energies.max_energy / energy

    # -------------------------------------------------------------------------
    # MQTT Event Handlers
    # -------------------------------------------------------------------------

    def on_connect(self, client, userdata, flags, rc):
        """Connected to MQTT broker."""
        try:
            self.is_connected = True
            _LOGGER.debug("Connected to MQTT broker")

            # Default subscriptions
            self.subscribe(
                HotwordDetected,
                AsrTextCaptured,
                NluIntent,
                NluIntentNotRecognized,
                AsrAudioCaptured,
                AudioSummary,
            )

            # Re-subscribe to everything
            for topic in self.all_mqtt_topics:
                self.client.subscribe(topic)
                _LOGGER.debug("Subscribed to %s", topic)

            self.connected_event.set()
        except Exception:
            _LOGGER.exception("on_connect")

    def on_disconnect(self, client, userdata, flags, rc):
        """Automatically reconnect when disconnected."""
        try:
            self.connected_event.clear()
            self.is_connected = False
            _LOGGER.warning("Disconnected. Trying to reconnect...")

            # Clear type/topic caches
            self.subscribed_types.clear()
            self.subscribed_topics.clear()

            self.client.reconnect()
        except Exception:
            _LOGGER.exception("on_disconnect")

    def on_message(self, client, userdata, msg):
        """Received message from MQTT broker."""
        try:
            topic, payload = msg.topic, msg.payload

            for message, site_id, _ in HermesClient.parse_mqtt_message(
                topic, payload, self.subscribed_types
            ):
                is_self_site_id = True
                if site_id and self.site_ids and (site_id not in self.site_ids):
                    # Invalid site id
                    is_self_site_id = False

                if is_self_site_id:
                    # Base-only messages
                    if isinstance(message, AsrAudioCaptured):
                        # Audio data from ASR session
                        assert site_id, "Missing site id"
                        self.last_audio_captured = message
                        self.handle_message(topic, message)
                    elif isinstance(message, AsrError):
                        # ASR service error
                        self.handle_message(topic, message)
                    elif isinstance(message, AsrTextCaptured):
                        # Successful transcription
                        self.handle_message(topic, message)

                        # Report to websockets
                        for queue in self.message_queues:
                            self.loop.call_soon_threadsafe(
                                queue.put_nowait, ("text", message)
                            )
                    elif isinstance(message, AudioDevices):
                        # Microphones or speakers
                        self.handle_message(topic, message)
                    elif isinstance(message, AudioPlayBytes):
                        # Request to play audio
                        assert site_id, "Missing site id"
                        self.handle_message(topic, message)
                    elif isinstance(message, AudioPlayError):
                        # Error playing audio
                        self.handle_message(topic, message)
                    elif isinstance(message, AudioPlayFinished):
                        # Audio finished playing
                        assert site_id, "Missing site id"
                        self.handle_message(topic, message)
                    elif isinstance(message, AudioRecordError):
                        # Error recording audio
                        self.handle_message(topic, message)
                    elif isinstance(message, AudioFrame):
                        # Recorded audio frame
                        assert site_id, "Missing site id"

                        self.compute_audio_energies(message.wav_bytes)

                        # Report to websockets
                        for queue in self.message_queues:
                            self.loop.call_soon_threadsafe(
                                queue.put_nowait, ("audiosummary", self.audio_energies)
                            )
                    elif isinstance(message, DialogueSessionStarted):
                        # Dialogue session started
                        self.handle_message(topic, message)
                    elif isinstance(message, G2pPhonemes):
                        # Word pronunciations
                        self.handle_message(topic, message)
                    elif isinstance(message, Hotwords):
                        # Hotword list
                        self.handle_message(topic, message)
                    elif isinstance(message, HotwordDetected):
                        _LOGGER.debug("<- %s", message)

                        # Hotword detected
                        wakeword_id = HotwordDetected.get_wakeword_id(topic)
                        self.handle_message(topic, message)

                        # Report to websockets
                        for queue in self.message_queues:
                            self.loop.call_soon_threadsafe(
                                queue.put_nowait, ("wake", message, wakeword_id)
                            )

                        # Warn user if they're expected wake -> ASR -> NLU workflow
                        if (self.dialogue_system == "dummy") and (
                            self.asr_system != "dummy"
                        ):
                            _LOGGER.warning(
                                "Dialogue management is disabled. ASR will NOT be automatically enabled."
                            )
                    elif isinstance(message, (HotwordError, HotwordExampleRecorded)):
                        # Other hotword message
                        self.handle_message(topic, message)
                    elif isinstance(message, NluError):
                        # NLU service error
                        self.handle_message(topic, message)
                    elif isinstance(message, NluIntent):
                        _LOGGER.debug("<- %s", message)

                        # Successful intent recognition
                        self.handle_message(topic, message)

                        # Report to websockets
                        for queue in self.message_queues:
                            self.loop.call_soon_threadsafe(
                                queue.put_nowait, ("intent", message)
                            )
                    elif isinstance(message, NluIntentNotRecognized):
                        _LOGGER.debug("<- %s", message)

                        # Failed intent recognition
                        self.handle_message(topic, message)

                        # Report to websockets
                        for queue in self.message_queues:
                            queue.put_nowait(("intent", message))
                    elif isinstance(message, TtsSayFinished):
                        # Text to speech complete
                        self.handle_message(topic, message)
                    elif isinstance(message, AsrTrainSuccess):
                        # ASR training success
                        assert site_id, "Missing site id"
                        self.handle_message(topic, message)
                    elif isinstance(message, NluTrainSuccess):
                        # NLU training success
                        assert site_id, "Missing site id"
                        self.handle_message(topic, message)
                    elif isinstance(message, TtsError):
                        # Error during text to speech
                        self.handle_message(topic, message)
                    elif isinstance(message, Voices):
                        # Text to speech voices
                        self.handle_message(topic, message)
                    else:
                        _LOGGER.warning("Unexpected message: %s", message)
                else:
                    # Check for satellite messages.
                    # This ensures that websocket events are reported on the base
                    # station as well as the satellite.
                    if isinstance(message, (NluIntent, NluIntentNotRecognized)) and (
                        site_id in self.satellite_site_ids["intent"]
                    ):
                        # Report satellite message to base websockets
                        for queue in self.message_queues:
                            self.loop.call_soon_threadsafe(
                                queue.put_nowait, ("intent", message)
                            )
                    elif isinstance(message, AsrTextCaptured) and (
                        site_id in self.satellite_site_ids["speech_to_text"]
                    ):
                        # Report satellite message to base websockets
                        for queue in self.message_queues:
                            self.loop.call_soon_threadsafe(
                                queue.put_nowait, ("text", message)
                            )
                    elif isinstance(message, HotwordDetected) and (
                        site_id in self.satellite_site_ids["wake"]
                    ):
                        # Report satellite message to base websockets
                        wakeword_id = HotwordDetected.get_wakeword_id(topic)
                        for queue in self.message_queues:
                            self.loop.call_soon_threadsafe(
                                queue.put_nowait, ("wake", message, wakeword_id)
                            )
                    elif isinstance(message, (TtsSayFinished, AudioPlayFinished)) and (
                        site_id in self.satellite_site_ids["text_to_speech"]
                    ):
                        # Satellite text to speech/audio finished
                        self.handle_message(topic, message)

            # -----------------------------------------------------------------

            # Forward to external message queues
            for queue in self.message_queues:
                queue.put_nowait(("mqtt", topic, payload))

        except Exception:
            _LOGGER.exception("on_message")

    def subscribe(self, *message_types):
        """Subscribe to one or more Hermes messages."""
        topics: typing.List[str] = []

        if self.site_ids:
            # Specific site_ids
            for site_id in self.site_ids:
                for message_type in message_types:
                    topics.append(message_type.topic(site_id=site_id))
                    self.subscribed_types.add(message_type)
        else:
            # All site_ids
            for message_type in message_types:
                topics.append(message_type.topic())
                self.subscribed_types.add(message_type)

        for topic in topics:
            self.all_mqtt_topics.add(topic)

            # Don't re-subscribe
            if topic not in self.subscribed_topics:
                self.client.subscribe(topic)
                self.subscribed_topics.add(topic)
                _LOGGER.debug("Subscribed to %s", topic)

    def unsubscribe(self, *message_types):
        """Unsubscribe to one or more Hermes messages."""
        topics: typing.List[str] = []

        if self.site_ids:
            # Specific site_ids
            for site_id in self.site_ids:
                for message_type in message_types:
                    topics.append(message_type.topic(site_id=site_id))
                    self.subscribed_types.discard(message_type)
        else:
            # All site_ids
            for message_type in message_types:
                topics.append(message_type.topic())
                self.subscribed_types.discard(message_type)

        for topic in topics:
            self.all_mqtt_topics.discard(topic)

            # Unsubscribe from topic
            self.client.unsubscribe(topic)
            self.subscribed_topics.discard(topic)
            _LOGGER.debug("Unsubscribed from %s", topic)

    def publish(self, message: Message, **topic_args):
        """Publish a Hermes message to MQTT."""
        try:
            topic = message.topic(**topic_args)
            payload = message.payload()

            if message.is_binary_payload():
                # Don't log audio frames
                if not isinstance(message, (AudioFrame, AudioSessionFrame)):
                    _LOGGER.debug(
                        "-> %s(%s byte(s))", message.__class__.__name__, len(payload)
                    )
            else:
                # Log most JSON messages
                if isinstance(message, (AsrTrain, NluTrain)):
                    # Just class name
                    _LOGGER.debug("-> %s", message.__class__.__name__)
                    _LOGGER.debug("Publishing %s bytes(s) to %s", len(payload), topic)
                elif not isinstance(message, AudioSummary):
                    # Entire message
                    _LOGGER.debug("-> %s", message)
                    _LOGGER.debug("Publishing %s bytes(s) to %s", len(payload), topic)

            self.client.publish(topic, payload)
        except Exception:
            _LOGGER.exception("publish")

    # -------------------------------------------------------------------------

    def parse_satellite_site_ids(self, system: str) -> typing.Set[str]:
        """Parse comma-separated satellite ids from profile for a system."""
        return set(
            site_id.strip()
            for site_id in str(
                self.profile.get(f"{system}.satellite_site_ids", "")
            ).split(",")
        )


# -----------------------------------------------------------------------------


class AsrException(Exception):
    """Unexpected error in automated speech recognition."""


class NluException(Exception):
    """Unexpected error in natural language understanding."""


class TtsException(Exception):
    """Unexpected error in text to speech."""


class HotwordException(Exception):
    """Unexpected error in hotword detection."""


class AudioServerException(Exception):
    """Unexpected error in audio playback or recording."""


class G2pException(Exception):
    """Unexpected error in grapheme to phoneme component."""
