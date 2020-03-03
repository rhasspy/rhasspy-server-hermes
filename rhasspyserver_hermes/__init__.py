"""Hermes implementation of RhasspyCore"""
import asyncio
import io
import json
import logging
import os
import ssl
import typing
import wave
from pathlib import Path
from uuid import uuid4

import aiohttp
import attr
import paho.mqtt.client as mqtt
import rhasspynlu
from paho.mqtt.matcher import MQTTMatcher
from rhasspyhermes.asr import (
    AsrAudioCaptured,
    AsrError,
    AsrStartListening,
    AsrStopListening,
    AsrTextCaptured,
    AsrToggleOff,
    AsrToggleOn,
    AsrTrain,
    AsrTrainSuccess,
)
from rhasspyhermes.audioserver import (
    AudioDeviceMode,
    AudioDevices,
    AudioFrame,
    AudioGetDevices,
    AudioPlayBytes,
    AudioPlayFinished,
    AudioSessionFrame,
)
from rhasspyhermes.base import Message
from rhasspyhermes.dialogue import DialogueSessionStarted
from rhasspyhermes.g2p import G2pPhonemes, G2pPronounce
from rhasspyhermes.nlu import (
    NluError,
    NluIntent,
    NluIntentNotRecognized,
    NluQuery,
    NluTrain,
    NluTrainSuccess,
)
from rhasspyhermes.tts import GetVoices, TtsSay, TtsSayFinished, Voices
from rhasspyhermes.wake import (
    GetHotwords,
    HotwordDetected,
    Hotwords,
    HotwordToggleOff,
    HotwordToggleOn,
)
from rhasspyprofile import Profile

from .train import sentences_to_graph
from .utils import get_ini_paths

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


@attr.s(auto_attribs=True)
class TrainingFailedException(Exception):
    """Raised when training fails."""

    reason: str

    def __str__(self):
        return self.reason


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
        local_mqtt_port: int = 12183,
        siteIds: typing.Optional[typing.List[str]] = None,
        loop=None,
        connection_retries: int = 10,
        retry_seconds: float = 1,
        message_timeout_seconds: float = 30,
        training_timeout_seconds: float = 600,
        certfile: typing.Optional[str] = None,
        keyfile: typing.Optional[str] = None,
    ):
        self.profile_name = profile_name

        if system_profiles_dir:
            system_profiles_dir = Path(system_profiles_dir)

        self.user_profiles_dir = Path(user_profiles_dir)
        self.loop = loop or asyncio.get_event_loop()

        # Default timeout for response messages
        self.message_timeout_seconds = message_timeout_seconds
        self.training_timeout_seconds = training_timeout_seconds

        self.profile = Profile(
            self.profile_name, system_profiles_dir, self.user_profiles_dir
        )

        self.defaults = Profile.load_defaults(self.profile.system_profiles_dir)

        self.client: typing.Optional[mqtt.Client] = None

        if self.profile.get("mqtt.enabled", False):
            # External broker
            self.host = host or self.profile.get("mqtt.host", "localhost")
            self.port = port or self.profile.get("mqtt.port", 1883)
        else:
            # Internal broker
            self.host = host or "localhost"
            self.port = port or local_mqtt_port

        # MQTT retries
        self.connection_retries = connection_retries
        self.retry_seconds = retry_seconds
        self.connect_event = asyncio.Event()

        if siteIds is None:
            # Look up in profile
            siteIds = str(self.profile.get("mqtt.site_id", "")).split(",")

        self.siteIds: typing.List[str] = siteIds or []

        # Site ID to use when publishing
        if self.siteIds:
            self.siteId = self.siteIds[0]
        else:
            self.siteId = "default"

        # System-specific siteIds
        self.asr_siteId = (
            self.profile.get("speech_to_text.hermes.siteId", "") or self.siteId
        )
        self.nlu_siteId = self.profile.get("intent.hermes.siteId", "") or self.siteId
        self.tts_siteId = (
            self.profile.get("text_to_speech.hermes.siteId", "") or self.siteId
        )
        self.sounds_siteId = self.profile.get("sounds.hermes.siteId", "") or self.siteId
        self.wake_siteId = self.profile.get("wake.hermes.siteId", "") or self.siteId
        self.microphone_siteId = (
            self.profile.get("microphone.hermes.siteId", "") or self.siteId
        )
        self.dialogue_siteId = (
            self.profile.get("dialogue.hermes.siteId", "") or self.siteId
        )
        self.handle_siteId = self.profile.get("handle.hermes.siteId", "") or self.siteId

        # Default subscription topics
        self.topics: typing.Set[str] = set(
            [
                HotwordDetected.topic(wakewordId="+"),
                AsrTextCaptured.topic(),
                NluIntentNotRecognized.topic(),
                NluIntent.topic(intentName="#"),
            ]
        )

        if self.siteIds:
            # Specific site ids
            self.topics.update(
                AsrAudioCaptured.topic(siteId=self.asr_siteId, sessionId="+")
                for siteId in self.siteIds
            )
        else:
            # All site ids
            self.topics.add(AsrAudioCaptured.topic(siteId="+", sessionId="+"))

        # id -> matcher
        self.handler_matchers: typing.Dict[str, MQTTMatcher] = {}

        # id -> async queue
        self.handler_queues: typing.Dict[str, asyncio.Queue] = {}

        # External message queues
        self.message_queues: typing.Set[asyncio.Queue] = set()

        # Shared aiohttp client session (enable SSL)
        self.ssl_context = ssl.SSLContext()
        if certfile:
            self.ssl_context.load_cert_chain(certfile, keyfile)

        self._http_session: typing.Optional[
            aiohttp.ClientSession
        ] = aiohttp.ClientSession()

        # Cached wake word IDs for sessions
        self.session_wakewordIds: typing.Dict[str, str] = {}

        # Holds last voice command
        self.last_audio_captured: typing.Optional[AsrAudioCaptured] = None

    @property
    def http_session(self) -> aiohttp.ClientSession:
        """Get HTTP client session."""
        assert self._http_session is not None
        return self._http_session

    # -------------------------------------------------------------------------

    async def start(self):
        """Connect to MQTT broker"""
        _LOGGER.debug("Starting core")

        # Connect to MQTT broker
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.loop_start()

        retries = 0
        connected = False
        while retries < self.connection_retries:
            try:
                _LOGGER.debug(
                    "Connecting to %s:%s (retries: %s)", self.host, self.port, retries
                )
                self.client.connect(self.host, self.port)
                await self.connect_event.wait()
                connected = True
                break
            except Exception:
                _LOGGER.exception("mqtt connect")
                retries += 1
                await asyncio.sleep(self.retry_seconds)

        if not connected:
            _LOGGER.fatal(
                "Failed to connect to MQTT broker (%s:%s)", self.host, self.port
            )
            raise RuntimeError("Failed to connect to MQTT broker")

    async def shutdown(self):
        """Disconnect from MQTT broker"""
        _LOGGER.debug("Shutting down core")

        # Shut down MQTT client
        if self.client:
            client = self.client
            self.client = None
            client.loop_stop()

        # Shut down HTTP session
        if self._http_session is not None:
            await self._http_session.close()
            self._http_session = None

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

        intent_graph = sentences_to_graph(
            sentences_dict,
            slots_dirs=[slots_dir, system_slots_dir],
            slot_programs_dirs=[slot_programs_dir, system_slot_programs_dir],
            language=language,
            word_transform=word_transform,
        )

        # Convert to dictionary
        graph_dict = rhasspynlu.graph_to_json(intent_graph)

        # Save to JSON file
        graph_path = self.profile.write_path(
            self.profile.get("intent.fsticuffs.intent_graph", "intent.json")
        )

        with open(graph_path, "w") as graph_file:
            json.dump(graph_dict, graph_file)

        _LOGGER.debug("Wrote intent graph to %s", str(graph_path))

        # Send to ASR/NLU systems
        speech_system = self.profile.get("speech_to_text.system", "dummy")
        has_speech = speech_system != "dummy"

        intent_system = self.profile.get("intent.system", "dummy")
        has_intent = intent_system != "dummy"

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
                        message.sessionId == request_id
                    ):
                        nlu_response = message
                    elif isinstance(message, AsrError) and (
                        message.sessionId == request_id
                    ):
                        asr_response = message

                    if asr_response and nlu_response:
                        return [asr_response, nlu_response]

            messages: typing.List[
                typing.Tuple[
                    typing.Union[NluTrain, AsrTrain], typing.Dict[str, typing.Any]
                ],
            ] = []

            topics = []

            if has_speech:
                # Request ASR training
                messages.append(
                    (
                        AsrTrain(id=request_id, graph_dict=graph_dict),
                        {"siteId": self.asr_siteId},
                    )
                )
                topics.extend(
                    [AsrTrainSuccess.topic(siteId=self.asr_siteId), AsrError.topic()]
                )

            if has_intent:
                # Request NLU training
                messages.append(
                    (
                        NluTrain(id=request_id, graph_dict=graph_dict),
                        {"siteId": self.nlu_siteId},
                    )
                )
                topics.extend(
                    [NluTrainSuccess.topic(siteId=self.nlu_siteId), NluError.topic()]
                )

            # Expecting only a single result
            result = None
            async for response in self.publish_wait(
                handle_train(),
                messages,
                topics,
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
        self, text: str
    ) -> typing.Union[NluIntent, NluIntentNotRecognized]:
        """Send an NLU query and wait for intent or not recognized"""
        if self.profile.get("intent.system", "dummy") == "dummy":
            _LOGGER.debug("No intent system configured")
            return NluIntentNotRecognized(input="")

        nlu_id = str(uuid4())
        query = NluQuery(id=nlu_id, input=text, siteId=self.nlu_siteId)

        def handle_intent():
            while True:
                _, message = yield

                if isinstance(
                    message, (NluIntent, NluIntentNotRecognized, NluError)
                ) and (message.id == nlu_id):
                    return message

        messages = [query]
        topics = [
            NluIntent.topic(intentName="#"),
            NluIntentNotRecognized.topic(),
            NluError.topic(),
        ]

        # Expecting only a single result
        result = None
        async for response in self.publish_wait(handle_intent(), messages, topics):
            result = response

        if isinstance(result, NluError):
            _LOGGER.debug(result)
            raise RuntimeError(result.error)

        assert isinstance(result, (NluIntent, NluIntentNotRecognized))
        return result

    # -------------------------------------------------------------------------

    async def speak_sentence(
        self, sentence: str, language: typing.Optional[str] = None
    ) -> TtsSayFinished:
        """Speak a sentence using text to speech."""
        if self.profile.get("text_to_speech.system", "dummy") == "dummy":
            _LOGGER.debug("No text to speech system configured")
            return TtsSayFinished()

        tts_id = str(uuid4())

        def handle_finished():
            while True:
                _, message = yield

                if isinstance(message, TtsSayFinished) and (message.id == tts_id):
                    return message

        say = TtsSay(id=tts_id, text=sentence, siteId=self.tts_siteId)
        if language:
            say.lang = language

        messages = [say]
        topics = [TtsSayFinished.topic()]

        # Expecting only a single result
        result = None
        async for response in self.publish_wait(handle_finished(), messages, topics):
            result = response

        assert isinstance(result, TtsSayFinished)

        return result

    # -------------------------------------------------------------------------

    async def transcribe_wav(
        self, wav_bytes: bytes, frames_per_chunk: int = 1024
    ) -> AsrTextCaptured:
        """Transcribe WAV data"""
        if self.profile.get("speech_to_text.system", "dummy") == "dummy":
            _LOGGER.debug("No speech to text system configured")
            return AsrTextCaptured(text="", likelihood=0, seconds=0)

        sessionId = str(uuid4())

        def handle_captured():
            while True:
                _, message = yield

                if isinstance(message, (AsrTextCaptured, AsrError)) and (
                    message.sessionId == sessionId
                ):
                    return message

        def messages():
            yield AsrStartListening(
                siteId=self.asr_siteId,
                sessionId=sessionId,
                stopOnSilence=False,
                sendAudioCaptured=True,
            )

            # Break WAV into chunks
            with io.BytesIO(wav_bytes) as wav_buffer:
                with wave.open(wav_buffer, "rb") as wav_file:
                    frames_left = wav_file.getnframes()
                    while frames_left > 0:
                        with io.BytesIO() as chunk_buffer:
                            chunk_file: wave.Wave_write = wave.open(chunk_buffer, "wb")
                            with chunk_file:
                                chunk_file.setframerate(wav_file.getframerate())
                                chunk_file.setsampwidth(wav_file.getsampwidth())
                                chunk_file.setnchannels(wav_file.getnchannels())
                                chunk_file.writeframes(
                                    wav_file.readframes(frames_per_chunk)
                                )

                            yield (
                                AudioSessionFrame(wav_bytes=chunk_buffer.getvalue()),
                                {"siteId": self.asr_siteId, "sessionId": sessionId},
                            )

                        frames_left -= frames_per_chunk

            yield AsrStopListening(siteId=self.asr_siteId, sessionId=sessionId)

        topics = [AsrTextCaptured.topic(), AsrError.topic()]

        # Expecting only a single result
        result = None
        async for response in self.publish_wait(handle_captured(), messages(), topics):
            result = response

        if isinstance(result, AsrError):
            _LOGGER.debug(result)
            raise RuntimeError(result.error)

        assert isinstance(result, AsrTextCaptured)
        return result

    # -------------------------------------------------------------------------

    async def play_wav_data(self, wav_bytes: bytes) -> AudioPlayFinished:
        """Play WAV data through speakers."""
        if self.profile.get("sounds.system", "dummy") == "dummy":
            _LOGGER.debug("No audio output system configured")
            return AudioPlayFinished()

        requestId = str(uuid4())

        def handle_finished():
            while True:
                _, message = yield

                if isinstance(message, AudioPlayFinished) and (message.id == requestId):
                    return message

        def messages():
            yield (
                AudioPlayBytes(wav_bytes=wav_bytes),
                {"siteId": self.sounds_siteId, "requestId": requestId},
            )

        topics = [AudioPlayFinished.topic(siteId=self.sounds_siteId)]

        # Disable hotword/ASR (local site only)
        if self.wake_siteId == self.siteId:
            self.publish(HotwordToggleOff(siteId=self.wake_siteId))

        if self.asr_siteId == self.siteId:
            self.publish(AsrToggleOff(siteId=self.asr_siteId))

        try:
            # Expecting only a single result
            result = None
            async for response in self.publish_wait(
                handle_finished(), messages(), topics
            ):
                result = response

            assert isinstance(result, AudioPlayFinished)
            return result
        finally:
            # Enable hotword/ASR (local site only)
            if self.wake_siteId == self.siteId:
                self.publish(HotwordToggleOn(siteId=self.wake_siteId))

            if self.asr_siteId == self.siteId:
                self.publish(AsrToggleOn(siteId=self.asr_siteId))

    # -------------------------------------------------------------------------

    async def get_word_pronunciations(
        self, words: typing.Iterable[str], num_guesses: int = 5
    ) -> G2pPhonemes:
        """Look up or guess word phonetic pronunciations."""
        requestId = str(uuid4())

        def handle_finished():
            while True:
                _, message = yield

                if isinstance(message, G2pPhonemes) and (message.id == requestId):
                    return message

        messages = [
            G2pPronounce(
                words=list(words),
                numGuesses=num_guesses,
                id=requestId,
                siteId=self.asr_siteId,
            )
        ]
        topics = [G2pPhonemes.topic()]

        # Expecting only a single result
        result = None
        async for response in self.publish_wait(handle_finished(), messages, topics):
            result = response

        assert isinstance(result, G2pPhonemes)
        return result

    # -------------------------------------------------------------------------

    async def get_microphones(self, test: bool = False) -> AudioDevices:
        """Get available microphones and optionally test them."""
        if self.profile.get("microphone.system", "dummy") == "dummy":
            _LOGGER.warning("Microphone disabled. Cannot get available input devices.")
            return AudioDevices()

        requestId = str(uuid4())

        def handle_finished():
            while True:
                _, message = yield

                if isinstance(message, AudioDevices) and (message.id == requestId):
                    return message

        messages = [
            AudioGetDevices(
                id=requestId,
                siteId=self.microphone_siteId,
                modes=[AudioDeviceMode.INPUT],
                test=test,
            )
        ]
        topics = [AudioDevices.topic()]

        # Expecting only a single result
        result = None
        async for response in self.publish_wait(handle_finished(), messages, topics):
            result = response

        assert isinstance(result, AudioDevices)
        return result

    async def get_speakers(self) -> AudioDevices:
        """Get available speakers."""
        if self.profile.get("sounds.system", "dummy") == "dummy":
            _LOGGER.warning("Speakers disabled. Cannot get available output devices.")
            return AudioDevices()

        requestId = str(uuid4())

        def handle_finished():
            while True:
                _, message = yield

                if isinstance(message, AudioDevices) and (message.id == requestId):
                    return message

        messages = [
            AudioGetDevices(
                id=requestId, siteId=self.sounds_siteId, modes=[AudioDeviceMode.OUTPUT]
            )
        ]
        topics = [AudioDevices.topic()]

        # Expecting only a single result
        result = None
        async for response in self.publish_wait(handle_finished(), messages, topics):
            result = response

        assert isinstance(result, AudioDevices)
        return result

    async def get_hotwords(self) -> Hotwords:
        """Get available hotwords."""
        if self.profile.get("wake.system", "dummy") == "dummy":
            _LOGGER.warning(
                "Wake word detection disabled. Cannot get available wake words."
            )
            return Hotwords()

        requestId = str(uuid4())

        def handle_finished():
            while True:
                _, message = yield

                if isinstance(message, Hotwords) and (message.id == requestId):
                    return message

        messages = [GetHotwords(id=requestId, siteId=self.wake_siteId)]
        topics = [Hotwords.topic()]

        # Expecting only a single result
        result = None
        async for response in self.publish_wait(handle_finished(), messages, topics):
            result = response

        assert isinstance(result, Hotwords)
        return result

    async def get_voices(self) -> Voices:
        """Get available voices for text to speech system."""
        if self.profile.get("text_to_speech.system", "dummy") == "dummy":
            _LOGGER.warning("Text to speech disabled. Cannot get available voices.")
            return Voices()

        requestId = str(uuid4())

        def handle_finished():
            while True:
                _, message = yield

                if isinstance(message, Voices) and (message.id == requestId):
                    return message

        messages = [GetVoices(id=requestId, siteId=self.tts_siteId)]
        topics = [Voices.topic()]

        # Expecting only a single result
        result = None
        async for response in self.publish_wait(handle_finished(), messages, topics):
            result = response

        assert isinstance(result, Voices)
        return result

    # -------------------------------------------------------------------------

    async def maybe_play_sound(self, sound_name: str):
        """Play WAV sound through audio out if it exists."""
        sound_system = self.profile.get("sounds.system", "dummy")
        if sound_system == "dummy":
            # No feedback sounds
            _LOGGER.debug("Sounds diabled (system=%s)", sound_system)
            return

        wav_path_str = os.path.expandvars(self.profile.get(f"sounds.{sound_name}", ""))
        if wav_path_str:
            wav_path = self.profile.read_path(wav_path_str)
            if not wav_path.is_file():
                _LOGGER.error("WAV does not exist: %s", str(wav_path))
                return

            _LOGGER.debug("Playing WAV %s", str(wav_path))
            await self.play_wav_data(wav_path.read_bytes())

    # -------------------------------------------------------------------------
    # Supporting Functions
    # -------------------------------------------------------------------------

    async def publish_wait(
        self,
        handler,
        messages: typing.Sequence[
            typing.Union[Message, typing.Tuple[Message, typing.Dict[str, typing.Any]]]
        ],
        topics: typing.List[str],
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

        # Subscribe to any outstanding topics
        assert self.client
        for topic in topics:
            if topic not in self.topics:
                self.client.subscribe(topic)
                _LOGGER.debug("Subscribed to %s", topic)

            # Register handler for each message topic
            handler_matcher[topic] = handler

        # Publish messages
        for maybe_message in messages:
            if isinstance(maybe_message, Message):
                # Just a message
                self.publish(maybe_message)
            else:
                # Message and keyword arguments
                message, kwargs = maybe_message
                self.publish(message, **kwargs)

        # Wait for response or timeout
        result_awaitable = self.handler_queues[handler_id].get()

        try:
            if timeout_seconds > 0:
                # With timeout
                _, result = await asyncio.wait_for(result_awaitable, timeout_seconds)
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
        if isinstance(message, AsrAudioCaptured):
            _LOGGER.debug(
                "<- %s(%s byte(s))", message.__class__.__name__, len(message.wav_bytes)
            )
        else:
            _LOGGER.debug("<- %s", message)

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
                        self.loop.call_soon_threadsafe(
                            self.handler_queues[handler_id].put_nowait, (done, result)
                        )
                except Exception:
                    _LOGGER.exception("handle_message")

    # -------------------------------------------------------------------------

    def on_connect(self, client, userdata, flags, rc):
        """Connected to MQTT broker."""
        try:
            # Signal start
            self.loop.call_soon_threadsafe(self.connect_event.set)

            for topic in self.topics:
                client.subscribe(topic)
                _LOGGER.debug("Subscribed to %s", topic)
        except Exception:
            _LOGGER.exception("on_connect")

    def on_message(self, client, userdata, msg):
        """Received message from MQTT broker."""
        try:
            _LOGGER.debug("Received %s byte(s) on %s", len(msg.payload), msg.topic)

            if NluIntent.is_topic(msg.topic):
                # Intent from query
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload, self.nlu_siteId):
                    return

                intent = NluIntent.from_dict(json_payload)

                # Report to websockets
                for queue in self.message_queues:
                    self.loop.call_soon_threadsafe(queue.put_nowait, ("intent", intent))

                self.handle_message(msg.topic, intent)
            elif msg.topic == NluIntentNotRecognized.topic():
                # Intent not recognized
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload, self.nlu_siteId):
                    return

                not_recognized = NluIntentNotRecognized.from_dict(json_payload)

                # Report to websockets
                for queue in self.message_queues:
                    self.loop.call_soon_threadsafe(
                        queue.put_nowait, ("intent", not_recognized)
                    )

                # Play error sound
                asyncio.run_coroutine_threadsafe(
                    self.maybe_play_sound("error"), self.loop
                )

                self.handle_message(msg.topic, not_recognized)
            elif msg.topic == TtsSayFinished.topic():
                # Text to speech finished
                json_payload = json.loads(msg.payload)
                self.handle_message(msg.topic, TtsSayFinished.from_dict(json_payload))
            elif msg.topic == AsrTextCaptured.topic():
                # Speech to text result
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload, self.asr_siteId):
                    return

                text_captured = AsrTextCaptured.from_dict(json_payload)

                # Remove cached wake word ID
                wakewordId = self.session_wakewordIds.pop(
                    text_captured.sessionId, "default"
                )

                # Report to websockets
                for queue in self.message_queues:
                    self.loop.call_soon_threadsafe(
                        queue.put_nowait, ("text", text_captured, wakewordId)
                    )

                # Play recorded sound
                asyncio.run_coroutine_threadsafe(
                    self.maybe_play_sound("recorded"), self.loop
                )

                self.handle_message(msg.topic, text_captured)
            elif msg.topic == AudioPlayFinished.topic():
                # Audio has finished playing
                json_payload = json.loads(msg.payload)
                self.handle_message(
                    msg.topic, AudioPlayFinished.from_dict(json_payload)
                )
            elif msg.topic == G2pPhonemes.topic():
                # Grapheme to phoneme result
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload, self.asr_siteId):
                    return

                self.handle_message(msg.topic, G2pPhonemes.from_dict(json_payload))
            elif msg.topic == AudioDevices.topic():
                # Audio device list
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(
                    json_payload, self.microphone_siteId, self.sounds_siteId
                ):
                    return

                self.handle_message(msg.topic, AudioDevices.from_dict(json_payload))
            elif msg.topic == Hotwords.topic():
                # Hotword list
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload, self.wake_siteId):
                    return

                self.handle_message(msg.topic, Hotwords.from_dict(json_payload))
            elif msg.topic == Voices.topic():
                # TTS voices
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload, self.tts_siteId):
                    return

                self.handle_message(msg.topic, Voices.from_dict(json_payload))
            elif AsrTrainSuccess.is_topic(msg.topic):
                # ASR training success
                siteId = AsrTrainSuccess.get_siteId(msg.topic)
                if (
                    (siteId == self.asr_siteId)
                    or (not self.siteIds)
                    or (siteId in self.siteIds)
                ):
                    json_payload = json.loads(msg.payload)
                    self.handle_message(
                        msg.topic, AsrTrainSuccess.from_dict(json_payload)
                    )
            elif AsrAudioCaptured.is_topic(msg.topic):
                # Audio data from ASR session
                siteId = AsrAudioCaptured.get_siteId(msg.topic)
                if (
                    (siteId == self.asr_siteId)
                    or (not self.siteIds)
                    or (siteId in self.siteIds)
                ):
                    self.last_audio_captured = AsrAudioCaptured(wav_bytes=msg.payload)
                    self.handle_message(msg.topic, self.last_audio_captured)
            elif NluTrainSuccess.is_topic(msg.topic):
                # NLU training success
                siteId = NluTrainSuccess.get_siteId(msg.topic)
                if (
                    (siteId == self.nlu_siteId)
                    or (not self.siteIds)
                    or (siteId in self.siteIds)
                ):
                    json_payload = json.loads(msg.payload)
                    self.handle_message(
                        msg.topic, NluTrainSuccess.from_dict(json_payload)
                    )
            elif HotwordDetected.is_topic(msg.topic):
                # Hotword detected
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload, self.wake_siteId):
                    return

                hotword_detected = HotwordDetected.from_dict(json_payload)
                wakewordId = HotwordDetected.get_wakewordId(msg.topic)

                # Cache wake word ID for session
                self.session_wakewordIds[hotword_detected.sessionId] = wakewordId

                # TODO: Report to webhooks

                # Report to websockets
                for queue in self.message_queues:
                    self.loop.call_soon_threadsafe(
                        queue.put_nowait, ("wake", hotword_detected, wakewordId)
                    )

                # Play wake sound
                asyncio.run_coroutine_threadsafe(
                    self.maybe_play_sound("wake"), self.loop
                )

                self.handle_message(msg.topic, hotword_detected)
            elif msg.topic == DialogueSessionStarted.topic():
                # Dialogue session started
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload, self.dialogue_siteId):
                    return

                self.handle_message(
                    msg.topic, DialogueSessionStarted.from_dict(json_payload)
                )
            elif msg.topic == AsrError.topic():
                # ASR service error
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload, self.asr_siteId):
                    return

                self.handle_message(msg.topic, AsrError.from_dict(json_payload))
            elif msg.topic == NluError.topic():
                # NLU service error
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload, self.nlu_siteId):
                    return

                self.handle_message(msg.topic, NluError.from_dict(json_payload))

            # Forward to external message queues
            for queue in self.message_queues:
                self.loop.call_soon_threadsafe(
                    queue.put_nowait, ("mqtt", msg.topic, msg.payload)
                )
        except Exception:
            _LOGGER.exception("on_message")

    def on_disconnect(self, client, userdata, flags, rc):
        """Disconnected from MQTT broker."""
        try:
            if self.client:
                # Automatically reconnect
                _LOGGER.info("Disconnected. Trying to reconnect...")
                self.client.reconnect()
        except Exception:
            _LOGGER.exception("on_disconnect")

    # -------------------------------------------------------------------------

    def publish(self, message: Message, **topic_args):
        """Publish a Hermes message to MQTT."""
        try:
            assert self.client
            topic = message.topic(**topic_args)
            payload: typing.Union[str, bytes]

            if isinstance(message, (AudioFrame, AudioSessionFrame, AudioPlayBytes)):
                # Handle binary payloads specially
                payload = message.wav_bytes
            elif isinstance(message, (AsrTrain, NluTrain)):
                # Don't print entire intent graph to console
                _LOGGER.debug("-> %s(id=%s)", message.__class__.__name__, message.id)
                payload = json.dumps(attr.asdict(message))
            else:
                _LOGGER.debug("-> %s", message)
                payload = json.dumps(attr.asdict(message))

            self.client.publish(topic, payload)
        except Exception:
            _LOGGER.exception("on_message")

    # -------------------------------------------------------------------------

    def _check_siteId(
        self, json_payload: typing.Dict[str, typing.Any], *otherIds
    ) -> bool:
        if self.siteIds:
            payload_siteId = json_payload.get("siteId", "default")

            return (payload_siteId in self.siteIds) or (payload_siteId in otherIds)

        # All sites
        return True
