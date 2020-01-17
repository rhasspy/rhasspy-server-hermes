"""Hermes implementation of RhasspyCore"""
import asyncio
import io
import json
import logging
import typing
import wave
from pathlib import Path
from uuid import uuid4

import attr
import paho.mqtt.client as mqtt
import rhasspyhermes.intent
import rhasspynlu.intent
from paho.mqtt.matcher import MQTTMatcher
from rhasspyhermes.asr import AsrStartListening, AsrStopListening, AsrTextCaptured
from rhasspyhermes.audioserver import AudioFrame, AudioPlayBytes, AudioPlayFinished
from rhasspyhermes.base import Message
from rhasspyhermes.nlu import NluIntent, NluIntentNotRecognized, NluQuery
from rhasspyhermes.tts import TtsSay, TtsSayFinished
from rhasspyprofile import Profile

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class RhasspyCore:
    """Core Rhasspy functionality, abstracted over MQTT Hermes services"""

    def __init__(
        self,
        profile_name: str,
        system_profiles_dir: typing.Union[str, Path],
        user_profiles_dir: typing.Union[str, Path],
        host: typing.Optional[str] = None,
        port: typing.Optional[int] = None,
        local_mqtt_port: int = 12183,
        siteIds: typing.Optional[typing.List[str]] = None,
        loop=None,
        connection_retries: int = 10,
        retry_seconds: float = 1,
    ):
        self.profile_name = profile_name
        self.system_profiles_dir = Path(system_profiles_dir)
        self.user_profiles_dir = Path(user_profiles_dir)
        self.loop = loop or asyncio.get_event_loop()

        self.profile = Profile(
            self.profile_name, self.system_profiles_dir, self.user_profiles_dir
        )

        self.defaults = Profile.load_defaults(self.system_profiles_dir)

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

        self.siteIds: typing.List[str] = siteIds or []

        # Site ID to use when publishing
        if self.siteIds:
            self.siteId = self.siteIds[0]
        else:
            self.siteId = "default"

        # Default subscription topics
        self.topics: typing.Set[str] = set()

        # id -> matcher
        self.handler_matchers: typing.Dict[str, MQTTMatcher] = {}

        # id -> async queue
        self.handler_queues: typing.Dict[str, asyncio.Queue] = {}

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

        if self.client:
            client = self.client
            self.client = None
            client.loop_stop()

    # -------------------------------------------------------------------------

    async def recognize_intent(self, text: str) -> typing.Dict[str, typing.Any]:
        """Send an NLU query and wait for intent or not recognized"""
        nlu_id = str(uuid4())
        query = NluQuery(id=nlu_id, input=text, siteId=self.siteId)

        def handle_intent():
            while True:
                topic, message = yield

                if isinstance(message, (NluIntent, NluIntentNotRecognized)) and (
                    message.id == nlu_id
                ):
                    if isinstance(message, NluIntent):
                        # Finish parsing
                        message.intent = rhasspyhermes.intent.Intent(**message.intent)
                        message.slots = [
                            rhasspyhermes.intent.Slot(**s) for s in message.slots
                        ]

                    return True, message

        messages = [query]
        topics = [NluIntent.topic(intentName="#"), NluIntentNotRecognized.topic()]

        # Expecting only a single result
        async for result in self.publish_wait(handle_intent(), messages, topics):
            return result

    # -------------------------------------------------------------------------

    async def speak_sentence(
        self, sentence: str, language: typing.Optional[str] = None
    ):
        """Speak a sentence using text to speech."""
        tts_id = str(uuid4())

        def handle_finished():
            while True:
                topic, message = yield

                if isinstance(message, TtsSayFinished) and (message.id == tts_id):
                    return True, None

        say = TtsSay(id=tts_id, text=sentence, siteId=self.siteId)
        if language:
            say.lang = language

        messages = [say]
        topics = [TtsSayFinished.topic()]

        # Expecting only a single result
        async for result in self.publish_wait(handle_finished(), messages, topics):
            return result

    # -------------------------------------------------------------------------

    async def transcribe_wav(
        self, wav_bytes: bytes, frames_per_chunk: int = 1024
    ) -> AsrTextCaptured:
        """Transcribe WAV data"""
        sessionId = str(uuid4())

        def handle_captured():
            while True:
                topic, message = yield

                if isinstance(message, AsrTextCaptured) and (
                    message.sessionId == sessionId
                ):
                    return True, message

        def messages():
            yield AsrStartListening(siteId=self.siteId, sessionId=sessionId)

            # Break WAV into chunks
            with io.BytesIO(wav_bytes) as wav_buffer:
                with wave.open(wav_buffer, "rb") as wav_file:
                    frames_left = wav_file.getnframes()
                    while frames_left > 0:
                        with io.BytesIO() as chunk_buffer:
                            with wave.open(chunk_buffer, "wb") as chunk_file:
                                chunk_file.setframerate(wav_file.getframerate())
                                chunk_file.setsampwidth(wav_file.getsampwidth())
                                chunk_file.setnchannels(wav_file.getnchannels())
                                chunk_file.writeframes(
                                    wav_file.readframes(frames_per_chunk)
                                )

                            yield (
                                AudioFrame(wav_data=chunk_buffer.getvalue()),
                                {"siteId": self.siteId},
                            )

                        frames_left -= frames_per_chunk

            yield AsrStopListening(siteId=self.siteId, sessionId=sessionId)

        topics = [AsrTextCaptured.topic()]

        # Expecting only a single result
        async for result in self.publish_wait(handle_captured(), messages(), topics):
            return result

    # -------------------------------------------------------------------------

    async def play_wav_data(self, wav_data: bytes):
        """Play WAV data through speakers."""
        requestId = str(uuid4())

        def handle_finished():
            while True:
                topic, message = yield

                if (
                    isinstance(message, AudioPlayFinished)
                    and (message.siteId == self.siteId)
                    and (message.id == requestId)
                ):
                    return True, None

        def messages():
            yield (
                AudioPlayBytes(wav_data=wav_data),
                {"siteId": self.siteId, "requestId": requestId},
            )

        topics = [AudioPlayFinished.topic(siteId=self.siteId)]

        # Expecting only a single result
        async for result in self.publish_wait(handle_finished(), messages(), topics):
            return result

    # -------------------------------------------------------------------------

    async def publish_wait(
        self,
        handler,
        messages: typing.List[
            typing.Union[Message, typing.Tuple[Message, typing.Dict[str, typing.Any]]]
        ],
        topics: typing.List[str],
    ):
        """Publish messages and wait for responses."""
        # Start generator
        handler.send(None)

        handler_id = str(uuid4())
        handler_matcher = MQTTMatcher()

        self.handler_matchers[handler_id] = handler_matcher
        self.handler_queues[handler_id] = asyncio.Queue()

        # Subscribe to any outstanding topics
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

        # TODO: Add timeout
        done, result = await self.handler_queues[handler_id].get()
        print(done, result)
        yield result

        if done:
            # Remove queue
            self.handler_queues.pop(handler_id)

    # -------------------------------------------------------------------------

    def handle_message(self, topic: str, message: Message):
        _LOGGER.debug("<- %s", message)

        for handler_id in list(self.handler_matchers):
            handler_matcher = self.handler_matchers[handler_id]

            for handler in handler_matcher.iter_match(topic):
                try:
                    # Run handler
                    _LOGGER.debug(
                        "Handling %s (id=%s)", message.__class__.__name__, handler_id
                    )

                    try:
                        handler.send((topic, message))
                        done, result = next(handler)
                    except StopIteration as e:
                        if e.value:
                            _, result = e.value
                        else:
                            result = None

                        done = True

                    if done:
                        # Message has satistfied handler
                        self.handler_matchers.pop(handler_id)

                    # Signal other thread
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
                if not self._check_siteId(json_payload):
                    return

                self.handle_message(msg.topic, NluIntent(**json_payload))
            elif msg.topic == NluIntentNotRecognized.topic():
                # Intent not recognized
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload):
                    return

                self.handle_message(msg.topic, NluIntentNotRecognized(**json_payload))
            elif msg.topic == TtsSayFinished.topic():
                # Text to speech finished
                json_payload = json.loads(msg.payload)
                self.handle_message(msg.topic, TtsSayFinished(**json_payload))
            elif msg.topic == AsrTextCaptured.topic():
                # Speech to text result
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload):
                    return

                self.handle_message(msg.topic, AsrTextCaptured(**json_payload))
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
            topic = message.topic(**topic_args)
            if isinstance(message, (AudioFrame, AudioPlayBytes)):
                # Handle binary payloads specially
                _LOGGER.debug(
                    "-> %s(%s byte(s)) on %s",
                    message.__class__.__name__,
                    len(message.wav_data),
                    topic,
                )
                payload = message.wav_data
            else:
                _LOGGER.debug("-> %s", message)
                payload = json.dumps(attr.asdict(message))

            _LOGGER.debug("Publishing %s char(s) to %s", len(payload), topic)
            self.client.publish(topic, payload)
        except Exception:
            _LOGGER.exception("on_message")

    # -------------------------------------------------------------------------

    def _check_siteId(self, json_payload: typing.Dict[str, typing.Any]) -> bool:
        if self.siteIds:
            return json_payload.get("siteId", "default") in self.siteIds

        # All sites
        return True
