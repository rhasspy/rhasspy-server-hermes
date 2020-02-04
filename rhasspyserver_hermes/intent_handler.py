"""Methods for handling intents."""
import asyncio
import json
import logging
import os
import ssl
import subprocess
import typing
from urllib.parse import urljoin

import aiohttp

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


async def handle_home_assistant_event(
    intent_dict: typing.Dict[str, typing.Any],
    hass_url: str,
    session: aiohttp.ClientSession,
    event_type_format: str = "rhasspy_{0}",
    pem_file: typing.Optional[str] = None,
    access_token: typing.Optional[str] = None,
    api_password: typing.Optional[str] = None,
) -> typing.Dict[str, typing.Any]:
    """POSTs an event to Home Assistant's /api/events endpoint."""
    if "hass_event" not in intent_dict:
        # Create new Home Assistant event
        event_type = event_type_format.format(intent_dict["intent"]["name"])
        slots = {}
        for entity in intent_dict.get("entities", []):
            slots[entity["entity"]] = entity["value"]

        # Add meta slots
        slots["_text"] = intent_dict.get("text", "")
        slots["_raw_text"] = intent_dict.get("raw_text", "")
        intent_dict["hass_event"] = {"event_type": event_type, "event_data": slots}
    else:
        # Use existing Home Assistant event
        event_type = intent_dict["hass_event"]["event_type"]
        slots = intent_dict["hass_event"]["event_data"]

    # Send event
    post_url = urljoin(hass_url, "api/events/" + event_type)
    ssl_context = None

    if pem_file:
        _LOGGER.debug("Using PEM: %s", pem_file)
        ssl_context = ssl.create_default_context()
        ssl_context.load_cert_chain(pem_file)

    await session.post(post_url, json=slots, ssl=ssl_context, raise_for_status=True)

    return intent_dict


async def handle_home_assistant_intent(
    intent_dict: typing.Dict[str, typing.Any],
    hass_url: str,
    session: aiohttp.ClientSession,
    pem_file: typing.Optional[str] = None,
    access_token: typing.Optional[str] = None,
    api_password: typing.Optional[str] = None,
) -> typing.Dict[str, typing.Any]:
    """POSTs a JSON intent to Home Assistant's /api/intent/handle endpoint."""
    slots = {}
    for entity in intent_dict.get("entities", []):
        slots[entity["entity"]] = entity["value"]

    # Add meta slots
    slots["_text"] = intent_dict.get("text", "")
    slots["_raw_text"] = intent_dict.get("raw_text", "")

    hass_intent = {"name": intent_dict["intent"]["name"], "data": slots}

    # POST intent JSON
    post_url = urljoin(hass_url, "api/intent/handle")
    ssl_context = None

    if pem_file:
        _LOGGER.debug("Using PEM: %s", pem_file)
        ssl.create_default_context()
        ssl_context = ssl_context.load_cert_chain(pem_file)

    await session.post(
        post_url, json=hass_intent, ssl=ssl_context, raise_for_status=True
    )

    return intent_dict


def get_hass_headers(
    access_token: typing.Optional[str] = None, api_password: typing.Optional[str] = None
) -> typing.Dict[str, str]:
    """Gets HTTP authorization headers for Home Assistant POST."""
    if access_token:
        return {"Authorization": access_token}

    if api_password:
        return {"X-HA-Access": api_password}

    hassio_token = os.environ.get("HASSIO_TOKEN")
    if hassio_token:
        return {"Authorization": f"Bearer {hassio_token}"}

    # No headers
    return {}


# -----------------------------------------------------------------------------


async def handle_remote(
    intent_dict: typing.Dict[str, typing.Any],
    remote_url: str,
    session: aiohttp.ClientSession,
) -> typing.Dict[str, typing.Any]:
    """POSTs intent JSON to a remote Rhasspy server."""
    _LOGGER.debug("Handling intent: %s", remote_url)

    async with session.post(
        remote_url, json=intent_dict, raise_for_status=True
    ) as response:
        # Return intent as response
        return await response.json()


# -----------------------------------------------------------------------------


async def handle_command(
    intent_dict: typing.Dict[str, typing.Any],
    handle_program: str,
    handle_arguments: typing.Optional[typing.List[str]] = None,
) -> typing.Dict[str, typing.Any]:
    """Runs external program with intent JSON as input."""
    handle_arguments = handle_arguments or []
    _LOGGER.debug("%s %s", handle_program, handle_arguments)

    command_proc = await asyncio.create_subprocess_exec(
        handle_program, *handle_arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )

    input_bytes = json.dumps(intent_dict).encode()
    command_stdout, _ = await command_proc.communicate(input_bytes)

    _LOGGER.debug("Command returned %s byte(s)", len(command_stdout))

    # Return response as JSON object
    return json.loads(command_stdout.decode())
