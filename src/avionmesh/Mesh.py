import asyncio
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from recsrmesh import CSRMesh
from recsrmesh.mcp import MODEL_OPCODE

logger = logging.getLogger(__name__)

CAPABILITIES = {"dimming": {0, 90, 93, 94, 97, 134, 137, 162}, "color_temp": {0, 93, 134, 137, 162}}
PRODUCT_NAMES = {
    0: "Group",
    90: "Lamp Dimmer",
    93: "Recessed Downlight (RL)",
    94: "Light Adapter",
    97: "Smart Dimmer",
    134: "Smart Bulb (A19)",
    137: "Surface Downlight (BLD)",
    162: "MicroEdge (HLB)",
    167: "Smart Switch",
}


class Verb(Enum):
    WRITE = 0
    READ = 1
    INSERT = 2
    TRUNCATE = 3
    COUNT = 4
    DELETE = 5
    PING = 6
    SYNC = 7
    OTA = 8
    PUSH = 11
    SCAN_WIFI = 12
    CANCEL_DATASTREAM = 13
    UPDATE = 16
    TRIM = 17
    DISCONNECT_OTA = 18
    UNREGISTER = 20
    MARK = 21
    REBOOT = 22
    RESTART = 23
    OPEN_SSH = 32
    NONE = 255


class Noun(Enum):
    DIMMING = 10
    FADE_TIME = 25
    COUNTDOWN = 9
    DATE = 21
    TIME = 22
    SCHEDULE = 7
    GROUPS = 3
    SUNRISE_SUNSET = 6
    ASSOCIATION = 27
    WAKE_STATUS = 28
    COLOR = 29
    CONFIG = 30
    WIFI_NETWORKS = 31
    DIMMING_TABLE = 17
    ASSOCIATED_WIFI_NETWORK = 32
    ASSOCIATED_WIFI_NETWORK_STATUS = 33
    SCENES = 34
    SCHEDULE_2 = 35
    RAB_IP = 36
    RAB_ENV = 37
    RAB_CONFIG = 38
    THERMOMETER = 39
    FIRMWARE_VERSION = 40
    LUX_VALUE = 41
    TEST_MODE = 42
    HARCODED_STRING = 43
    RAB_MARKS = 44
    MOTION_SENSOR = 45
    ALS_DIMMING = 46
    ASSOCIATION_2 = 48
    RTC_SUN_RISE_SET_TABLE = 71
    RTC_DATE = 72
    RTC_TIME = 73
    RTC_DAYLIGHT_SAVING_TIME_TABLE = 74
    AVION_SENSOR = 91
    NONE = 255


def _parse_data(target_id: int, data: bytes) -> Optional[dict]:
    logger.info(f"mesh: parsing data {data!r} from {target_id}")

    if data[0] == 0 and data[1] == 0:
        logger.warning("empty data")
        return None

    try:
        verb = Verb(data[0])
        noun = Noun(data[1])

        if verb == Verb.WRITE:
            target_id = (
                target_id
                if target_id
                else int.from_bytes(bytes([data[2], data[3]]), byteorder="big")
            )
            value_bytes = data[4:]
        else:
            value_bytes = data[2:]

        logger.info(
            f"mesh: target_id({target_id}), verb({verb}), noun({noun}), value:{value_bytes!r})"
        )

        if noun == Noun.DIMMING:
            brightness = int.from_bytes(value_bytes[1:2], byteorder="big")
            return {"avid": target_id, "brightness": brightness}
        elif noun == Noun.COLOR:
            kelvin = int.from_bytes(value_bytes[2:4], byteorder="big")
            return {"avid": target_id, "color_temp": kelvin}
        else:
            logger.warning(f"unknown noun {noun}")
    except Exception as e:
        logger.exception(f"mesh: Exception parsing {data!r} from {target_id}")
        raise e
    return None


def _create_packet(target_id: int, verb: Verb, noun: Noun, value_bytes: bytes) -> tuple[int, bytes]:
    """Returns (dest_id, avi-on payload) for use with CSRMesh.send()."""
    if target_id < 32896:
        group_id = target_id
        dest_id = 0
    else:
        group_id = 0
        dest_id = target_id

    group_bytes = group_id.to_bytes(2, byteorder="big")
    payload = bytes([verb.value, noun.value, group_bytes[0], group_bytes[1], 0, *value_bytes, 0, 0])
    return dest_id, payload


def _get_color_temp_packet(target_id: int, color: int) -> tuple[int, bytes]:
    return _create_packet(
        target_id,
        Verb.WRITE,
        Noun.COLOR,
        bytes([0x01, *bytearray(color.to_bytes(2, byteorder="big"))]),
    )


def _get_brightness_packet(target_id: int, brightness: int) -> tuple[int, bytes]:
    return _create_packet(target_id, Verb.WRITE, Noun.DIMMING, bytes([brightness, 0, 0]))


def _get_date_packet(year: int, month: int, day: int) -> tuple[int, bytes]:
    return _create_packet(0, Verb.WRITE, Noun.DATE, bytearray([year - 2000, month, day]))


def _get_time_packet(hour: int, minute: int, seconds: int) -> tuple[int, bytes]:
    return _create_packet(0, Verb.WRITE, Noun.TIME, bytearray([hour, minute, seconds]))


def apply_overrides_from_settings(settings: dict):
    capabilities_overrides = settings.get("capabilities_overrides")
    if capabilities_overrides is not None:
        dimming_overrides = capabilities_overrides.get("dimming")
        if dimming_overrides is not None:
            for product_id in dimming_overrides:
                CAPABILITIES["dimming"].add(product_id)
        color_temp_overrides = capabilities_overrides.get("color_temp")
        if color_temp_overrides is not None:
            for product_id in color_temp_overrides:
                CAPABILITIES["color_temp"].add(product_id)


class Mesh:
    def __init__(self, csr: CSRMesh) -> None:
        self._csr = csr
        self._notification_callback: Optional[Callable] = None
        # target_id -> (brightness, timestamp)
        self._dimming_commands: dict[int, tuple[int, float]] = {}

    async def read_all(self):
        dest_id, payload = _create_packet(0, Verb.READ, Noun.DIMMING, bytearray(3))
        await self._csr.send(dest_id, MODEL_OPCODE, payload)

    async def set_network_time(self):
        now = datetime.now()
        dest_id, payload = _get_date_packet(now.year, now.month, now.day)
        await self._csr.send(dest_id, MODEL_OPCODE, payload)
        await asyncio.sleep(3)
        now = datetime.now()
        dest_id, payload = _get_time_packet(now.hour, now.minute, now.second)
        await self._csr.send(dest_id, MODEL_OPCODE, payload)

    async def subscribe(self, callback: Callable[[dict], Any]):
        self._notification_callback = callback
        await self.read_all()
        while self._csr.client.is_connected:
            resp = await self._csr.recv(timeout=1.0)
            if resp is not None:
                self._process_response(resp)

    def _process_response(self, resp: dict):
        if resp["opcode"] != MODEL_OPCODE:
            return
        # Broadcast responses (source=0x8000) use crypto_source as device ID
        device_id = resp["crypto_source"] if resp["source"] == 0x8000 else resp["source"]
        parsed = _parse_data(device_id, resp["payload"])
        if not parsed:
            return
        if "brightness" in parsed:
            rapid = self._check_rapid_dimming(parsed["avid"], parsed["brightness"])
            if rapid is not None:
                asyncio.create_task(self._send_brightness_async(parsed["avid"], rapid))
                return
        if self._notification_callback:
            asyncio.create_task(self._notification_callback(parsed))

    def _check_rapid_dimming(self, target_id: int, brightness: int) -> Optional[int]:
        current_time = time.time()

        if target_id in self._dimming_commands:
            prev_brightness, prev_time = self._dimming_commands[target_id]
            time_diff = (current_time - prev_time) * 1000

            if time_diff < 750:
                if brightness == prev_brightness:
                    logger.debug(
                        f"Rapid commands but same brightness ({brightness}) for target_id {target_id}, ignoring"
                    )
                    return None

                is_incrementing = brightness > prev_brightness
                logger.info(
                    f"Rapid dimming detected for target_id {target_id}: "
                    f"{prev_brightness} -> {brightness} ({time_diff:.0f}ms), "
                    f"direction: {'incrementing' if is_incrementing else 'decrementing'}"
                )
                final_brightness = 255 if is_incrementing else 5
                del self._dimming_commands[target_id]
                return final_brightness

        self._dimming_commands[target_id] = (brightness, current_time)
        return None

    async def _send_brightness_async(self, target_id: int, brightness: int):
        try:
            dest_id, payload = _get_brightness_packet(target_id, brightness)
            await self._csr.send(dest_id, MODEL_OPCODE, payload)
            logger.info(f"mesh: Sent brightness {brightness} to {target_id}")
            if self._notification_callback:
                await self._notification_callback({"avid": target_id, "brightness": brightness})
        except Exception as e:
            logger.error(f"Error sending brightness command: {e}")

    async def send_command(self, command_data: dict):
        try:
            avid: int = command_data.get("avid")  # type: ignore
            command = command_data.get("command")
            if command == "read_all":
                await self.read_all()

            elif command == "update":
                json_payload = json.loads(command_data.get("json"))  # type: ignore
                if "brightness" in json_payload:
                    dest_id, payload = _get_brightness_packet(avid, json_payload["brightness"])
                elif "color_temp" in json_payload:
                    dest_id, payload = _get_color_temp_packet(avid, json_payload["color_temp"])
                elif "state" in json_payload:
                    dest_id, payload = _get_brightness_packet(
                        avid, 255 if json_payload["state"] == "ON" else 0
                    )
                else:
                    logger.warning("mesh: Unknown payload")
                    return False

                await self._csr.send(dest_id, MODEL_OPCODE, payload)
                logger.info("mesh: Acknowedging directly")
                parsed = _parse_data(avid, payload)
                if self._notification_callback and parsed:
                    await self._notification_callback(parsed)

            logger.debug(f"Sent command to device {avid}: {command_data}")

        except Exception as e:
            logger.error(f"Error sending command to mesh: {e}")
            raise
