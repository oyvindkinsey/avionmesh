#!/usr/bin/env python3
"""CSRMesh terminal UI — manage mesh devices from the command line."""

import asyncio
import json
import os
from collections.abc import Callable

from bleak import BleakClient, BleakError, BleakScanner
from recsrmesh import CSRMesh
from recsrmesh.mcp import MODEL_OPCODE

from .Mesh import CAPABILITIES, PRODUCT_NAMES, Mesh, Noun, Verb, _create_packet

try:
    from avionhttp.Http import HTTP_HOST, http_make_request

    HAS_AVIONHTTP = True
except ImportError:
    HAS_AVIONHTTP = False

DB_PATH = os.path.join(os.getcwd(), "mesh_db.json")
OUI_PREFIX = "1C:D6:BD"

# Avi-on device ID ranges (from BluetoothLeService.java)
MIN_DEVICE_ID = 32896
MAX_DEVICE_ID = 65407

MIN_GROUP_ID = 256
MAX_GROUP_ID = 24575


# --- Async input ---


async def ainput(prompt: str = "") -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


# --- Database ---


def load_db() -> dict:
    if os.path.exists(DB_PATH):
        with open(DB_PATH) as f:
            db: dict = json.load(f)
        db.setdefault("groups", [])
        db.setdefault("products", [])
        return db
    return {"passphrase": None, "devices": [], "groups": [], "products": []}


def save_db(db: dict) -> None:
    tmp = DB_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(db, f, indent=2)
        f.write("\n")
    os.replace(tmp, DB_PATH)


def find_device(db: dict, device_id: int) -> dict | None:
    for d in db["devices"]:  # type: dict
        if d["device_id"] == device_id:
            return d
    return None


def upsert_device(db: dict, device: dict) -> None:
    for i, d in enumerate(db["devices"]):
        if d["device_id"] == device["device_id"]:
            db["devices"][i] = device
            return
    db["devices"].append(device)


def remove_device(db: dict, device_id: int) -> None:
    db["devices"] = [d for d in db["devices"] if d["device_id"] != device_id]


def next_device_id(db: dict) -> int:
    """Return next available device ID (max existing + 1, or MIN_DEVICE_ID)."""
    existing = [
        d["device_id"] for d in db["devices"] if MIN_DEVICE_ID <= d["device_id"] <= MAX_DEVICE_ID
    ]
    return int(max(existing)) + 1 if existing else MIN_DEVICE_ID


def find_group(db: dict, group_id: int) -> dict | None:
    for g in db["groups"]:  # type: dict
        if g["group_id"] == group_id:
            return g
    return None


def upsert_group(db: dict, group: dict) -> None:
    for i, g in enumerate(db["groups"]):
        if g["group_id"] == group["group_id"]:
            db["groups"][i] = group
            return
    db["groups"].append(group)


def remove_group(db: dict, group_id: int) -> None:
    db["groups"] = [g for g in db["groups"] if g["group_id"] != group_id]


def next_group_id(db: dict) -> int:
    existing = [
        g["group_id"] for g in db["groups"] if MIN_GROUP_ID <= g["group_id"] <= MAX_GROUP_ID
    ]
    return int(max(existing)) + 1 if existing else MIN_GROUP_ID


def find_product(db: dict, vendor_code: int, product_code: int) -> dict | None:
    for p in db["products"]:  # type: dict
        if p["vendor_code"] == vendor_code and p["product_code"] == product_code:
            return p
    return None


def upsert_product(db: dict, product: dict) -> None:
    for i, p in enumerate(db["products"]):
        if (
            p["vendor_code"] == product["vendor_code"]
            and p["product_code"] == product["product_code"]
        ):
            db["products"][i] = product
            return
    db["products"].append(product)


def parse_fw_version(s: str) -> tuple[int, int, int]:
    parts = s.split(".")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def fw_version_newer(candidate: str, current: str) -> bool:
    """True if candidate is strictly newer than current."""
    return parse_fw_version(candidate) > parse_fw_version(current)


def _model_match(min_len: int = 1, **checks: int) -> Callable[[dict], bool]:
    """Return a match predicate for MODEL_OPCODE responses.

    Args:
        min_len: Minimum payload length.
        **checks: payload index → expected value (e.g. verb=6, key=46).
                  Special key names are mapped: verb→0, noun→1, key→3.
    """
    idx_map = {"verb": 0, "noun": 1, "key": 3}
    constraints = {idx_map.get(k, k): v for k, v in checks.items()}

    def predicate(r):
        if r["opcode"] != MODEL_OPCODE:
            return False
        p = r.get("payload", b"")
        if len(p) < min_len:
            return False
        return all(p[i] == v for i, v in constraints.items())

    return predicate


async def verify_device_id_free(csr: CSRMesh, device_id: int) -> bool:
    """PING a candidate device ID — returns True if no device responds (ID is free)."""
    dest, payload = _create_packet(device_id, Verb.PING, Noun.NONE, b"\x01")
    await csr.send(dest, MODEL_OPCODE, payload)
    responses = await csr.receive(timeout=2.0, match=_model_match(10, verb=6))
    return len(responses) == 0


# --- BLE ---


async def find_bridges(timeout: float = 5.0) -> list[str]:
    """Scan for CSRMesh devices by OUI, sorted by RSSI (strongest first)."""
    devices_advs = await BleakScanner.discover(timeout=timeout, return_adv=True)
    candidates = []
    for addr, (dev, adv) in devices_advs.items():
        if addr.upper().startswith(OUI_PREFIX):
            candidates.append((dev, adv))
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[1].rssi or -999, reverse=True)
    return [c[0].address for c in candidates]


class BLEConnection:
    def __init__(self):
        self._client: BleakClient | None = None
        self._csr: CSRMesh | None = None
        self._mesh: Mesh | None = None
        self._passphrase: str | None = None
        self._bridge: str | None = None
        self._bridge_candidates: list[str] = []

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected

    async def connect(self, passphrase: str, bridge: str | None = None) -> None:
        cached = bridge or self._bridge
        await self.disconnect()
        if cached:
            bridge_addr = cached
            print(f"Reconnecting to {bridge_addr}...")
        else:
            print("Scanning for bridge...")
            self._bridge_candidates = await find_bridges()
            if not self._bridge_candidates:
                raise RuntimeError("No CSRMesh bridge found")
            bridge_addr = self._bridge_candidates[0]
            print(f"Connecting to bridge {bridge_addr}...")
        self._bridge = bridge_addr
        self._client = BleakClient(bridge_addr)
        # Force-disconnect stale BlueZ state from crashed previous sessions
        try:
            await self._client.disconnect()
        except Exception:
            pass
        try:
            await self._client.__aenter__()
        except Exception:
            self._bridge = None
            raise
        self._csr = CSRMesh(self._client, passphrase)
        await self._csr.__aenter__()
        self._mesh = Mesh(self._csr)
        self._passphrase = passphrase
        print(f"Connected to {bridge_addr}")

    async def disconnect(self) -> None:
        self._mesh = None
        if self._csr:
            try:
                await self._csr.__aexit__(None, None, None)
            except Exception:
                pass
            self._csr = None
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass
            self._client = None
        self._passphrase = None

    def alternative_bridges(self) -> list[str]:
        """Return bridge candidates excluding the current one."""
        return [b for b in self._bridge_candidates if b != self._bridge]

    async def ensure_connected(self, passphrase: str) -> tuple[CSRMesh, Mesh]:
        if not (self.is_connected and self._passphrase == passphrase):
            await self.connect(passphrase)
        assert self._csr is not None and self._mesh is not None
        return self._csr, self._mesh


# --- Menu actions ---


async def action_setup_passphrase(db: dict) -> dict:
    passphrase = (await ainput("Enter passphrase: ")).strip()
    if not passphrase:
        print("Cancelled.")
        return db
    db["passphrase"] = passphrase
    save_db(db)
    print("Passphrase saved.")
    return db


async def _cloud_auth(email: str, password: str) -> str:
    """Authenticate with Avi-on cloud, return auth_token."""
    host = HTTP_HOST + "/"
    response = await http_make_request(host, "sessions", {"email": email, "password": password})
    if "credentials" not in response:
        raise RuntimeError("Invalid credentials")
    return str(response["credentials"]["auth_token"])


async def action_import_cloud(db: dict) -> dict:
    if not HAS_AVIONHTTP:
        print("Cloud features require avionhttp.  Install with: pip install avionhttp")
        return db
    email = (await ainput("Avi-on email: ")).strip()
    password = (await ainput("Avi-on password: ")).strip()
    if not email or not password:
        print("Cancelled.")
        return db

    host = HTTP_HOST + "/"
    print("Fetching from Avi-on cloud...")
    auth_token = await _cloud_auth(email, password)

    # Find location matching our passphrase, or import the first one if not set
    response = await http_make_request(host, "user/locations", auth_token=auth_token)
    location_pid = None
    if not db["passphrase"] and response["locations"]:
        # Import passphrase from first location
        raw_loc = response["locations"][0]
        loc = await http_make_request(host, f"locations/{raw_loc['pid']}", auth_token=auth_token)
        db["passphrase"] = loc["location"]["passphrase"]
        location_pid = raw_loc["pid"]
        print(f"Imported passphrase from location: {loc['location'].get('name', '(unnamed)')}")
    else:
        for raw_loc in response["locations"]:
            loc = await http_make_request(
                host, f"locations/{raw_loc['pid']}", auth_token=auth_token
            )
            if loc["location"]["passphrase"] == db["passphrase"]:
                location_pid = raw_loc["pid"]
                break
    if not location_pid:
        print("No location matching current passphrase found.")
        return db

    # Fetch products to build product_id → (vendor_code, product_code) mapping
    products_resp = await http_make_request(host, "products", auth_token=auth_token)
    product_map = {}  # product_id (pid) → {vendor_code, product_code}
    for rp in products_resp.get("products", []):
        pid = rp.get("pid") or rp.get("id")
        if pid is not None:
            product_map[pid] = {
                "vendor_code": rp["vendor_code"],
                "product_code": rp["product_code"],
            }
        # Also upsert into db products
        latest_fw: dict[str, str] = {}
        for fw in rp.get("firmwares", []):
            mcu_id = str(fw.get("micro_controller_identifier", 0))
            version = fw.get("version", "")
            if not version:
                continue
            if mcu_id not in latest_fw or fw_version_newer(version, latest_fw[mcu_id]):
                latest_fw[mcu_id] = version
        upsert_product(
            db,
            {
                "vendor_code": rp["vendor_code"],
                "product_code": rp["product_code"],
                "name": rp.get("name", ""),
                "latest_fw": latest_fw,
            },
        )

    # Fetch devices — versions from cloud, vendor/product via product_map
    response = await http_make_request(
        host, f"locations/{location_pid}/abstract_devices", auth_token=auth_token
    )
    dev_count = 0
    for raw in response["abstract_devices"]:
        if raw.get("type") != "device":
            continue
        mac = raw.get("friendly_mac_address", "")
        if mac:
            it = iter(mac.lower())
            mac = ":".join(a + b for a, b in zip(it, it, strict=False))
        device = {
            "device_id": raw["avid"],
            "name": raw["name"],
            "product_id": raw.get("product_id", 0),
            "mac_address": mac,
        }
        # Firmware versions from cloud
        versions = raw.get("versions") or raw.get("firm_versions")
        if versions and isinstance(versions, list):
            device["firmware"] = {str(i): v for i, v in enumerate(versions) if v and v != "0.0.0"}
        # Map product_id → (vendor_code, product_code) via products catalog
        prod_info = product_map.get(raw.get("product_id"))
        if prod_info:
            device["vendor_id"] = prod_info["vendor_code"]
            device["csr_product_id"] = prod_info["product_code"]
        # Merge with existing record to preserve BLE-scanned fields
        existing = find_device(db, raw["avid"])
        if existing:
            existing.update(device)
        else:
            upsert_device(db, device)
        dev_count += 1

    # Fetch groups
    response = await http_make_request(
        host, f"locations/{location_pid}/groups", auth_token=auth_token
    )
    grp_count = 0
    for raw in response.get("groups", []):
        upsert_group(db, {"group_id": raw["avid"], "name": raw["name"]})
        grp_count += 1

    save_db(db)
    print(f"Imported {dev_count} device(s), {grp_count} group(s).")
    return db


async def action_scan_claim(db: dict, conn: BLEConnection) -> dict:
    if not db["passphrase"]:
        print("Set passphrase first (option 1).")
        return db

    csr, _ = await conn.ensure_connected(db["passphrase"])
    print("Scanning for unclaimed devices (5s)...")
    found = await csr.discover_unassociated(timeout=5.0)
    if not found:
        print("No unclaimed devices found.")
        return db

    print(f"\nFound {len(found)} unclaimed device(s):")
    for i, dev in enumerate(found):
        print(f"  {i + 1}. uuid_hash=0x{dev['uuid_hash']:08x}")

    choice = (await ainput("Pick device # (or Enter to cancel): ")).strip()
    if not choice:
        return db
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(found):
            raise ValueError
    except ValueError:
        print("Invalid choice.")
        return db

    picked = found[idx]
    uuid_hash = picked["uuid_hash"]

    candidate_id = next_device_id(db)
    print(f"Verifying device_id {candidate_id} is free...")
    if not await verify_device_id_free(csr, candidate_id):
        print(f"  device_id {candidate_id} is already in use on the mesh!")
        return db

    print(f"Associating 0x{uuid_hash:08x} as device_id={candidate_id}...")
    try:
        device_id = await csr.associate(uuid_hash, candidate_id)
    except RuntimeError as e:
        if "WAIT_DEVICE_ID_ACK" not in str(e):
            raise
        # Self-bridging: current bridge is likely the target
        alternatives = conn.alternative_bridges()
        if not alternatives:
            print("  DEVICE_ID_ACK timeout and no alternative bridge available.")
            raise
        device_id = None
        for alt in alternatives:
            print(f"  DEVICE_ID_ACK timeout — switching to bridge {alt}...")
            await conn.connect(db["passphrase"], bridge=alt)
            assert conn._csr is not None
            csr = conn._csr
            try:
                device_id = await csr.associate(uuid_hash, candidate_id)
                break
            except RuntimeError as e2:
                if "WAIT_DEVICE_ID_ACK" not in str(e2):
                    raise
        if device_id is None:
            print("  All bridges exhausted.")
            return db
    print(f"Assigned device_id={device_id}")

    name = (await ainput("Name for this device: ")).strip() or f"Device {device_id}"
    device_record = {
        "device_id": device_id,
        "name": name,
        "product_id": 0,
        "mac_address": "",
    }
    # Extract vendor/product identity from device UUID
    uuid_bytes = picked.get("uuid")
    if uuid_bytes and len(uuid_bytes) >= 16:
        identity = _parse_uuid_identity(uuid_bytes)
        device_record["vendor_id"] = identity["vendor_id"]
        device_record["csr_product_id"] = identity["csr_product_id"]
    upsert_device(db, device_record)
    save_db(db)
    print(f"Saved '{name}' (id={device_id}).")
    return db


def _parse_uuid_identity(uuid_bytes: bytes) -> dict:
    """Extract vendor_id and csr_product_id from device UUID (16 bytes).

    Java UUID layout: bytes 0-7 = mostSignificantBits, bytes 8-15 = leastSignificantBits.
    product_code at MSB[6] = uuid_bytes[6], vendor at LSB[0:2] = uuid_bytes[8:10].
    """
    return {
        "vendor_id": int.from_bytes(uuid_bytes[8:10], "big"),
        "csr_product_id": uuid_bytes[6],
    }


def _parse_ping(resp: dict) -> dict | None:
    """Extract device info from a PING response dict."""
    p = resp["payload"]
    if len(p) < 10 or p[0] != 6:
        return None
    device_id = resp.get("crypto_source") or resp.get("source")
    return {
        "device_id": device_id,
        "fw": f"{p[3]}.{p[4]}.{p[5]}",
        "flags": p[6],
        "vendor_id": int.from_bytes(p[7:9], "big"),
        "csr_product_id": p[9],
    }


async def fetch_cloud_products(email: str, password: str) -> list[dict]:
    """Fetch product catalog from Avi-on cloud, return normalized product list."""
    host = HTTP_HOST + "/"
    auth_token = await _cloud_auth(email, password)
    response = await http_make_request(host, "products", auth_token=auth_token)
    raw_products = response.get("products", [])

    result = []
    for rp in raw_products:
        latest_fw: dict[str, str] = {}
        for fw in rp.get("firmwares", []):
            mcu_id = str(fw.get("micro_controller_identifier", 0))
            version = fw.get("version", "")
            if not version:
                continue
            if mcu_id not in latest_fw or fw_version_newer(version, latest_fw[mcu_id]):
                latest_fw[mcu_id] = version
        result.append(
            {
                "vendor_code": rp["vendor_code"],
                "product_code": rp["product_code"],
                "name": rp.get("name", ""),
                "latest_fw": latest_fw,
            }
        )
    return result


async def scan_device_firmware(csr: CSRMesh, device_id: int) -> dict | None:
    """Read firmware info from a device via BLE. Returns vendor/product/fw dict or None."""
    # PING for vendor_id, csr_product_id — switches may not respond
    dest, payload = _create_packet(device_id, Verb.PING, Noun.NONE, b"\x01")
    await csr.send(dest, MODEL_OPCODE, payload)
    responses = await csr.receive(timeout=2.0, match=_model_match(10, verb=6))
    ping = _parse_ping(responses[0]) if responses else None

    firmware = {}
    # READ FIRMWARE_VERSION MCU 0
    dest, payload = _create_packet(device_id, Verb.READ, Noun.FIRMWARE_VERSION, b"\x00")
    await csr.send(dest, MODEL_OPCODE, payload)
    responses = await csr.receive(timeout=2.0, match=_model_match(8))
    if responses:
        p = responses[0]["payload"]
        firmware["0"] = f"{p[5]}.{p[6]}.{p[7]}"

    # READ FIRMWARE_VERSION MCU 1
    dest, payload = _create_packet(device_id, Verb.READ, Noun.FIRMWARE_VERSION, b"\x01")
    await csr.send(dest, MODEL_OPCODE, payload)
    responses = await csr.receive(timeout=2.0, match=_model_match(8))
    if responses:
        p = responses[0]["payload"]
        firmware["1"] = f"{p[5]}.{p[6]}.{p[7]}"

    if not ping and not firmware:
        return None

    result = {"firmware": firmware}
    if ping:
        result["vendor_id"] = ping["vendor_id"]
        result["csr_product_id"] = ping["csr_product_id"]
    return result


async def action_discover_mesh(db: dict, conn: BLEConnection) -> dict:
    if not db["passphrase"]:
        print("Set passphrase first (option 1).")
        return db

    csr, _ = await conn.ensure_connected(db["passphrase"])
    known_ids = {d["device_id"] for d in db["devices"]}

    # Broadcast PING to all mesh devices
    print("Broadcasting PING to mesh (5s)...")
    dest, payload = _create_packet(0, Verb.PING, Noun.NONE, b"\x01")
    await csr.send(dest, MODEL_OPCODE, payload)
    responses = await csr.receive(timeout=5.0)

    seen = {}
    for r in responses:
        if r["opcode"] != MODEL_OPCODE:
            continue
        info = _parse_ping(r)
        if info and info["device_id"] not in seen:
            seen[info["device_id"]] = info

    if not seen:
        print("No devices responded.")
        return db

    unknown = {did: info for did, info in seen.items() if did not in known_ids}
    known_found = {did: info for did, info in seen.items() if did in known_ids}

    if known_found:
        print(f"\nKnown devices ({len(known_found)}):")
        for did, info in sorted(known_found.items()):
            dev = find_device(db, did)
            name = dev["name"] if dev else f"Device {did}"
            print(f"  {name} (id={did})  fw={info['fw']}")

    if not unknown:
        print(f"\nAll {len(seen)} responding device(s) are already in the database.")
        return db

    print(f"\nUnknown devices ({len(unknown)}):")
    unknown_list = sorted(unknown.values(), key=lambda x: x["device_id"])
    for i, info in enumerate(unknown_list):
        print(
            f"  {i + 1}. id={info['device_id']}  fw={info['fw']}  "
            f"vendor=0x{info['vendor_id']:04x}  csr_product={info['csr_product_id']}"
        )

    choice = (await ainput("Pick device # to add (or Enter to cancel): ")).strip()
    if not choice:
        return db
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(unknown_list):
            raise ValueError
    except ValueError:
        print("Invalid choice.")
        return db

    picked = unknown_list[idx]
    did = picked["device_id"]

    # Direct PING for confirmation
    print(f"Pinging device {did} directly...")
    dest, payload = _create_packet(did, Verb.PING, Noun.NONE, b"\x01")
    await csr.send(dest, MODEL_OPCODE, payload)
    responses = await csr.receive(timeout=2.0, match=_model_match(10, verb=6))
    confirmed = _parse_ping(responses[0]) if responses else None
    if not confirmed:
        print("  No response to direct PING — device may be unreachable.")
        confirm = (await ainput("  Add anyway with broadcast info? (y/N): ")).strip().lower()
        if confirm != "y":
            return db
        confirmed = picked

    # Read MAC address: READ CONFIG key=46, MAC at response payload[4:10]
    mac_str = ""
    print("  Reading MAC address...")
    dest, payload = _create_packet(did, Verb.READ, Noun.CONFIG, bytes([46]))
    await csr.send(dest, MODEL_OPCODE, payload)
    responses = await csr.receive(timeout=2.0, match=_model_match(10, key=46))
    if responses:
        mac_bytes = responses[0]["payload"][4:10]
        mac_str = ":".join(f"{b:02X}" for b in mac_bytes)

    print(f"  Device ID:   {confirmed['device_id']}")
    print(f"  Firmware:    {confirmed['fw']}")
    print(f"  Vendor ID:   0x{confirmed['vendor_id']:04x}")
    print(f"  CSR product: {confirmed['csr_product_id']}")
    print(f"  MAC address: {mac_str or '(no response)'}")

    name = (await ainput("Name for this device: ")).strip() or f"Device {did}"
    upsert_device(
        db,
        {
            "device_id": did,
            "name": name,
            "product_id": 0,
            "mac_address": mac_str,
        },
    )
    save_db(db)
    print(f"Saved '{name}' (id={did}).")
    return db


async def read_device_groups(csr: CSRMesh, device_id: int) -> list[int]:
    """Read group memberships from device by querying each slot."""
    dest, payload = _create_packet(device_id, Verb.COUNT, Noun.GROUPS, b"")
    await csr.send(dest, MODEL_OPCODE, payload)
    responses = await csr.receive(timeout=2.0, match=_model_match(3))
    if not responses:
        return []
    capacity = responses[0]["payload"][-1]

    group_ids = []
    for slot in range(capacity):
        dest, payload = _create_packet(device_id, Verb.READ, Noun.GROUPS, bytes([slot]))
        await csr.send(dest, MODEL_OPCODE, payload)
        responses = await csr.receive(timeout=2.0, match=_model_match(4))
        if not responses:
            continue
        p = responses[0]["payload"]
        # Response: [verb, noun, ..., group_id_hi, group_id_lo]
        # Per GROUPS.md: GROUPINDEX at byte offset 1 after verb/noun, GROUPID at last 2 bytes
        if len(p) >= 4:
            gid = int.from_bytes(p[-2:], "big")
            if gid != 0:
                group_ids.append(gid)
        else:
            print(f"  Slot {slot} unexpected response: {p.hex(' ')}")
    return group_ids


async def action_examine_device(csr: CSRMesh, device: dict, db: dict) -> None:
    avid = device["device_id"]
    print(f"\n=== Device Examination: {device['name']} (id={avid}) ===\n")

    # --- PING (product info) ---
    print("PING response:")
    dest, payload = _create_packet(avid, Verb.PING, Noun.NONE, b"\x01")
    await csr.send(dest, MODEL_OPCODE, payload)
    responses = await csr.receive(timeout=2.0, match=_model_match(10, verb=6))
    ping_resp = responses[0] if responses else None
    if ping_resp:
        # Format: [verb=6, noun=0xFF, 0x00, fw_major, fw_minor, fw_patch, flags, vendor_hi, vendor_lo, csr_product_id]
        # Device ID comes from MCP source, not payload
        # CSR product_id ≠ Avi-on catalog product_id (e.g. CSR 18 = Avi-on 162)
        ping = ping_resp["payload"]
        mcp_source = ping_resp.get("crypto_source") or ping_resp.get("source")
        csr_product_id = ping[9]
        vendor_id = int.from_bytes(ping[7:9], "big")
        print(f"  Device ID:   {mcp_source}")
        print(f"  Firmware:    {ping[3]}.{ping[4]}.{ping[5]}")
        print(f"  Flags:       0x{ping[6]:02x}")
        print(f"  Vendor ID:   0x{vendor_id:04x} ({vendor_id})")
        print(f"  CSR product: {csr_product_id}  (Avi-on catalog: {device.get('product_id', '?')})")
    else:
        print("  (no response)")

    # --- Firmware version (main MCU) ---
    # Response: [verb=0x0B(PUSH), noun=0x28(FIRMWARE_VERSION), 0x00, mcu_id, ??, major, minor, patch]
    print("\nFirmware (main MCU, id=0):")
    dest, payload = _create_packet(avid, Verb.READ, Noun.FIRMWARE_VERSION, b"\x00")
    await csr.send(dest, MODEL_OPCODE, payload)
    responses = await csr.receive(timeout=2.0, match=_model_match(8))
    fw_main = responses[0]["payload"] if responses else None
    if fw_main:
        print(f"  MCU id:  {fw_main[3]}")
        print(f"  Version: {fw_main[5]}.{fw_main[6]}.{fw_main[7]}")
    else:
        print("  (no response)")

    # --- Firmware version (custom MCU) ---
    print("\nFirmware (custom MCU, id=1):")
    dest, payload = _create_packet(avid, Verb.READ, Noun.FIRMWARE_VERSION, b"\x01")
    await csr.send(dest, MODEL_OPCODE, payload)
    responses = await csr.receive(timeout=2.0, match=_model_match(8))
    fw_custom = responses[0]["payload"] if responses else None
    if fw_custom:
        print(f"  MCU id:  {fw_custom[3]}")
        print(f"  Version: {fw_custom[5]}.{fw_custom[6]}.{fw_custom[7]}")
    else:
        print("  (no response — single MCU device)")

    # --- Firmware update status ---
    latest = _get_latest_fw(db, device)
    if latest:
        updates = []
        if fw_main:
            ver = f"{fw_main[5]}.{fw_main[6]}.{fw_main[7]}"
            lat = latest.get("0")
            if lat and fw_version_newer(lat, ver):
                updates.append(f"MCU 0: {ver} -> {lat}")
        if fw_custom:
            ver = f"{fw_custom[5]}.{fw_custom[6]}.{fw_custom[7]}"
            lat = latest.get("1")
            if lat and fw_version_newer(lat, ver):
                updates.append(f"MCU 1: {ver} -> {lat}")
        if updates:
            print("\nFirmware update available:")
            for u in updates:
                print(f"  {u}")
        elif fw_main or fw_custom:
            print("\nFirmware: up to date")

    # --- Groups ---
    print("\nGroups:")
    group_ids = await read_device_groups(csr, avid)
    if group_ids:
        for gid in group_ids:
            g = find_group(db, gid)
            label = f' "{g["name"]}"' if g else ""
            print(f"  - {gid}{label}")
    else:
        print("  (none)")

    # --- Current state: dimming ---
    print("\nCurrent state:")
    dest, payload = _create_packet(avid, Verb.READ, Noun.DIMMING, b"\x00\x00\x00")
    await csr.send(dest, MODEL_OPCODE, payload)
    responses = await csr.receive(timeout=2.0, match=_model_match(4))
    if responses:
        print(f"  Brightness: {responses[0]['payload'][3]}")
    else:
        print("  Brightness: (no response)")

    # --- Current state: color temp ---
    dest, payload = _create_packet(avid, Verb.READ, Noun.COLOR, b"\x00\x00\x00")
    await csr.send(dest, MODEL_OPCODE, payload)
    responses = await csr.receive(timeout=2.0, match=_model_match(6))
    if responses:
        kelvin = int.from_bytes(responses[0]["payload"][4:6], "big")
        print(f"  Color temp: {kelvin}K" if kelvin else "  Color temp: 0 (not supported or off)")
    else:
        print("  Color temp: (no response)")

    # --- MAC address: READ CONFIG key=46, MAC at payload[4:10] ---
    print("\nMAC address:")
    dest, payload = _create_packet(avid, Verb.READ, Noun.CONFIG, bytes([46]))
    await csr.send(dest, MODEL_OPCODE, payload)
    responses = await csr.receive(timeout=2.0, match=_model_match(10, key=46))
    if responses:
        mac_bytes = responses[0]["payload"][4:10]
        print(f"  {':'.join(f'{b:02X}' for b in mac_bytes)}")
    else:
        print("  (no response)")


async def _device_group_menu(db: dict, conn: BLEConnection, csr: CSRMesh, device: dict) -> None:
    avid = device["device_id"]
    while True:
        print(f"\nReading groups on {device['name']}...")
        group_ids = await read_device_groups(csr, avid)

        print(f"\nGroups on {device['name']}:")
        if group_ids:
            for gid in group_ids:
                g = find_group(db, gid)
                label = f' "{g["name"]}"' if g else ""
                print(f"  - {gid}{label}")
        else:
            print("  (none)")

        print("\n1. Add to group")
        print("2. Remove from group")
        print("3. Clear all groups")
        print("4. Back")

        choice = (await ainput("> ")).strip()
        try:
            if choice == "1":
                if db["groups"]:
                    print("Known groups:")
                    for i, grp in enumerate(db["groups"]):
                        print(f'  {i + 1}. Group {grp["group_id"]} "{grp["name"]}"')
                    pick = (await ainput("Group # or raw group ID (or Enter to cancel): ")).strip()
                    if not pick:
                        continue
                    val = int(pick)
                    if 1 <= val <= len(db["groups"]):
                        gid = db["groups"][val - 1]["group_id"]
                    else:
                        gid = val
                else:
                    pick = (await ainput("Group ID (or Enter to cancel): ")).strip()
                    if not pick:
                        continue
                    gid = int(pick)

                gid_bytes = gid.to_bytes(2, "big")
                dest, payload = _create_packet(avid, Verb.INSERT, Noun.GROUPS, gid_bytes)
                await csr.send(dest, MODEL_OPCODE, payload)
                await csr.receive(timeout=2.0)
                g = find_group(db, gid)
                label = f' "{g["name"]}"' if g else ""
                print(f"Sent INSERT GROUPS {gid}{label}.")

            elif choice == "2":
                if not group_ids:
                    print("No groups to remove.")
                    continue
                print("Current groups:")
                for i, gid in enumerate(group_ids):
                    g = find_group(db, gid)
                    label = f' "{g["name"]}"' if g else ""
                    print(f"  {i + 1}. {gid}{label}")
                pick = (await ainput("Group # to remove (or Enter to cancel): ")).strip()
                if not pick:
                    continue
                idx = int(pick) - 1
                if idx < 0 or idx >= len(group_ids):
                    raise ValueError
                gid = group_ids[idx]
                gid_bytes = gid.to_bytes(2, "big")
                dest, payload = _create_packet(avid, Verb.DELETE, Noun.GROUPS, gid_bytes)
                await csr.send(dest, MODEL_OPCODE, payload)
                await csr.receive(timeout=2.0)
                print(f"Sent DELETE GROUPS {gid}.")

            elif choice == "3":
                confirm = (await ainput("Clear ALL groups? (y/N): ")).strip().lower()
                if confirm != "y":
                    continue
                dest, payload = _create_packet(avid, Verb.TRUNCATE, Noun.GROUPS, b"")
                await csr.send(dest, MODEL_OPCODE, payload)
                await csr.receive(timeout=2.0)
                print("Sent TRUNCATE GROUPS.")

            elif choice == "4":
                return

        except (BleakError, RuntimeError, TimeoutError) as e:
            print(f"Error: {e}")
            await conn.disconnect()
        except (ValueError, IndexError):
            print("Invalid input.")
        except EOFError:
            return


async def device_control_menu(db: dict, conn: BLEConnection, device: dict) -> dict:
    avid = device["device_id"]
    pid = device.get("product_id", 0)
    has_color = pid in CAPABILITIES.get("color_temp", set())

    while True:
        print(f"\n=== {device['name']} (id={avid}) ===")
        print("1. Set brightness (0-255)")
        if has_color:
            print("2. Set color temp (2700-6500K)")
        print("3. Turn ON")
        print("4. Turn OFF")
        print("5. Manage groups")
        print("6. Examine device")
        print("7. Unclaim (remove from mesh)")
        print("8. Back")

        choice = (await ainput("> ")).strip()
        try:
            csr, mesh = await conn.ensure_connected(db["passphrase"])

            if choice == "1":
                val = int((await ainput("Brightness (0-255): ")).strip())
                await mesh.send_command(
                    {"avid": avid, "command": "update", "json": json.dumps({"brightness": val})}
                )
                print(f"Brightness -> {val}")

            elif choice == "2" and has_color:
                val = int((await ainput("Color temp (2700-6500): ")).strip())
                await mesh.send_command(
                    {"avid": avid, "command": "update", "json": json.dumps({"color_temp": val})}
                )
                print(f"Color temp -> {val}K")

            elif choice == "3":
                await mesh.send_command(
                    {"avid": avid, "command": "update", "json": json.dumps({"state": "ON"})}
                )
                print("ON")

            elif choice == "4":
                await mesh.send_command(
                    {"avid": avid, "command": "update", "json": json.dumps({"state": "OFF"})}
                )
                print("OFF")

            elif choice == "5":
                await _device_group_menu(db, conn, csr, device)

            elif choice == "6":
                await action_examine_device(csr, device, db)

            elif choice == "7":
                confirm = (await ainput(f"Unclaim '{device['name']}'? (y/N): ")).strip().lower()
                if confirm == "y":
                    await csr.disassociate(avid)
                    remove_device(db, avid)
                    save_db(db)
                    print("Device unclaimed and removed.")
                    return db

            elif choice == "8":
                return db

        except (BleakError, RuntimeError, TimeoutError) as e:
            print(f"Error: {e}")
            await conn.disconnect()
        except (ValueError, IndexError):
            print("Invalid input.")
        except EOFError:
            return db


async def action_manage_groups(db: dict, conn: BLEConnection) -> dict:
    while True:
        print("\nGroups:")
        if db["groups"]:
            for i, g in enumerate(db["groups"]):
                print(f'  {i + 1}. Group {g["group_id"]} "{g["name"]}"')
        else:
            print("  (none)")

        print("\na. Create group")
        print("b. Rename group")
        print("c. Delete group")
        print("d. Back")

        choice = (await ainput("> ")).strip().lower()
        try:
            if choice == "a":
                gid = next_group_id(db)
                name = (await ainput(f"Group name (id={gid}): ")).strip()
                if not name:
                    print("Cancelled.")
                    continue
                upsert_group(db, {"group_id": gid, "name": name})
                save_db(db)
                print(f'Created group {gid} "{name}".')

            elif choice == "b":
                if not db["groups"]:
                    print("No groups to rename.")
                    continue
                pick = (await ainput("Group # to rename: ")).strip()
                idx = int(pick) - 1
                if idx < 0 or idx >= len(db["groups"]):
                    raise ValueError
                g = db["groups"][idx]
                name = (await ainput(f"New name for group {g['group_id']}: ")).strip()
                if not name:
                    print("Cancelled.")
                    continue
                g["name"] = name
                save_db(db)
                print(f'Renamed group {g["group_id"]} to "{name}".')

            elif choice == "c":
                if not db["groups"]:
                    print("No groups to delete.")
                    continue
                pick = (await ainput("Group # to delete: ")).strip()
                idx = int(pick) - 1
                if idx < 0 or idx >= len(db["groups"]):
                    raise ValueError
                g = db["groups"][idx]
                confirm = (
                    (
                        await ainput(
                            f'Delete group {g["group_id"]} "{g["name"]}"? '
                            f"Send DELETE to all known devices? (y/N): "
                        )
                    )
                    .strip()
                    .lower()
                )
                if confirm != "y":
                    print("Cancelled.")
                    continue
                if db["passphrase"] and db["devices"]:
                    try:
                        csr, _ = await conn.ensure_connected(db["passphrase"])
                        gid_bytes = g["group_id"].to_bytes(2, "big")
                        for dev in db["devices"]:
                            dest, payload = _create_packet(
                                dev["device_id"], Verb.DELETE, Noun.GROUPS, gid_bytes
                            )
                            await csr.send(dest, MODEL_OPCODE, payload)
                            await csr.receive(timeout=1.0)
                        print(f"  Sent DELETE GROUPS to {len(db['devices'])} device(s).")
                    except (BleakError, RuntimeError, TimeoutError) as e:
                        print(f"  Warning: could not send to devices: {e}")
                remove_group(db, g["group_id"])
                save_db(db)
                print(f"Group {g['group_id']} removed from database.")

            elif choice == "d":
                return db

        except (ValueError, IndexError):
            print("Invalid input.")
        except EOFError:
            return db


async def group_control_menu(db: dict, conn: BLEConnection, group: dict) -> dict:
    gid = group["group_id"]

    while True:
        print(f'\n=== Group {gid} "{group["name"]}" ===')
        print("1. Set brightness (0-255)")
        print("2. Set color temp (2700-6500K)")
        print("3. Turn ON")
        print("4. Turn OFF")
        print("5. Back")

        choice = (await ainput("> ")).strip()
        try:
            csr, mesh = await conn.ensure_connected(db["passphrase"])

            if choice == "1":
                val = int((await ainput("Brightness (0-255): ")).strip())
                await mesh.send_command(
                    {"avid": gid, "command": "update", "json": json.dumps({"brightness": val})}
                )
                print(f"Brightness -> {val}")

            elif choice == "2":
                val = int((await ainput("Color temp (2700-6500): ")).strip())
                await mesh.send_command(
                    {"avid": gid, "command": "update", "json": json.dumps({"color_temp": val})}
                )
                print(f"Color temp -> {val}K")

            elif choice == "3":
                await mesh.send_command(
                    {"avid": gid, "command": "update", "json": json.dumps({"state": "ON"})}
                )
                print("ON")

            elif choice == "4":
                await mesh.send_command(
                    {"avid": gid, "command": "update", "json": json.dumps({"state": "OFF"})}
                )
                print("OFF")

            elif choice == "5":
                return db

        except (BleakError, RuntimeError, TimeoutError) as e:
            print(f"Error: {e}")
            await conn.disconnect()
        except (ValueError, IndexError):
            print("Invalid input.")
        except EOFError:
            return db


def _get_latest_fw(db: dict, device: dict) -> dict[str, str]:
    """Get latest firmware versions for a device from cloud + peer comparison.

    Returns dict of mcu_id → version string, combining cloud data (preferred)
    with peer comparison (fallback).
    """
    vid = device.get("vendor_id")
    cpid = device.get("csr_product_id")
    latest = {}

    # Cloud data
    if vid is not None and cpid is not None:
        product = find_product(db, vid, cpid)
        if product:
            latest.update(product.get("latest_fw", {}))

    # Peer comparison: highest fw among devices with same (vendor_id, csr_product_id)
    if vid is not None and cpid is not None:
        for d in db["devices"]:
            if d.get("vendor_id") != vid or d.get("csr_product_id") != cpid:
                continue
            for mcu_id, ver in d.get("firmware", {}).items():
                if mcu_id not in latest or fw_version_newer(ver, latest[mcu_id]):
                    latest[mcu_id] = ver

    return latest


def _fw_status_label(device: dict, latest: dict[str, str]) -> str:
    """Return 'up to date', 'UPDATE (MCU x)', or '—' for unknown."""
    fw = device.get("firmware", {})
    if not fw:
        return "unknown"
    updates = []
    for mcu_id, ver in fw.items():
        latest_ver = latest.get(mcu_id)
        if latest_ver and fw_version_newer(latest_ver, ver):
            updates.append(f"MCU {mcu_id}")
    if updates:
        return f"UPDATE ({', '.join(updates)})"
    return "up to date"


async def action_repair_group_memberships(db: dict, conn: BLEConnection) -> dict:
    if not db["passphrase"]:
        print("Set passphrase first (option 1).")
        return db
    if not db["devices"]:
        print("No devices in database.")
        return db

    csr, _ = await conn.ensure_connected(db["passphrase"])

    print("\nScanning group memberships from all devices...\n")
    device_memberships: dict[int, list[int]] = {}
    for dev in db["devices"]:
        avid = dev["device_id"]
        print(f"  {dev['name']} (id={avid})...", end=" ", flush=True)
        try:
            group_ids = await read_device_groups(csr, avid)
        except (BleakError, RuntimeError, TimeoutError) as e:
            print(f"ERROR: {e}")
            continue
        device_memberships[avid] = group_ids
        labels = []
        for gid in group_ids:
            g = find_group(db, gid)
            labels.append(str(gid) + (f' "{g["name"]}"' if g else " (unknown)"))
        print(", ".join(labels) if labels else "(none)")

    # Build group → [device_ids] map from scan results
    group_members: dict[int, list[int]] = {}
    for avid, gids in device_memberships.items():
        for gid in gids:
            group_members.setdefault(gid, []).append(avid)

    known_ids = {g["group_id"] for g in db["groups"]}
    all_referenced = set(group_members.keys())
    missing_groups = all_referenced - known_ids
    orphaned_groups = known_ids - all_referenced

    # Show full membership picture
    print("\n=== Group membership (from mesh) ===\n")
    for gid in sorted(all_referenced):
        g = find_group(db, gid)
        name = g["name"] if g else "(not in DB)"
        members = group_members[gid]
        member_names = [
            next((d["name"] for d in db["devices"] if d["device_id"] == did), str(did))
            for did in sorted(members)
        ]
        marker = " [MISSING FROM DB]" if gid in missing_groups else ""
        print(f'  Group {gid} "{name}"{marker}: {", ".join(member_names)}')

    changed = False

    # Update device records to reflect actual memberships
    for dev in db["devices"]:
        avid = dev["device_id"]
        if avid in device_memberships:
            dev["groups"] = sorted(device_memberships[avid])

    # Add groups found on mesh but absent from DB
    if missing_groups:
        print(
            f"\n{len(missing_groups)} group(s) found on mesh but not in database — adding with placeholder names."
        )
        for gid in sorted(missing_groups):
            upsert_group(db, {"group_id": gid, "name": f"Group {gid}"})
        print("  Use 'Manage groups' to rename.")
        changed = True
    else:
        print("\nNo missing groups.")

    # Offer to remove groups that no device belongs to
    if orphaned_groups:
        print(f"\n{len(orphaned_groups)} group(s) in database with no device members:")
        for gid in sorted(orphaned_groups):
            g = find_group(db, gid)
            print(f'  Group {gid} "{g["name"] if g else "?"}"')
        confirm = (await ainput("Remove orphaned groups from database? (y/N): ")).strip().lower()
        if confirm == "y":
            for gid in orphaned_groups:
                remove_group(db, gid)
            print(f"Removed {len(orphaned_groups)} orphaned group(s).")
            changed = True
    else:
        print("No orphaned groups.")

    # Always save if we updated device membership records
    if device_memberships:
        changed = True

    if changed:
        save_db(db)
        print("\nDatabase saved.")

    return db


async def action_firmware_status(db: dict, conn: BLEConnection) -> dict:
    while True:
        print("\n=== Firmware Status ===\n")

        # Build table
        has_any_fw = any(d.get("firmware") for d in db["devices"])
        if db["devices"] and has_any_fw:
            # Header
            print(
                f"  {'Device':<20s} {'MCU 0':<10s} {'MCU 1':<10s} "
                f"{'Latest 0':<10s} {'Latest 1':<10s} Status"
            )
            for dev in db["devices"]:
                fw = dev.get("firmware", {})
                latest = _get_latest_fw(db, dev)
                mcu0 = fw.get("0", "\u2014")
                mcu1 = fw.get("1", "\u2014")
                lat0 = latest.get("0", "\u2014")
                lat1 = latest.get("1", "\u2014")
                status = _fw_status_label(dev, latest)
                name = dev["name"][:20]
                print(f"  {name:<20s} {mcu0:<10s} {mcu1:<10s} {lat0:<10s} {lat1:<10s} {status}")

            sources = []
            if db["products"]:
                sources.append("cloud")
            sources.append("peer comparison")
            print(f"\n  Latest source: {' + '.join(sources)}")
        elif db["devices"]:
            print("  No firmware data scanned yet.")
        else:
            print("  No devices in database.")

        print("\na. Scan devices for firmware (BLE)")
        print("b. Fetch latest from Avi-on cloud")
        print("c. Back")

        choice = (await ainput("> ")).strip().lower()
        try:
            if choice == "a":
                if not db["passphrase"]:
                    print("Set passphrase first (option 1).")
                    continue
                if not db["devices"]:
                    print("No devices in database.")
                    continue
                csr, _ = await conn.ensure_connected(db["passphrase"])
                for dev in db["devices"]:
                    print(f"  Scanning {dev['name']} (id={dev['device_id']})...")
                    info = await scan_device_firmware(csr, dev["device_id"])
                    if info:
                        dev["firmware"] = info["firmware"]
                        if "vendor_id" in info:
                            dev["vendor_id"] = info["vendor_id"]
                            dev["csr_product_id"] = info["csr_product_id"]
                        parts = [f"fw={info['firmware']}"]
                        if "vendor_id" in info:
                            parts.append(f"vendor=0x{info['vendor_id']:04x}")
                            parts.append(f"csr_product={info['csr_product_id']}")
                        else:
                            parts.append("(no PING response)")
                        print(f"    {'  '.join(parts)}")
                    else:
                        print("    (no response)")
                save_db(db)
                print("Firmware data saved.")

            elif choice == "b":
                if not HAS_AVIONHTTP:
                    print("Cloud features require avionhttp.  Install with: pip install avionhttp")
                    continue
                email = (await ainput("Avi-on email: ")).strip()
                password = (await ainput("Avi-on password: ")).strip()
                if not email or not password:
                    print("Cancelled.")
                    continue
                print("Fetching product catalog...")
                products = await fetch_cloud_products(email, password)
                for p in products:
                    upsert_product(db, p)
                save_db(db)
                print(f"Fetched {len(products)} product(s).")

            elif choice == "c":
                return db

        except (BleakError, RuntimeError, TimeoutError) as e:
            print(f"Error: {e}")
            await conn.disconnect()
        except (ValueError, IndexError):
            print("Invalid input.")
        except EOFError:
            return db


async def action_view_control(db: dict, conn: BLEConnection) -> dict:
    if not db["devices"] and not db["groups"]:
        print("No devices or groups in database.")
        return db

    n_dev = len(db["devices"])
    print("\nDevices:")
    if db["devices"]:
        for i, dev in enumerate(db["devices"]):
            product = PRODUCT_NAMES.get(dev.get("product_id", 0), "Unknown")
            mac = dev.get("mac_address", "")
            print(f"  {i + 1}. {dev['name']}  ({mac}, id={dev['device_id']}, {product})")
    else:
        print("  (none)")

    print("Groups:")
    if db["groups"]:
        for i, grp in enumerate(db["groups"]):
            print(f"  {n_dev + i + 1}. {grp['name']}  (group_id={grp['group_id']})")
    else:
        print("  (none)")

    choice = (await ainput("Pick # (or Enter to cancel): ")).strip()
    if not choice:
        return db
    try:
        idx = int(choice) - 1
        total = n_dev + len(db["groups"])
        if idx < 0 or idx >= total:
            raise ValueError
    except ValueError:
        print("Invalid choice.")
        return db

    if idx < n_dev:
        return await device_control_menu(db, conn, db["devices"][idx])
    else:
        return await group_control_menu(db, conn, db["groups"][idx - n_dev])


# --- Main ---


async def main_menu() -> None:
    db = load_db()
    conn = BLEConnection()

    try:
        while True:
            pp = db["passphrase"] or "not set"
            print(f"""
========================================
  CSRMesh Manager
========================================
  Passphrase: {pp}
  Devices: {len(db["devices"])}  Groups: {len(db["groups"])}
  BLE connected: {"yes" if conn.is_connected else "no"}

1. Setup passphrase
2. Import devices from Avi-on cloud
3. Scan for unclaimed devices
4. View/control devices
5. Manage groups
6. Firmware status
7. Discover mesh devices
8. Repair group memberships
9. Quit""")

            choice = (await ainput("> ")).strip()
            try:
                if choice == "1":
                    db = await action_setup_passphrase(db)
                elif choice == "2":
                    db = await action_import_cloud(db)
                elif choice == "3":
                    db = await action_scan_claim(db, conn)
                elif choice == "4":
                    db = await action_view_control(db, conn)
                elif choice == "5":
                    db = await action_manage_groups(db, conn)
                elif choice == "6":
                    db = await action_firmware_status(db, conn)
                elif choice == "7":
                    db = await action_discover_mesh(db, conn)
                elif choice == "8":
                    db = await action_repair_group_memberships(db, conn)
                elif choice == "9":
                    break
            except (BleakError, RuntimeError, TimeoutError) as e:
                print(f"Error: {e}")
                await conn.disconnect()
            except EOFError:
                break

    finally:
        await conn.disconnect()

    print("Bye.")


def main():
    try:
        asyncio.run(main_menu())
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
