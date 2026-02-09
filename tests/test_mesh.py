from unittest.mock import Mock

import pytest
from recsrmesh import CSRMesh

from avionmesh.Mesh import Mesh, _get_date_packet, _get_time_packet


class TestCreateMeshPackets:
    @pytest.fixture
    def mesh(self):
        """Create a Mesh instance with mocked CSRMesh."""
        mock_csr = Mock(spec=CSRMesh)
        return Mesh(mock_csr)

    def test_write_date_command(self):
        dest_id, payload = _get_date_packet(2025, 11, 9)
        assert dest_id == 0
        assert payload == bytes([0x0, 0x15, 0x0, 0x0, 0x0, 0x19, 0x0B, 0x9, 0x0, 0x0])

    def test_write_time_command(self):
        dest_id, payload = _get_time_packet(21, 12, 45)
        assert dest_id == 0
        assert payload == bytes([0x0, 0x16, 0x0, 0x0, 0x0, 0x15, 0xC, 0x2D, 0x0, 0x0])
