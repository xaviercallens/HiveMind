# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Endpoint validation and observable protocol failures on loopback transports."""

import asyncio
import inspect
from types import SimpleNamespace

import pytest
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import ConnectionTerminated
from websockets.asyncio.client import connect as ws_connect
from websockets.asyncio.server import serve as ws_serve
from websockets.exceptions import ConnectionClosedError

from hivemind.swarm import quic_link as link

# IP-63: generous loopback timeout for loaded CI / multi-worker pytest runners
LOOPBACK_TIMEOUT_S = 10.0


@pytest.mark.parametrize("sequence", [-1, True, 1.5])
def test_envelope_sequence_rejects_non_integer_or_negative_values(sequence):
    with pytest.raises(ValueError, match="non-negative integer"):
        link.FrameEnvelope(sequence, 1.0, b"data").encode()


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), float("-inf")])
def test_envelope_encode_rejects_nonfinite_timestamps(timestamp):
    with pytest.raises(ValueError, match="timestamp must be finite"):
        link.FrameEnvelope(0, timestamp, b"data").encode()


def test_envelope_decoder_requires_binary_input():
    with pytest.raises(TypeError, match="must be bytes"):
        link.FrameEnvelope.decode("not a wire frame")


@pytest.mark.parametrize("field", ["host", "server_name", "ca_pem"])
def test_endpoint_requires_address_and_explicit_trust(field):
    kwargs = dict(host="127.0.0.1", port=4433, server_name="server", ca_pem="trusted CA")
    kwargs[field] = ""
    with pytest.raises(ValueError, match="required"):
        link.LinkEndpoint(**kwargs)


@pytest.mark.parametrize("port", [True, 1.5, 0, 65536])
def test_endpoint_rejects_invalid_port_before_connecting(port):
    with pytest.raises(ValueError, match="port must be between"):
        link.LinkEndpoint("127.0.0.1", port, "trusted CA")


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "224.0.0.1", "ff02::1"])
def test_endpoint_rejects_unspecified_and_multicast_addresses(host):
    with pytest.raises(ValueError, match="specific unicast"):
        link.LinkEndpoint(host, 4433, "trusted CA")


def test_dns_endpoint_is_preserved_for_later_async_resolution():
    endpoint = link.LinkEndpoint("worker.example", 4433, "trusted CA")
    assert endpoint.host == "worker.example"
    assert endpoint.port == 4433
    assert endpoint.ca_pem == "trusted CA"


@pytest.mark.parametrize("protocol", ["", "bad\nprotocol", "x" * link.MAX_PROTOCOL_BYTES])
def test_multistream_encoder_rejects_invalid_protocol_names(protocol):
    with pytest.raises(ValueError, match="protocol name"):
        link._multistream_message(protocol)


def test_websocket_port_zero_is_rejected_before_connecting():
    with pytest.raises(ValueError, match="invalid WebSocket port"):
        asyncio.run(link.websocket_send_frames("ws://127.0.0.1:0", [b"data"]))


def test_expired_quic_session_tickets_are_evicted_when_new_ticket_arrives():
    server = link.QuicHiddenStateServer(link.LinkIdentity.generate())
    expired = SimpleNamespace(ticket=b"expired", is_valid=False)
    fresh = SimpleNamespace(ticket=b"fresh", is_valid=True)
    server.tickets[expired.ticket] = expired
    server._store_ticket(fresh)
    assert set(server.tickets) == {b"fresh"}
    assert server._consume_ticket(b"expired") is None
    assert server._consume_ticket(b"fresh") is fresh


@pytest.mark.parametrize("scenario,expected_code", [("version", 1002), ("protocol", 1002), ("limit", 1002), ("handler", 1011)])
def test_websocket_protocol_errors_close_with_correct_code(monkeypatch, scenario, expected_code):
    received = []

    def on_frame(frame):
        if scenario == "handler":
            raise RuntimeError("frame consumer failed")
        received.append(frame)

    if scenario == "limit":
        monkeypatch.setattr(link, "MAX_BATCH_FRAMES", 1)

    async def run():
        server = await link.WebSocketHiddenStateServer(on_frame=on_frame).start()
        try:
            options = {"proxy": None} if "proxy" in inspect.signature(ws_connect).parameters else {}
            async with ws_connect(server.url, **options) as websocket:
                version = "/unsupported/1" if scenario == "version" else link.MULTISTREAM
                await websocket.send(link._multistream_message(version))
                if scenario != "version":
                    assert await websocket.recv() == link._multistream_message(link.MULTISTREAM)
                    protocol = "/unsupported/1" if scenario == "protocol" else link.LIBP2P_PROTOCOL
                    await websocket.send(link._multistream_message(protocol))
                    assert await websocket.recv() == link._multistream_message("na" if scenario == "protocol" else link.LIBP2P_PROTOCOL)
                    if scenario not in {"protocol"}:
                        await websocket.send(b"first")
                        if scenario == "limit":
                            await websocket.send(b"second")
                with pytest.raises(ConnectionClosedError) as caught:
                    await asyncio.wait_for(websocket.recv(), LOOPBACK_TIMEOUT_S)
                assert caught.value.rcvd.code == expected_code
            assert server.frames_received == (1 if scenario == "limit" else 0)
            assert received == ([b"first"] if scenario == "limit" else [])
        finally:
            await server.close()

    asyncio.run(run())


def test_started_websocket_and_quic_listeners_cannot_be_started_twice():
    async def run():
        websocket = await link.WebSocketHiddenStateServer().start()
        quic = await link.QuicHiddenStateServer(link.LinkIdentity.generate()).start()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                await websocket.start()
            with pytest.raises(RuntimeError, match="already started"):
                await quic.start()
        finally:
            await websocket.close()
            quic.close()

    asyncio.run(run())


@pytest.mark.parametrize("violation", ["unidirectional", "batch_bytes"])
def test_quic_rejects_invalid_stream_without_poisoning_listener(monkeypatch, violation):
    if violation == "batch_bytes":
        monkeypatch.setattr(link, "MAX_BATCH_BYTES", 128)

    async def run():
        identity = link.LinkIdentity.generate()
        received = []
        server = await link.QuicHiddenStateServer(identity, on_frame=received.append).start()
        configuration = QuicConfiguration(is_client=True, alpn_protocols=[link.ALPN], server_name=link.SERVER_NAME)
        configuration.load_verify_locations(cadata=identity.ca_pem.encode("ascii"))
        try:
            async with connect("127.0.0.1", server.port, configuration=configuration) as peer:
                _, writer = await peer.create_stream(is_unidirectional=violation == "unidirectional")
                payload = b"x" * (129 if violation == "batch_bytes" else 1)
                writer.write(link._LENGTH.pack(len(payload)) + payload)
                writer.write_eof()
                await asyncio.wait_for(peer.wait_closed(), LOOPBACK_TIMEOUT_S)
            assert server.frames_received == 0
            assert received == []
            report = await link.quic_send_frames("127.0.0.1", server.port, [b"recovered"], identity.ca_pem)
            assert report.delivered == 1
            assert received == [b"recovered"]
        finally:
            server.close()

    asyncio.run(run())


def test_plaintext_sender_rejects_tls_options_before_connecting():
    with pytest.raises(ValueError, match="TLS options require"):
        asyncio.run(link.websocket_send_frames("ws://127.0.0.1:1", [b"data"], server_name="worker"))


def test_loopback_demo_refuses_a_remote_host_before_starting_listeners():
    with pytest.raises(ValueError, match="loopback demonstration"):
        asyncio.run(link.run_local_link([b"data"], host="192.0.2.1"))


@pytest.mark.parametrize("fault", ["protocol", "text_ack", "short_ack", "wrong_count"])
def test_websocket_sender_rejects_bad_negotiation_and_acknowledgements(fault):
    received = []

    async def peer(websocket):
        assert await websocket.recv() == link._multistream_message(link.MULTISTREAM)
        assert await websocket.recv() == link._multistream_message(link.LIBP2P_PROTOCOL)
        await websocket.send(link._multistream_message(link.MULTISTREAM))
        await websocket.send(link._multistream_message("na" if fault == "protocol" else link.LIBP2P_PROTOCOL))
        if fault == "protocol":
            return
        while (frame := await websocket.recv()) != b"":
            received.append(frame)
        reply = {"text_ack": "1", "short_ack": b"x", "wrong_count": link._LENGTH.pack(2)}[fault]
        await websocket.send(reply)

    async def run():
        async with ws_serve(peer, "127.0.0.1", 0) as server:
            url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            with pytest.raises(link.LinkProtocolError, match="negotiation failed|acknowledgement"):
                await link.websocket_send_frames(url, [b"data"])
        assert received == ([] if fault == "protocol" else [b"data"])

    asyncio.run(run())


def test_expired_ticket_is_discarded_before_a_real_quic_handshake():
    async def run():
        identity = link.LinkIdentity.generate()
        received = []
        server = await link.QuicHiddenStateServer(identity, on_frame=received.append).start()
        try:
            report = await link.quic_send_frames(
                "127.0.0.1", server.port, [b"data"], identity.ca_pem,
                session_ticket=SimpleNamespace(is_valid=False),
            )
            assert report.delivered == 1
            assert received == [b"data"]
            assert server.handshakes[-1]["session_resumed"] is False
        finally:
            server.close()

    asyncio.run(run())


def test_resumption_probe_requires_an_explicit_frame_source():
    server = link.QuicHiddenStateServer(link.LinkIdentity.generate())
    with pytest.raises(ValueError, match="needs frames"):
        asyncio.run(link.quic_0rtt_resumption_probe(server))


def test_resumption_probe_delivers_raw_frames_that_have_no_latency_envelope():
    async def run():
        received = []
        server = await link.QuicHiddenStateServer(link.LinkIdentity.generate(), on_frame=received.append).start()
        try:
            report = await link.quic_0rtt_resumption_probe(server, frames=[b"raw"])
            assert report.ok
            assert report.frames_sent == report.frames_delivered == 1
            assert received[-1] == b"raw"
        finally:
            server.close()

    asyncio.run(run())


@pytest.mark.parametrize("close_from", ["server", "peer"])
def test_open_stream_deadlines_are_cancelled_on_connection_close(monkeypatch, close_from):
    async def run():
        delivered = asyncio.Event()
        terminated = asyncio.Event()
        protocols = []

        class ObservedProtocol(link._ServerProtocol):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                protocols.append(self)

            def quic_event_received(self, event):
                super().quic_event_received(event)
                if isinstance(event, ConnectionTerminated):
                    terminated.set()

        monkeypatch.setattr(link, "_ServerProtocol", ObservedProtocol)
        identity = link.LinkIdentity.generate()
        server = await link.QuicHiddenStateServer(identity, on_frame=lambda frame: delivered.set()).start()
        configuration = QuicConfiguration(is_client=True, alpn_protocols=[link.ALPN], server_name=link.SERVER_NAME)
        configuration.load_verify_locations(cadata=identity.ca_pem.encode("ascii"))
        try:
            async with connect("127.0.0.1", server.port, configuration=configuration) as peer:
                _, writer = await peer.create_stream()
                writer.write(link._LENGTH.pack(4) + b"data")
                await asyncio.wait_for(delivered.wait(), LOOPBACK_TIMEOUT_S)
                protocol = protocols[0]
                timers = list(protocol._deadlines.values())
                assert len(timers) == 1 and not timers[0].cancelled()
                (protocol if close_from == "server" else peer).close()
                await asyncio.wait_for(peer.wait_closed(), LOOPBACK_TIMEOUT_S)
                if close_from == "peer":
                    await asyncio.wait_for(terminated.wait(), LOOPBACK_TIMEOUT_S)
                assert timers[0].cancelled()
                assert not protocol._deadlines
                assert not protocol._buffers
                assert not protocol._counts
                assert not protocol._totals
            assert server.frames_received == 1
        finally:
            server.close()

    asyncio.run(run())


def test_quic_frame_count_is_enforced_across_separate_data_events(monkeypatch):
    monkeypatch.setattr(link, "MAX_BATCH_FRAMES", 1)

    async def run():
        delivered = asyncio.Event()
        received = []

        def on_frame(frame):
            received.append(frame)
            delivered.set()

        identity = link.LinkIdentity.generate()
        server = await link.QuicHiddenStateServer(identity, on_frame=on_frame).start()
        configuration = QuicConfiguration(is_client=True, alpn_protocols=[link.ALPN], server_name=link.SERVER_NAME)
        configuration.load_verify_locations(cadata=identity.ca_pem.encode("ascii"))
        try:
            async with connect("127.0.0.1", server.port, configuration=configuration) as peer:
                reader, writer = await peer.create_stream()
                writer.write(link._LENGTH.pack(5) + b"first")
                await asyncio.wait_for(delivered.wait(), LOOPBACK_TIMEOUT_S)
                writer.write(link._LENGTH.pack(6) + b"second")
                writer.write_eof()
                assert await asyncio.wait_for(reader.read(), LOOPBACK_TIMEOUT_S) == b""
            assert received == [b"first"]
            assert server.frames_received == 1
        finally:
            server.close()

    asyncio.run(run())
