"""Bounded framing, authenticated endpoints and real cross-process link coverage."""

from __future__ import annotations

import asyncio
import multiprocessing
import socket
import ssl
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from aioquic.asyncio import connect, serve
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import StreamDataReceived
from cryptography.hazmat.primitives import serialization
from websockets.asyncio.client import connect as ws_connect
from websockets.asyncio.server import serve as ws_serve
from websockets.exceptions import ConnectionClosed

from hivemind.swarm import quic_link as ql


@pytest.mark.parametrize(
    "raw", [b"", b"\x80", b"\x80" * 9, b"\xff" * 4 + b"\x7f", b"\x82\x00x\n",
            b"\x03x\n", b"\x02x\ntrailing", b"\x01\n", b"\x02xx", b"\x03\xff\xff\n",
            b"\x04x\n\n\n", "text"],
)
def test_multistream_parser_rejects_malformed_input(raw: object) -> None:
    with pytest.raises(ql.LinkProtocolError):
        ql._parse_multistream(raw)


@pytest.mark.parametrize("value", [-1, 1 << 32])
def test_varint_encoder_is_bounded(value: int) -> None:
    with pytest.raises(ValueError):
        ql._uvarint(value)


@pytest.mark.parametrize("size", [0, ql.MAX_FRAME_BYTES + 1, 0xFFFFFFFF])
def test_oversized_or_empty_quic_frame_length_is_rejected_immediately(size: int) -> None:
    with pytest.raises(ql.LinkProtocolError):
        ql._drain_frames(bytearray(struct.pack("!I", size)))


@pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.1", "example.org", "localhost"])
def test_plaintext_websocket_listener_requires_literal_loopback(host: str) -> None:
    with pytest.raises(ValueError, match="require TLS"):
        ql.WebSocketHiddenStateServer(host)


@pytest.mark.parametrize("url", ["ws://192.0.2.1:80", "ws://example.org", "ws://localhost:1234",
                                 "http://127.0.0.1", "ws://user:password@127.0.0.1", "ws://127.0.0.1/#x"])
def test_unsafe_websocket_urls_fail_before_any_connection(url: str) -> None:
    with pytest.raises(ValueError):
        asyncio.run(ql.websocket_send_frames(url, [b"payload"], timeout=0.05))


def test_unverified_websocket_tls_context_is_rejected_before_connection() -> None:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with pytest.raises(ValueError, match="verify both"):
        asyncio.run(ql.websocket_send_frames("wss://192.0.2.1", [b"payload"], ssl_context=context))


@pytest.mark.parametrize("frames", [[b""], ["text"], [b"x"] * (ql.MAX_BATCH_FRAMES + 1)])
def test_sender_rejects_invalid_batches_before_connecting(frames: list) -> None:
    for send in (lambda: ql.quic_send_frames("127.0.0.1", 1, frames, "unused"),
                 lambda: ql.websocket_send_frames("ws://127.0.0.1:1", frames)):
        with pytest.raises((TypeError, ValueError)):
            asyncio.run(send())


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_deadlines_are_rejected(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        asyncio.run(ql.quic_send_frames("127.0.0.1", 1, [b"x"], "unused", timeout=timeout))


def _client_configuration(identity: ql.LinkIdentity) -> QuicConfiguration:
    configuration = QuicConfiguration(is_client=True, alpn_protocols=[ql.ALPN], server_name=ql.SERVER_NAME)
    configuration.load_verify_locations(cadata=identity.ca_pem.encode("ascii"))
    return configuration


async def _assert_stream_closed_without_ack(reader: asyncio.StreamReader, timeout: float = 1.0) -> None:
    """Verify the peer closed the stream/connection without sending any acknowledgement bytes.

    ``aioquic < 1.9`` feeds EOF (``b""``) on ``connection_lost``, whereas ``aioquic >= 1.9``
    raises ``ConnectionError("Connection closed")`` on open streams lacking a stream FIN.
    """
    try:
        data = await asyncio.wait_for(reader.read(), timeout)
    except ConnectionError:
        return
    assert data == b""


@pytest.mark.parametrize("payload", [b"\x00", struct.pack("!I", 100) + b"short",
                                     struct.pack("!I", 0), struct.pack("!I", ql.MAX_FRAME_BYTES + 1)])
def test_real_quic_rejects_malformed_stream_without_acknowledging_it(payload: bytes) -> None:
    async def run() -> None:
        identity = ql.LinkIdentity.generate()
        received: list[bytes] = []
        server = await ql.QuicHiddenStateServer(identity, on_frame=received.append).start()
        try:
            async with connect("127.0.0.1", server.port, configuration=_client_configuration(identity)) as peer:
                reader, writer = await peer.create_stream()
                writer.write(payload)
                writer.write_eof()
                await _assert_stream_closed_without_ack(reader, 1.0)
            assert received == []
            good = await ql.quic_send_frames("127.0.0.1", server.port, [b"still-alive"], identity.ca_pem)
            assert good.delivered == 1 and received == [b"still-alive"]
        finally:
            server.close()
    asyncio.run(run())


def test_partial_quic_stream_has_a_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ql, "STREAM_TIMEOUT_S", 0.05)

    async def run() -> None:
        identity = ql.LinkIdentity.generate()
        server = await ql.QuicHiddenStateServer(identity).start()
        try:
            async with connect("127.0.0.1", server.port, configuration=_client_configuration(identity)) as peer:
                reader, writer = await peer.create_stream()
                writer.write(struct.pack("!I", 100) + b"partial")
                await _assert_stream_closed_without_ack(reader, 1.0)
            assert server.frames_received == 0
        finally:
            server.close()
    asyncio.run(run())


@pytest.mark.parametrize("ack", [b"", b"x", b"12345", struct.pack("!I", 99)])
def test_quic_acknowledgement_must_have_exact_length_and_count(ack: bytes) -> None:
    class BadAck(QuicConnectionProtocol):
        def quic_event_received(self, event) -> None:
            if isinstance(event, StreamDataReceived) and event.end_stream:
                self._quic.send_stream_data(event.stream_id, ack, end_stream=True)
                self.transmit()

    async def run() -> None:
        identity = ql.LinkIdentity.generate()
        config = QuicConfiguration(is_client=False, alpn_protocols=[ql.ALPN])
        config.certificate, config.private_key = identity.certificate, identity.private_key
        server = await serve("127.0.0.1", 0, configuration=config, create_protocol=BadAck)
        try:
            port = server._transport.get_extra_info("sockname")[1]
            with pytest.raises(ql.DeliveryUncertainError) as caught:
                await ql.quic_send_frames("127.0.0.1", port, [b"frame"], identity.ca_pem, timeout=0.6)
            assert isinstance(caught.value.__cause__, ql.LinkProtocolError)
        finally:
            server.close()
    asyncio.run(run())


def test_plaintext_websocket_server_rejects_text_frames_and_recovers() -> None:
    async def run() -> None:
        received: list[bytes] = []
        server = await ql.WebSocketHiddenStateServer(on_frame=received.append).start()
        try:
            async with ws_connect(server.url) as peer:
                await peer.send(ql._multistream_message(ql.MULTISTREAM))
                await peer.recv()
                await peer.send(ql._multistream_message(ql.LIBP2P_PROTOCOL))
                await peer.recv()
                await peer.send("text frame")
                with pytest.raises(ConnectionClosed):
                    await peer.recv()
            assert received == []
            assert await ql.websocket_send_frames(server.url, [b"binary frame"]) == 1
            assert received == [b"binary frame"]
        finally:
            await server.close()
    asyncio.run(run())


def test_websocket_has_one_overall_deadline() -> None:
    async def run() -> None:
        async def slow(websocket) -> None:
            try:
                await websocket.recv()
                await asyncio.sleep(0.06)
                await websocket.send(ql._multistream_message(ql.MULTISTREAM))
                await websocket.recv()
                await asyncio.sleep(0.06)
                await websocket.send(ql._multistream_message(ql.LIBP2P_PROTOCOL))
            except ConnectionClosed:
                pass
        async with ws_serve(slow, "127.0.0.1", 0) as server:
            url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                await ql.websocket_send_frames(url, [b"frame"], timeout=0.09)
            assert time.monotonic() - started < 0.3
    asyncio.run(run())


def test_cancelled_quic_send_closes_its_udp_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        create = loop.create_datagram_endpoint
        transports = []

        async def record(*args, **kwargs):
            transport, protocol = await create(*args, **kwargs)
            transports.append(transport)
            return transport, protocol
        monkeypatch.setattr(loop, "create_datagram_endpoint", record)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as dead:
            dead.bind(("127.0.0.1", 0))
            identity = ql.LinkIdentity.generate()
            task = asyncio.create_task(ql.quic_send_frames(
                "127.0.0.1", dead.getsockname()[1], [b"frame"], identity.ca_pem, timeout=5.0,
            ))
            for _ in range(100):
                if transports:
                    break
                await asyncio.sleep(0.005)
            assert transports
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert all(transport.is_closing() for transport in transports)
    asyncio.run(run())


def test_ambiguous_quic_delivery_is_never_replayed_over_websocket() -> None:
    async def run() -> None:
        delivered: list[bytes] = []
        fallback: list[bytes] = []

        def process_then_fail(frame: bytes) -> None:
            delivered.append(frame)
            raise RuntimeError("application failed after accepting the payload")
        identity = ql.LinkIdentity.generate()
        quic = await ql.QuicHiddenStateServer(identity, on_frame=process_then_fail).start()
        ws = await ql.WebSocketHiddenStateServer(on_frame=fallback.append).start()
        try:
            endpoint = ql.LinkEndpoint("127.0.0.1", quic.port, identity.ca_pem, websocket_url=ws.url)
            with pytest.raises(ql.DeliveryUncertainError):
                await ql.RemoteHiddenStateLink(endpoint, timeout=0.6).send_frames([b"do-not-replay"])
            assert delivered == [b"do-not-replay"]
            assert fallback == []
        finally:
            quic.close()
            await ws.close()
    asyncio.run(run())


def test_unreachable_initial_quic_path_can_fall_back_to_explicit_loopback_websocket() -> None:
    async def run() -> None:
        received: list[bytes] = []
        server = await ql.WebSocketHiddenStateServer(on_frame=received.append).start()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as dead:
            dead.bind(("127.0.0.1", 0))
            try:
                endpoint = ql.LinkEndpoint("127.0.0.1", dead.getsockname()[1], ql.LinkIdentity.generate().ca_pem,
                                           websocket_url=server.url)
                report = await ql.RemoteHiddenStateLink(endpoint, timeout=0.1).send_frames([b"actual-frame"])
                assert report.protocol == "libp2p_websocket"
                assert received == [b"actual-frame"]
            finally:
                await server.close()
    asyncio.run(run())


def _server_tls_context(identity: ql.LinkIdentity, directory: Path) -> ssl.SSLContext:
    cert, key = directory / "test-cert.pem", directory / "test-key.pem"
    cert.write_bytes(identity.certificate.public_bytes(serialization.Encoding.PEM))
    key.touch(mode=0o600)
    key.write_bytes(identity.private_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    ))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    return context


def test_wss_fallback_verifies_the_endpoint_ca_and_delivers_real_bytes(tmp_path: Path) -> None:
    async def run() -> None:
        identity = ql.LinkIdentity.generate()
        received: list[bytes] = []
        server = await ql.WebSocketHiddenStateServer(
            on_frame=received.append, ssl_context=_server_tls_context(identity, tmp_path),
        ).start()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as dead:
            dead.bind(("127.0.0.1", 0))
            try:
                endpoint = ql.LinkEndpoint("127.0.0.1", dead.getsockname()[1], identity.ca_pem,
                                           websocket_url=server.url)
                report = await ql.RemoteHiddenStateLink(endpoint, timeout=0.3).send_frames([b"encrypted-frame"])
                assert report.frames_delivered == 1
                assert received == [b"encrypted-frame"]
                wrong = ql.LinkEndpoint("127.0.0.1", dead.getsockname()[1], ql.LinkIdentity.generate().ca_pem,
                                        websocket_url=server.url)
                with pytest.raises(ssl.SSLCertVerificationError):
                    await ql.RemoteHiddenStateLink(wrong, timeout=0.3).send_frames([b"must-not-arrive"])
                assert received == [b"encrypted-frame"]
            finally:
                await server.close()
    asyncio.run(run())


def _receiver_process(pipe) -> None:
    """A long-running receiver in a separate interpreter, never a per-send fixture."""
    async def run() -> None:
        identity = ql.LinkIdentity.generate()
        frames: list[bytes] = []
        server = await ql.QuicHiddenStateServer(identity, on_frame=frames.append).start()
        pipe.send((server.port, identity.ca_pem))
        try:
            while True:
                command = await asyncio.to_thread(pipe.recv)
                if command == "snapshot":
                    pipe.send(list(frames))
                elif command == "stop":
                    break
        finally:
            server.close()
            pipe.close()
    asyncio.run(run())


def test_persistent_endpoint_crosses_processes_and_event_loops_without_warmup_frames() -> None:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_receiver_process, args=(child,))
    process.start()
    child.close()
    try:
        assert parent.poll(20), "receiver process did not become ready"
        port, ca_pem = parent.recv()
        link = ql.RemoteHiddenStateLink(ql.LinkEndpoint("127.0.0.1", port, ca_pem))
        frames = [b"hidden-state:first:" + bytes(range(256)) * 4,
                  b"hidden-state:second:" + bytes(reversed(range(256))) * 4,
                  b"hidden-state:third:" + b"\x00\xff" * 100]
        first = asyncio.run(link.send_frames(frames[:1]))
        second = asyncio.run(link.send_frames(frames[1:]))
        assert first.frames_delivered == 1 and not first.session_resumed
        assert second.frames_delivered == 2 and second.session_resumed and second.early_data_accepted
        parent.send("snapshot")
        assert parent.poll(5)
        assert parent.recv() == frames
        assert process.is_alive()
    finally:
        if process.is_alive():
            parent.send("stop")
            process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
    assert process.exitcode == 0


def test_frame_parser_enforces_count_limit_before_building_an_unbounded_list() -> None:
    raw = (struct.pack("!I", 1) + b"x") * (ql.MAX_BATCH_FRAMES + 1)
    with pytest.raises(ql.LinkProtocolError, match="too many"):
        ql._drain_frames(bytearray(raw))


def test_outgoing_frame_and_batch_size_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ql, "MAX_FRAME_BYTES", 16)
    monkeypatch.setattr(ql, "MAX_BATCH_BYTES", 20)
    with pytest.raises(ValueError, match="frame"):
        ql._validate_frames([b"x" * 17])
    with pytest.raises(ValueError, match="batch"):
        ql._validate_frames([b"x" * 8, b"y" * 8])


def test_too_many_open_quic_streams_close_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ql, "MAX_OPEN_STREAMS", 2)

    async def run() -> None:
        identity = ql.LinkIdentity.generate()
        server = await ql.QuicHiddenStateServer(identity).start()
        try:
            async with connect("127.0.0.1", server.port, configuration=_client_configuration(identity)) as peer:
                readers, writers = [], []
                for _ in range(3):
                    reader, writer = await peer.create_stream()
                    readers.append(reader)
                    writers.append(writer)
                    writer.write(b"\x00")
                await _assert_stream_closed_without_ack(readers[-1], 1.0)
            assert server.frames_received == 0
        finally:
            server.close()
    asyncio.run(run())


def test_oversized_websocket_message_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ql, "MAX_FRAME_BYTES", 64)

    async def run() -> None:
        received: list[bytes] = []
        server = await ql.WebSocketHiddenStateServer(on_frame=received.append).start()
        try:
            async with ws_connect(server.url) as peer:
                await peer.send(ql._multistream_message(ql.MULTISTREAM))
                await peer.recv()
                await peer.send(ql._multistream_message(ql.LIBP2P_PROTOCOL))
                await peer.recv()
                await peer.send(b"x" * 65)
                with pytest.raises(ConnectionClosed):
                    await peer.recv()
            assert received == []
            assert await ql.websocket_send_frames(server.url, [b"still-running"]) == 1
        finally:
            await server.close()
    asyncio.run(run())


def test_wrong_quic_ca_fails_before_sending_application_frames() -> None:
    async def run() -> None:
        identity = ql.LinkIdentity.generate()
        server = await ql.QuicHiddenStateServer(identity).start()
        try:
            endpoint = ql.LinkEndpoint("127.0.0.1", server.port, ql.LinkIdentity.generate().ca_pem)
            with pytest.raises(ConnectionError) as caught:
                await ql.RemoteHiddenStateLink(endpoint, timeout=0.6).send_frames([b"must-not-arrive"])
            assert not isinstance(caught.value, ql.DeliveryUncertainError)
            assert server.frames_received == 0
        finally:
            server.close()
    asyncio.run(run())


def test_session_ticket_cache_is_bounded_and_single_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ql, "MAX_SESSION_TICKETS", 3)
    server = ql.QuicHiddenStateServer(ql.LinkIdentity.generate())
    for index in range(5):
        server._store_ticket(SimpleNamespace(ticket=bytes([index]), is_valid=True))
    assert len(server.tickets) == 3
    assert server._consume_ticket(b"\x04") is not None
    assert server._consume_ticket(b"\x04") is None
    server._store_ticket(SimpleNamespace(ticket=b"expired", is_valid=False))
    assert server._consume_ticket(b"expired") is None
