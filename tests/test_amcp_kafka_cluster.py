# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The three-node Kafka quorum generator: the rules that decide whether the cluster survives a failure are checked here, without starting Kafka."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "amcp_kafka_cluster.py"
_spec = importlib.util.spec_from_file_location("amcp_kafka_cluster", _PATH)
assert _spec is not None and _spec.loader is not None
cluster = importlib.util.module_from_spec(_spec)
sys.modules["amcp_kafka_cluster"] = cluster  # dataclasses look their module up here
_spec.loader.exec_module(cluster)

NODES = [cluster.Node(1, "10.77.0.1"), cluster.Node(2, "10.77.0.2"), cluster.Node(3, "10.77.0.3")]


def props(node_id: int, **kw) -> dict[str, str]:  # noqa: ANN003
    text = cluster.node_properties(NODES[node_id - 1], NODES, "/data", **kw)
    return dict(line.split("=", 1) for line in text.splitlines() if line and not line.startswith("#"))


@pytest.mark.parametrize("voters,quorum,tolerated", [(3, 2, 1), (5, 3, 2), (7, 4, 3)])
def test_quorum_arithmetic(voters: int, quorum: int, tolerated: int) -> None:
    assert (cluster.quorum_size(voters), cluster.tolerated_failures(voters)) == (quorum, tolerated)


def test_four_voters_tolerate_no_more_than_three_so_even_counts_are_refused() -> None:
    assert cluster.tolerated_failures(4) == cluster.tolerated_failures(3) == 1
    for count in (1, 2, 4):
        nodes = [cluster.Node(i, f"10.0.0.{i}") for i in range(1, count + 1)]
        with pytest.raises(cluster.ClusterError, match="odd number"):
            cluster.validate(nodes, 1, 1)


def test_unsafe_replication_settings_are_refused() -> None:
    with pytest.raises(cluster.ClusterError, match="below a majority"):
        cluster.validate(NODES, replication=3, min_isr=1)  # one failure could lose an acknowledged write
    with pytest.raises(cluster.ClusterError, match="min_isr <= replication"):
        cluster.validate(NODES, replication=4, min_isr=2)
    with pytest.raises(cluster.ClusterError, match="min_isr <= replication"):
        cluster.validate(NODES, replication=3, min_isr=0)
    cluster.validate(NODES, replication=3, min_isr=2)


def test_duplicate_ids_and_shared_ports_are_refused() -> None:
    with pytest.raises(cluster.ClusterError, match="unique"):
        cluster.validate([cluster.Node(1, "a"), cluster.Node(1, "b"), cluster.Node(3, "c")], 3, 2)
    with pytest.raises(cluster.ClusterError, match="share"):
        cluster.validate([cluster.Node(1, "a"), cluster.Node(2, "a"), cluster.Node(3, "c")], 3, 2)  # nodes 1 and 2 on the same host and ports


def test_every_node_carries_the_same_voters_and_its_own_identity() -> None:
    voters = {props(i)["controller.quorum.voters"] for i in (1, 2, 3)}
    assert voters == {"1@10.77.0.1:9093,2@10.77.0.2:9093,3@10.77.0.3:9093"}
    assert [props(i)["node.id"] for i in (1, 2, 3)] == ["1", "2", "3"]
    assert all(props(i)["process.roles"] == "broker,controller" for i in (1, 2, 3))


def test_listeners_bind_only_to_the_nodes_private_address_never_to_all_interfaces() -> None:
    for i in (1, 2, 3):
        p = props(i)
        host = f"10.77.0.{i}"
        assert p["listeners"] == f"BROKER://{host}:9092,CONTROLLER://{host}:9093" and p["advertised.listeners"] == f"BROKER://{host}:9092"
        assert "0.0.0.0" not in p["listeners"]


def test_the_durability_settings_that_make_acks_meaningful() -> None:
    p = props(1)
    assert p["default.replication.factor"] == p["offsets.topic.replication.factor"] == p["transaction.state.log.replication.factor"] == "3"
    assert p["min.insync.replicas"] == p["transaction.state.log.min.isr"] == "2"
    assert p["unclean.leader.election.enable"] == "false"  # an out-of-date replica must never become leader and drop acknowledged data
    assert p["auto.create.topics.enable"] == "false"  # topics come from `hivemind_amcp.py provision`, with the right replication


def test_a_node_waits_ten_minutes_for_its_peers_instead_of_shutting_down_after_one() -> None:
    """Measured: with the default 60 s a lone node logged 'unable to register with the controller quorum' and exited."""
    assert props(1)["initial.broker.registration.timeout.ms"] == "600000"


def test_parse_node() -> None:
    assert cluster.parse_node("2=10.1.2.3") == cluster.Node(2, "10.1.2.3")
    for bad in ("x=1.2.3.4", "3", "3="):
        with pytest.raises(Exception):  # noqa: B017, PT011 - argparse.ArgumentTypeError
            cluster.parse_node(bad)


def test_gen_writes_one_file_per_node_and_refuses_an_even_cluster(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cluster.main(["gen", "--node", "1=10.77.0.1", "--node", "2=10.77.0.2", "--node", "3=10.77.0.3", "--out", str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["node1.properties", "node2.properties", "node3.properties"]
    assert '"tolerated_voter_failures": 1' in capsys.readouterr().out
    assert cluster.main(["gen", "--node", "1=a", "--node", "2=b", "--node", "3=c", "--node", "4=d", "--out", str(tmp_path / "even")]) == 2
    assert "odd number" in capsys.readouterr().err


def test_the_local_harness_uses_three_distinct_loopback_listeners() -> None:
    nodes = cluster.local_nodes()
    ports = [p for n in nodes for p in (n.broker_port, n.controller_port)]
    assert len(set(ports)) == 6 and all(n.host == "127.0.0.1" for n in nodes) and 9092 not in ports and 9093 not in ports  # never the shared dev broker's ports


FIVE = [cluster.Node(1, "10.77.0.1"), cluster.Node(2, "10.77.0.2"), cluster.Node(3, "10.77.0.3"), cluster.Node(4, "10.77.0.4", voter=False), cluster.Node(5, "10.77.0.5", voter=False)]


def five_props(i: int) -> dict[str, str]:
    text = cluster.node_properties(FIVE[i - 1], FIVE, "/data", replication=5, min_isr=3)
    return dict(line.split("=", 1) for line in text.splitlines() if line and not line.startswith("#"))


def test_broker_only_nodes_do_not_vote_and_do_not_count_toward_the_quorum() -> None:
    assert len(cluster.voters_of(FIVE)) == 3 and cluster.quorum_size(len(cluster.voters_of(FIVE))) == 2
    for i in (4, 5):
        p = five_props(i)
        assert p["process.roles"] == "broker" and p["listeners"] == f"BROKER://10.77.0.{i}:9092"
        assert p["controller.quorum.voters"] == "1@10.77.0.1:9093,2@10.77.0.2:9093,3@10.77.0.3:9093"  # the same voters as everyone, never the broker itself
    assert five_props(1)["process.roles"] == "broker,controller"


def test_two_voters_plus_brokers_is_not_a_quorum() -> None:
    with pytest.raises(cluster.ClusterError, match="odd number"):
        cluster.validate([cluster.Node(1, "a"), cluster.Node(2, "b"), cluster.Node(3, "c", voter=False), cluster.Node(4, "d", voter=False), cluster.Node(5, "e", voter=False)], 5, 3)


def test_replication_five_needs_min_isr_three_to_claim_two_failures() -> None:
    cluster.validate(FIVE, replication=5, min_isr=3)
    with pytest.raises(cluster.ClusterError, match="below a majority"):
        cluster.validate(FIVE, replication=5, min_isr=2)
    with pytest.raises(cluster.ClusterError, match="min_isr <= replication"):
        cluster.validate(FIVE, replication=6, min_isr=3)  # more replicas than brokers


def test_gen_reports_the_two_separate_limits(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["gen", "--out", str(tmp_path), "--replication", "5", "--min-isr", "3"]
    for n in FIVE[:3]:
        argv += ["--node", f"{n.node_id}={n.host}"]
    for n in FIVE[3:]:
        argv += ["--broker", f"{n.node_id}={n.host}"]
    assert cluster.main(argv) == 0
    out = capsys.readouterr().out
    assert '"tolerated_voter_failures": 1' in out and '"tolerated_replica_failures_per_partition": 2' in out and '"brokers": 5' in out
    assert len(list(tmp_path.iterdir())) == 5
