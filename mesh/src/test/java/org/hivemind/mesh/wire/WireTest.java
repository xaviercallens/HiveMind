package org.hivemind.mesh.wire;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WireTest {

    private static final List<Wire.Message> MESSAGES = List.of(
            new Wire.Message("system", "be brief"),
            new Wire.Message("user", "hello \"quoted\" ünïcode"));

    @Test
    void everyMessageTypeRoundTrips() {
        Wire.Heartbeat hb = new Wire.Heartbeat(1, "bee-1", List.of("coder", "general"), 2, 4, 1, "eu-west",
                1_700_000_000_000L, 1_700_000_005_000L, 5000);
        assertRoundTrip(hb, Wire.Heartbeat.class);

        assertRoundTrip(new Wire.Lifecycle(1, "bee-1", "JOINING", 42L), Wire.Lifecycle.class);
        assertRoundTrip(new Wire.Lifecycle(1, "bee-1", "LEAVING", 43L), Wire.Lifecycle.class);

        assertRoundTrip(new Wire.TaskAssignment(1, "t-1", 2, "bee-9", "coder", MESSAGES, "hivemind.user.reply.gw",
                1_700_000_030_000L, "host-a"), Wire.TaskAssignment.class);

        assertRoundTrip(new Wire.TaskSubmit(1, "t-1", "coder", MESSAGES, "hivemind.user.reply.gw",
                1_700_000_030_000L, "user-7"), Wire.TaskSubmit.class);
        assertRoundTrip(new Wire.TaskSubmit(1, "t-2", "general", MESSAGES, "r", 1L, null), Wire.TaskSubmit.class);

        assertRoundTrip(new Wire.TaskDone(1, "t-1", 2, "bee-9", true, "out ☃", null, 17, "host-a"),
                Wire.TaskDone.class);
        assertRoundTrip(new Wire.TaskDone(1, "t-1", 1, "bee-9", false, null, "boom", 0, "host-a"),
                Wire.TaskDone.class);

        assertRoundTrip(new Wire.TaskReply(1, "t-1", true, "answer", null, "bee-9", 2, 123), Wire.TaskReply.class);

        assertRoundTrip(new Wire.HelpRequest(1, "bee-1", "t-1", "coder", "full"), Wire.HelpRequest.class);
        assertRoundTrip(new Wire.HelpOffer(1, "bee-2", "bee-1", "t-1"), Wire.HelpOffer.class);

        assertRoundTrip(new Wire.HostAnnounce(1, "host-a", 5, Map.of("coder", 5), 3, 10, 7, 1, 2, 99L),
                Wire.HostAnnounce.class);
    }

    @Test
    void unknownFieldsAreIgnoredForForwardCompatibility() {
        String json = "{\"v\":1,\"beeId\":\"b\",\"state\":\"JOINING\",\"atMs\":5,\"extra\":true}";
        Wire.Lifecycle l = Codec.decode(json.getBytes(StandardCharsets.UTF_8), Wire.Lifecycle.class);
        assertEquals("b", l.beeId());
    }

    @Test
    void rejectsOtherProtocolVersions() {
        byte[] json = "{\"v\":2,\"beeId\":\"b\",\"state\":\"JOINING\",\"atMs\":5}".getBytes(StandardCharsets.UTF_8);
        assertThrows(IllegalArgumentException.class, () -> Codec.decode(json, Wire.Lifecycle.class));
    }

    @Test
    void rejectsMalformedJson() {
        byte[] json = "not json".getBytes(StandardCharsets.UTF_8);
        assertThrows(IllegalArgumentException.class, () -> Codec.decode(json, Wire.Lifecycle.class));
    }

    @Test
    void shardFunctionIsStable() {
        // String.hashCode("a") == 97 and 97 mod 32 == 1; the empty string hashes to 0.
        assertEquals(1, Topics.shardOf("a"));
        assertEquals(0, Topics.shardOf(""));
        // Negative hash codes must still map into 0..31 (floorMod, not %).
        for (int i = 0; i < 10_000; i++) {
            int s = Topics.shardOf("bee-" + i);
            assertTrue(s >= 0 && s < Topics.SHARDS, "shard out of range: " + s);
            assertEquals(s, Topics.shardOf("bee-" + i), "shard must be deterministic");
        }
        Set<Integer> seen = new java.util.HashSet<>();
        for (int i = 0; i < 10_000; i++) {
            seen.add(Topics.shardOf("bee-" + i));
        }
        assertEquals(Topics.SHARDS, seen.size(), "10k bee ids should cover all 32 shards");
    }

    @Test
    void topicNamesFollowTheProtocol() {
        Topics t = Topics.defaults();
        assertEquals("hivemind.bee.heartbeat", t.heartbeat());
        assertEquals("hivemind.bee.lifecycle", t.lifecycle());
        assertEquals("hivemind.bee.inbox.s01", t.inboxShard(1));
        assertEquals("hivemind.bee.inbox.s31", t.inboxShard(31));
        assertEquals("hivemind.task.submit", t.taskSubmit());
        assertEquals("hivemind.task.done", t.taskDone());
        assertEquals("hivemind.user.reply.gw1", t.userReply("gw1"));
        assertEquals("hivemind.bee.mesh.help", t.help());
        assertEquals("hivemind.bee.mesh.offer", t.offer());
        assertEquals("hivemind.host.announce", t.hostAnnounce());
        assertEquals(t.inboxShard(Topics.shardOf("bee-x")), t.inboxFor("bee-x"));
    }

    @Test
    void rootIsValidatedAndConfigurable() {
        assertEquals("hmitabc123.bee.heartbeat", new Topics("hmitabc123").heartbeat());
        assertThrows(IllegalArgumentException.class, () -> new Topics("bad.root"));
        assertThrows(IllegalArgumentException.class, () -> new Topics(""));
        assertThrows(IllegalArgumentException.class, () -> Topics.defaults().inboxShard(32));
    }

    @Test
    void wireJsonUsesTheProtocolFieldNames() {
        String json = new String(Codec.encode(new Wire.HelpOffer(1, "bee-2", "bee-1", "t-1")),
                StandardCharsets.UTF_8);
        assertTrue(json.contains("\"v\":1"), json);
        assertTrue(json.contains("\"fromBeeId\":\"bee-2\""), json);
        assertTrue(json.contains("\"toBeeId\":\"bee-1\""), json);
        assertFalse(json.contains("\"partitionkey\""), json);
    }

    private static <T extends Wire.Versioned> void assertRoundTrip(T value, Class<T> type) {
        byte[] json = Codec.encode(value);
        T back = Codec.decode(json, type);
        assertEquals(value, back, "round trip of " + type.getSimpleName());
    }
}
