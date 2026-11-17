package org.hivemind.mesh.host;

import org.hivemind.mesh.wire.Wire;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Unit tests for the routing policy: no Kafka involved. */
class RegistryTest {

    private static final long HB = 5_000;
    private static final long T0 = 1_000_000L;

    private static Wire.Heartbeat hb(String bee, List<String> tiers, int inFlight, int max) {
        return new Wire.Heartbeat(1, bee, tiers, inFlight, max, 0, "local", T0, T0, HB);
    }

    @Test
    void routesToTheLeastLoadedLiveBee() {
        Registry r = new Registry();
        r.onHeartbeat(hb("busy", List.of("coder"), 3, 4), T0, HB);   // score 0.75
        r.onHeartbeat(hb("idle", List.of("coder"), 1, 4), T0, HB);   // score 0.25
        r.onHeartbeat(hb("mid", List.of("coder"), 2, 4), T0, HB);    // score 0.5
        assertEquals("idle", r.choose("coder", Set.of(), T0 + 1, new Random(1)));
    }

    @Test
    void assignmentsSinceTheLastHeartbeatCountTowardLoad() {
        Registry r = new Registry();
        r.onHeartbeat(hb("a", List.of("coder"), 0, 2), T0, HB);
        r.onHeartbeat(hb("b", List.of("coder"), 1, 2), T0, HB);
        assertEquals("a", r.choose("coder", Set.of(), T0 + 1, new Random(1)));
        r.noteAssignment("a");
        r.noteAssignment("a");
        // a: (0 + 2) / 2 = 1.0, b: (1 + 0) / 2 = 0.5
        assertEquals("b", r.choose("coder", Set.of(), T0 + 1, new Random(1)));
        // A new heartbeat resets this host's own assignment count.
        r.onHeartbeat(hb("a", List.of("coder"), 0, 2), T0 + 100, HB);
        assertEquals("a", r.choose("coder", Set.of(), T0 + 101, new Random(1)));
    }

    @Test
    void tiesAreBrokenAtRandom() {
        Registry r = new Registry();
        r.onHeartbeat(hb("a", List.of("coder"), 0, 4), T0, HB);
        r.onHeartbeat(hb("b", List.of("coder"), 0, 4), T0, HB);
        Random rnd = new Random(42);
        Set<String> picked = new java.util.HashSet<>();
        for (int i = 0; i < 200; i++) {
            picked.add(r.choose("coder", Set.of(), T0 + 1, rnd));
        }
        assertEquals(Set.of("a", "b"), picked);
    }

    @Test
    void beeSilentForThreeIntervalsIsDeadAndExcluded() {
        Registry r = new Registry();
        r.onHeartbeat(hb("a", List.of("coder"), 0, 4), T0, HB);
        long justBefore = T0 + 3 * HB - 1;   // 14 999 ms of silence: still alive
        long atLimit = T0 + 3 * HB;          // 15 000 ms: dead
        assertTrue(r.isLive("a", justBefore));
        assertEquals("a", r.choose("coder", Set.of(), justBefore, new Random(1)));
        assertNotEquals(true, r.isLive("a", atLimit));
        assertNull(r.choose("coder", Set.of(), atLimit, new Random(1)));
        assertEquals(0, r.liveCount(atLimit));
    }

    @Test
    void aFreshHeartbeatRevivesADeadBee() {
        Registry r = new Registry();
        r.onHeartbeat(hb("a", List.of("coder"), 0, 4), T0, HB);
        assertNull(r.choose("coder", Set.of(), T0 + 20_000, new Random(1)));
        r.onHeartbeat(hb("a", List.of("coder"), 0, 4), T0 + 20_000, HB);
        assertEquals("a", r.choose("coder", Set.of(), T0 + 20_001, new Random(1)));
    }

    @Test
    void reassignmentExcludesTheFailedBee() {
        Registry r = new Registry();
        r.onHeartbeat(hb("failed", List.of("coder"), 0, 4), T0, HB);
        r.onHeartbeat(hb("other", List.of("coder"), 3, 4), T0, HB);
        // The failed bee is the least loaded, but excluding it must pick the other one.
        assertEquals("failed", r.choose("coder", Set.of(), T0 + 1, new Random(1)));
        assertEquals("other", r.choose("coder", Set.of("failed"), T0 + 1, new Random(1)));
        // With every candidate excluded there is nowhere left to reassign.
        assertNull(r.choose("coder", Set.of("failed", "other"), T0 + 1, new Random(1)));
    }

    @Test
    void noCapacityWhenNoLiveBeeServesTheTier() {
        Registry r = new Registry();
        r.onHeartbeat(hb("g", List.of("general"), 0, 4), T0, HB);
        assertNull(r.choose("coder", Set.of(), T0 + 1, new Random(1)));
        assertEquals("g", r.choose("general", Set.of(), T0 + 1, new Random(1)));
        assertEquals(java.util.Map.of("general", 1), r.liveByTier(T0 + 1));
    }

    @Test
    void leavingRemovesTheBeeImmediately() {
        Registry r = new Registry();
        r.onHeartbeat(hb("a", List.of("coder"), 0, 4), T0, HB);
        r.onLeaving("a");
        assertNull(r.choose("coder", Set.of(), T0 + 1, new Random(1)));
        assertEquals(0, r.liveCount(T0 + 1));
    }

    @Test
    void liveCountOverThousandBees() {
        Registry r = new Registry();
        for (int i = 0; i < 1000; i++) {
            r.onHeartbeat(hb("bee-" + i, List.of("coder"), i % 4, 4), T0, HB);
        }
        assertEquals(1000, r.liveCount(T0 + 1));
        assertEquals(1000, r.liveByTier(T0 + 1).get("coder"));
    }
}
