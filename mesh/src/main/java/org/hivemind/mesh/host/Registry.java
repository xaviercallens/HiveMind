package org.hivemind.mesh.host;

import org.hivemind.mesh.wire.Wire;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

/**
 * The host's view of the bees, built from heartbeats and lifecycle events, plus the routing policy.
 *
 * <p>A bee is dead when it has been silent for three heartbeat intervals (silence &gt;= 3 x interval)
 * or after LEAVING. Routing picks the live bee that serves the tier with the lowest load score,
 * {@code (inFlight from its latest heartbeat + this host's assignments since that heartbeat) / maxConcurrent};
 * ties are broken at random. Pure logic: no I/O, so the policy is unit-tested directly.
 */
public final class Registry {

    /** Silent for this many heartbeat intervals means dead. */
    public static final int DEAD_AFTER_INTERVALS = 3;

    private static final class Entry {
        final String beeId;
        Set<String> tiers;
        int maxConcurrent;
        int inFlightAtHeartbeat;
        int assignedSinceHeartbeat;
        long lastSeenMs;
        long heartbeatMs;

        Entry(String beeId) {
            this.beeId = beeId;
        }
    }

    private final Map<String, Entry> entries = new HashMap<>();

    public synchronized void onHeartbeat(Wire.Heartbeat hb, long nowMs, long defaultHeartbeatMs) {
        Entry e = entries.computeIfAbsent(hb.beeId(), Entry::new);
        e.tiers = Set.copyOf(hb.tiers());
        e.maxConcurrent = Math.max(1, hb.maxConcurrent());
        e.inFlightAtHeartbeat = hb.inFlight();
        e.assignedSinceHeartbeat = 0;
        e.lastSeenMs = nowMs;
        e.heartbeatMs = hb.heartbeatMs() > 0 ? hb.heartbeatMs() : defaultHeartbeatMs;
    }

    public synchronized void onLeaving(String beeId) {
        entries.remove(beeId);
    }

    /** Counts an assignment this host just sent, so the next routing decision sees it. */
    public synchronized void noteAssignment(String beeId) {
        Entry e = entries.get(beeId);
        if (e != null) {
            e.assignedSinceHeartbeat++;
        }
    }

    public synchronized boolean isLive(String beeId, long nowMs) {
        Entry e = entries.get(beeId);
        return e != null && alive(e, nowMs);
    }

    public synchronized int liveCount(long nowMs) {
        int n = 0;
        for (Entry e : entries.values()) {
            if (alive(e, nowMs)) {
                n++;
            }
        }
        return n;
    }

    public synchronized Map<String, Integer> liveByTier(long nowMs) {
        Map<String, Integer> out = new TreeMap<>();
        for (Entry e : entries.values()) {
            if (alive(e, nowMs)) {
                for (String tier : e.tiers) {
                    out.merge(tier, 1, Integer::sum);
                }
            }
        }
        return out;
    }

    /**
     * Least-loaded live bee serving {@code tier}, skipping {@code exclude}. Returns null when no such
     * bee exists (the caller turns that into "no capacity" or "no bee answered").
     */
    public synchronized String choose(String tier, Set<String> exclude, long nowMs, Random random) {
        double best = Double.MAX_VALUE;
        List<String> ties = new ArrayList<>();
        for (Entry e : entries.values()) {
            if (!alive(e, nowMs) || !e.tiers.contains(tier) || exclude.contains(e.beeId)) {
                continue;
            }
            double score = (double) (e.inFlightAtHeartbeat + e.assignedSinceHeartbeat) / e.maxConcurrent;
            if (score < best) {
                best = score;
                ties.clear();
                ties.add(e.beeId);
            } else if (score == best) {
                ties.add(e.beeId);
            }
        }
        if (ties.isEmpty()) {
            return null;
        }
        return ties.get(ties.size() == 1 ? 0 : random.nextInt(ties.size()));
    }

    private static boolean alive(Entry e, long nowMs) {
        return nowMs - e.lastSeenMs < DEAD_AFTER_INTERVALS * e.heartbeatMs;
    }
}
