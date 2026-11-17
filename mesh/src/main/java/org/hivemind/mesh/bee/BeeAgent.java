package org.hivemind.mesh.bee;

import io.amcp.core.AbstractMobileAgent;
import io.amcp.core.Event;
import org.hivemind.mesh.wire.Codec;
import org.hivemind.mesh.wire.Wire;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

/**
 * One bee: an AMCP agent (identity and lifecycle through AgentContext) whose work is driven by the
 * JVM-level {@link BeeRuntime}. Holds no Kafka subscription of its own. All mutable state is guarded
 * by {@code this}.
 */
public final class BeeAgent extends AbstractMobileAgent {

    private final String beeId;
    private final List<String> tiers;
    private final Set<String> tierSet;
    private final int maxConcurrent;
    private final int queueBound;
    private final long startedAtMs;
    private final BeeRuntime runtime;
    private final Backend backend;

    private int inFlight;                                                  // guarded by this
    private final ArrayDeque<Wire.TaskAssignment> queue = new ArrayDeque<>(); // guarded by this
    private final Map<String, Wire.TaskAssignment> awaitingHelp = new HashMap<>(); // guarded by this
    private volatile boolean dead;   // set once by a scale-run kill; never reset

    BeeAgent(String beeId, List<String> tiers, int maxConcurrent, int queueBound, long startedAtMs,
             BeeRuntime runtime, Backend backend) {
        this.beeId = beeId;
        this.tiers = List.copyOf(tiers);
        this.tierSet = Set.copyOf(tiers);
        this.maxConcurrent = maxConcurrent;
        this.queueBound = queueBound;
        this.startedAtMs = startedAtMs;
        this.runtime = runtime;
        this.backend = backend;
    }

    public String beeId() {
        return beeId;
    }

    public List<String> tiers() {
        return tiers;
    }

    public int maxConcurrent() {
        return maxConcurrent;
    }

    public long startedAtMs() {
        return startedAtMs;
    }

    public synchronized int inFlight() {
        return inFlight;
    }

    public synchronized int queueDepth() {
        return queue.size();
    }

    public boolean isDead() {
        return dead;
    }

    /**
     * Simulated crash: stops heartbeats, answers and offers, and drops all held work (queued and
     * awaiting-help tasks are logged as DROP; in-flight work is dropped when its backend returns).
     * No LEAVING is sent, so the hosts can only notice through heartbeat silence.
     */
    synchronized void markDead() {
        dead = true;
        for (Wire.TaskAssignment a : queue) {
            runtime.logDrop(a, beeId, "queued");
        }
        queue.clear();
        for (Wire.TaskAssignment a : awaitingHelp.values()) {
            runtime.logDrop(a, beeId, "help");
        }
        awaitingHelp.clear();
    }

    /** Assignment for this bee (called by the JVM-level dispatcher after routing by targetBeeId). */
    synchronized void onAssignment(Wire.TaskAssignment a) {
        if (dead) {
            runtime.logDrop(a, beeId, "assign");
            return;
        }
        if (a.deadlineMs() > 0 && System.currentTimeMillis() > a.deadlineMs()) {
            runtime.publishDone(a, beeId, false, null, "deadline exceeded", 0);
            return;
        }
        boolean serves = tierSet.contains(a.tier());
        if (serves && inFlight < maxConcurrent) {
            start(a);
            return;
        }
        if (serves && a.attempt() > 1) {
            // A handed-off task that arrives while this bee is busy waits locally, it is not handed on again.
            enqueueLocal(a);
            return;
        }
        // Full, or the tier is not served here: ask the mesh for help and wait for the first offer.
        awaitingHelp.put(a.taskId(), a);
        runtime.publishHelpRequest(beeId, a.taskId(), a.tier(), serves ? "full" : "tier");
        runtime.schedule(() -> onHelpTimeout(a.taskId()), runtime.helpWaitMs());
    }

    /** The first offer for a task we are waiting on wins: hand the task to the helper. */
    synchronized void onOffer(Wire.HelpOffer o) {
        if (dead) {
            return;
        }
        Wire.TaskAssignment a = awaitingHelp.remove(o.taskId());
        if (a == null) {
            return; // someone else already won, or the wait expired
        }
        runtime.publishAssignment(new Wire.TaskAssignment(Wire.VERSION, a.taskId(), a.attempt() + 1,
                o.fromBeeId(), a.tier(), a.messages(), a.replyTopic(), a.deadlineMs(), a.hostId()));
    }

    /** True when this bee can take a helper task right now. */
    synchronized boolean canHelp(Wire.HelpRequest r) {
        return !dead && !beeId.equals(r.fromBeeId()) && tierSet.contains(r.tier())
                && inFlight < maxConcurrent && queue.isEmpty();
    }

    /** Publishes an offer for a request that canHelp() accepted. */
    void offerFor(Wire.HelpRequest r) {
        runtime.publishOffer(new Wire.HelpOffer(Wire.VERSION, beeId, r.fromBeeId(), r.taskId()));
    }

    private synchronized void onHelpTimeout(String taskId) {
        Wire.TaskAssignment a = awaitingHelp.remove(taskId);
        if (a == null) {
            return; // an offer already handed the task away
        }
        if (tierSet.contains(a.tier()) && queue.size() < queueBound) {
            enqueueLocal(a);
        } else {
            fail(a, tierSet.contains(a.tier()) ? "bee overloaded, no help offered" : "no help offered for tier");
        }
    }

    private void enqueueLocal(Wire.TaskAssignment a) {
        if (queue.size() >= queueBound) {
            fail(a, "bee queue full");
            return;
        }
        queue.add(a);
        drain();
    }

    private void drain() {
        while (inFlight < maxConcurrent && !queue.isEmpty()) {
            start(queue.poll());
        }
    }

    private void start(Wire.TaskAssignment a) {
        inFlight++;
        runtime.execute(() -> run(a));
    }

    private void run(Wire.TaskAssignment a) {
        long t0 = System.nanoTime();
        String output = null;
        String error = null;
        boolean ok = false;
        try {
            output = backend.complete(a.tier(), a.messages());
            ok = true;
        } catch (Throwable t) {
            error = t.getClass().getSimpleName() + ": " + t.getMessage();
        }
        long serviceMs = (System.nanoTime() - t0) / 1_000_000L;
        if (dead) {
            runtime.logDrop(a, beeId, "inflight"); // a crashed bee never answers
            synchronized (this) {
                inFlight--;
            }
            return;
        }
        runtime.publishDone(a, beeId, ok, output, error, serviceMs);
        synchronized (this) {
            inFlight--;
            drain();
        }
    }

    private void fail(Wire.TaskAssignment a, String error) {
        runtime.publishDone(a, beeId, false, null, error, 0);
    }

    /**
     * AMCP entry point. The runtime dispatches by typed decode, so this adapter only exists so the bee
     * is a complete AMCP agent; it routes an event to the same typed methods.
     */
    @Override
    public CompletableFuture<Void> handleEvent(Event event) {
        String topic = event.getTopic();
        if (topic.contains(".bee.inbox.s")) {
            Wire.TaskAssignment a = Codec.decode(event, Wire.TaskAssignment.class);
            if (beeId.equals(a.targetBeeId())) {
                onAssignment(a);
            }
        } else if (topic.endsWith(".bee.mesh.offer")) {
            Wire.HelpOffer o = Codec.decode(event, Wire.HelpOffer.class);
            if (beeId.equals(o.toBeeId())) {
                onOffer(o);
            }
        } else if (topic.endsWith(".bee.mesh.help")) {
            Wire.HelpRequest r = Codec.decode(event, Wire.HelpRequest.class);
            if (canHelp(r)) {
                offerFor(r);
            }
        }
        return CompletableFuture.completedFuture(null);
    }
}
