package org.hivemind.mesh.bee;

import io.amcp.core.AgentContext;
import org.hivemind.mesh.wire.Codec;
import org.hivemind.mesh.wire.MeshBus;
import org.hivemind.mesh.wire.Topics;
import org.hivemind.mesh.wire.Wire;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.IntFunction;

/**
 * Runs many bees in one JVM on one {@link MeshBus}. The JVM registers every bee as an AMCP agent,
 * subscribes once per inbox shard that holds one of its bees (plus help and offer, once), and
 * dispatches each inbox event to its bee through a map keyed by beeId. Heartbeats for all bees are
 * sent by one scheduled sweep, not by one thread or timer per bee.
 */
public final class BeeRuntime implements AutoCloseable {

    /** Per-JVM configuration shared by the bees it runs. */
    public record Config(List<String> tiers, int maxConcurrent, String region, long heartbeatMs,
                         long helpWaitMs, int queueBound) {
        public Config {
            tiers = List.copyOf(tiers);
            if (tiers.isEmpty()) {
                throw new IllegalArgumentException("at least one tier");
            }
            if (maxConcurrent < 1 || heartbeatMs < 1 || helpWaitMs < 0 || queueBound < 0) {
                throw new IllegalArgumentException("invalid bee configuration");
            }
        }
    }

    private static final Logger log = LoggerFactory.getLogger(BeeRuntime.class);

    /** AgentContext keys are "SimpleName-currentTimeMillis", so registrations must be 1 ms apart. */
    private static final Object REG_LOCK = new Object();
    private static long lastRegMs;

    private final MeshBus bus;
    private final Topics topics;
    private final AgentContext context;
    private final Config config;
    private final Map<String, BeeAgent> bees = new LinkedHashMap<>();
    private final Map<String, String> agentKeys = new HashMap<>();
    private final ScheduledExecutorService scheduler;
    private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();
    private ScheduledFuture<?> heartbeatTask;

    // Scale-run hooks (all optional; defaults mean "no kill, no event log").
    private long killAfterS;
    private double killFraction;
    private long killSeed = 20261027L;
    private EventLog events;
    private final AtomicBoolean killArmed = new AtomicBoolean();

    /**
     * Creates and registers {@code count} bees with ids {@code prefix-00000 ...}. The backend factory
     * gives each bee its model (for example a SimBackend with a per-bee seed).
     */
    public BeeRuntime(MeshBus bus, Config config, int count, String prefix, IntFunction<Backend> backends) {
        this.bus = bus;
        this.topics = bus.topics();
        this.context = bus.context();
        this.config = config;
        ScheduledThreadPoolExecutor s = new ScheduledThreadPoolExecutor(2, r -> {
            Thread t = new Thread(r, "bee-scheduler");
            t.setDaemon(true);
            return t;
        });
        s.setRemoveOnCancelPolicy(true);
        this.scheduler = s;
        long now = System.currentTimeMillis();
        for (int i = 0; i < count; i++) {
            String beeId = String.format("%s-%05d", prefix, i);
            BeeAgent bee = new BeeAgent(beeId, config.tiers(), config.maxConcurrent(), config.queueBound(),
                    now, this, backends.apply(i));
            String key = register(bee);
            bee.setAgentId(beeId);
            bees.put(beeId, bee);
            agentKeys.put(beeId, key);
        }
        if (context.getAgents().size() != count) {
            throw new IllegalStateException("agent registry holds " + context.getAgents().size()
                    + " agents, expected " + count);
        }
    }

    /**
     * Registers a bee with the AgentContext. registerAgent() keys agents by class name plus the clock in
     * milliseconds, so two bees registered in the same millisecond would overwrite each other. Waiting
     * until the clock has moved past the previous registration keeps every key distinct.
     */
    private String register(BeeAgent bee) {
        synchronized (REG_LOCK) {
            while (System.currentTimeMillis() <= lastRegMs) {
                Thread.onSpinWait();
            }
            String key = context.registerAgent(bee);
            lastRegMs = System.currentTimeMillis();
            return key;
        }
    }

    /**
     * Scale-run kill: when the load-start marker arrives, at startMs + killAfterS seconds a random
     * {@code fraction} of this JVM's bees stop heartbeating and answering, without LEAVING. Must be
     * called before start(). A killAfterS of 0 disables the kill.
     */
    public void configureKill(long killAfterS, double fraction, long seed) {
        if (killAfterS < 0 || fraction < 0 || fraction > 1) {
            throw new IllegalArgumentException("invalid kill configuration");
        }
        this.killAfterS = killAfterS;
        this.killFraction = fraction;
        this.killSeed = seed;
    }

    /** Scale-run event log (ASSIGN, DROP, KILL lines). Must be called before start(). */
    public void setEventLog(EventLog log) {
        this.events = log;
    }

    public List<String> beeIds() {
        return new ArrayList<>(bees.keySet());
    }

    public int beeCount() {
        return bees.size();
    }

    /** Subscribes, starts the bus, activates the bees, announces JOINING and starts heartbeats. */
    public void start() throws Exception {
        Set<Integer> shards = new TreeSet<>();
        for (String beeId : bees.keySet()) {
            shards.add(Topics.shardOf(beeId));
        }
        for (int shard : shards) {
            bus.subscribe(topics.inboxShard(shard), this::onInbox);
        }
        bus.subscribe(topics.help(), this::onHelp);
        bus.subscribe(topics.offer(), this::onOffer);
        if (killAfterS > 0) {
            bus.subscribe(topics.control(), this::onControl);
        }
        bus.start();
        if (events != null) {
            scheduler.scheduleAtFixedRate(events::flush, 1, 1, TimeUnit.SECONDS);
        }
        for (BeeAgent bee : bees.values()) {
            context.activateAgent(agentKeys.get(bee.beeId()));
        }
        long now = System.currentTimeMillis();
        for (BeeAgent bee : bees.values()) {
            send(topics.lifecycle(), bee.beeId(),
                    new Wire.Lifecycle(Wire.VERSION, bee.beeId(), "JOINING", now));
        }
        heartbeatTask = scheduler.scheduleAtFixedRate(this::heartbeatSweep, 0, config.heartbeatMs(),
                TimeUnit.MILLISECONDS);
        log.info("{} bees joined on {} inbox shards", bees.size(), shards.size());
    }

    private void heartbeatSweep() {
        try {
            long now = System.currentTimeMillis();
            for (BeeAgent bee : bees.values()) {
                if (bee.isDead()) {
                    continue; // a killed bee stops heartbeating
                }
                send(topics.heartbeat(), bee.beeId(), new Wire.Heartbeat(Wire.VERSION, bee.beeId(),
                        config.tiers(), bee.inFlight(), config.maxConcurrent(), bee.queueDepth(),
                        config.region(), bee.startedAtMs(), now, config.heartbeatMs()));
            }
        } catch (RuntimeException e) {
            log.warn("heartbeat sweep failed: {}", e.toString());
        }
    }

    private void onInbox(io.amcp.core.Event event) {
        Wire.TaskAssignment a = Codec.decode(event, Wire.TaskAssignment.class);
        BeeAgent bee = bees.get(a.targetBeeId());
        if (bee != null) {
            // Logged before dispatch, so an assignment to a killed bee is still on record.
            logLine("ASSIGN " + a.taskId() + " " + a.attempt() + " " + bee.beeId() + " " + System.currentTimeMillis());
            bee.onAssignment(a);
        }
    }

    /** Start-of-load marker from the load generator: arms the kill, if one is configured. */
    private void onControl(io.amcp.core.Event event) {
        Wire.LoadStart s = Codec.decode(event, Wire.LoadStart.class);
        if (!killArmed.compareAndSet(false, true)) {
            return;
        }
        long killAt = s.startMs() + killAfterS * 1000L;
        long delay = Math.max(0, killAt - System.currentTimeMillis());
        log.info("kill armed by {}: {} bees at +{} s (in {} ms)", s.loadgenId(), killFraction, killAfterS, delay);
        scheduler.schedule(this::kill, delay, TimeUnit.MILLISECONDS);
    }

    /** Kills round(fraction * bees) bees chosen with the kill seed. Killed bees are never told to LEAVE. */
    synchronized void kill() {
        List<String> ids = new ArrayList<>(bees.keySet());
        Collections.shuffle(ids, new Random(killSeed));
        int k = (int) Math.round(killFraction * ids.size());
        long now = System.currentTimeMillis();
        for (int i = 0; i < k; i++) {
            BeeAgent bee = bees.get(ids.get(i));
            bee.markDead();
            logLine("KILL " + bee.beeId() + " " + now);
        }
        if (events != null) {
            events.flush();
        }
        log.info("killed {} of {} bees", k, ids.size());
    }

    /** Logs a suppressed unit of work of a killed bee. */
    void logDrop(Wire.TaskAssignment a, String beeId, String phase) {
        logLine("DROP " + a.taskId() + " " + beeId + " " + a.attempt() + " " + phase + " " + System.currentTimeMillis());
    }

    private void logLine(String text) {
        if (events != null) {
            events.line(text);
        }
    }

    private void onHelp(io.amcp.core.Event event) {
        Wire.HelpRequest r = Codec.decode(event, Wire.HelpRequest.class);
        for (BeeAgent bee : bees.values()) {
            if (bee.canHelp(r)) {
                bee.offerFor(r);
                return; // one offer per JVM; the requester takes the first offer it gets
            }
        }
    }

    private void onOffer(io.amcp.core.Event event) {
        Wire.HelpOffer o = Codec.decode(event, Wire.HelpOffer.class);
        BeeAgent bee = bees.get(o.toBeeId());
        if (bee != null) {
            bee.onOffer(o);
        }
    }

    // ---- publishing, called by the bees ----

    void publishDone(Wire.TaskAssignment a, String beeId, boolean ok, String output, String error,
                     long serviceMs) {
        send(topics.taskDone(), a.taskId(), new Wire.TaskDone(Wire.VERSION, a.taskId(), a.attempt(), beeId,
                ok, output, error, serviceMs, a.hostId()));
    }

    void publishAssignment(Wire.TaskAssignment a) {
        send(topics.inboxFor(a.targetBeeId()), a.targetBeeId(), a);
    }

    void publishHelpRequest(String fromBeeId, String taskId, String tier, String reason) {
        send(topics.help(), taskId, new Wire.HelpRequest(Wire.VERSION, fromBeeId, taskId, tier, reason));
    }

    void publishOffer(Wire.HelpOffer o) {
        send(topics.offer(), o.taskId(), o);
    }

    private void send(String topic, String key, Object payload) {
        bus.publish(topic, key, payload).whenComplete((v, e) -> {
            if (e != null) {
                log.warn("publish to {} failed: {}", topic, e.toString());
            }
        });
    }

    void schedule(Runnable task, long delayMs) {
        scheduler.schedule(task, delayMs, TimeUnit.MILLISECONDS);
    }

    void execute(Runnable task) {
        workers.execute(task);
    }

    long helpWaitMs() {
        return config.helpWaitMs();
    }

    /** Sends LEAVING for every bee, stops the sweep and the workers, deactivates the agents. */
    @Override
    public void close() {
        List<CompletableFuture<Void>> leaving = new ArrayList<>();
        long now = System.currentTimeMillis();
        for (BeeAgent bee : bees.values()) {
            leaving.add(bus.publish(topics.lifecycle(), bee.beeId(),
                    new Wire.Lifecycle(Wire.VERSION, bee.beeId(), "LEAVING", now)));
        }
        try {
            CompletableFuture.allOf(leaving.toArray(new CompletableFuture[0])).get(10, TimeUnit.SECONDS);
        } catch (Exception e) {
            log.warn("LEAVING not acknowledged for every bee: {}", e.toString());
        }
        if (heartbeatTask != null) {
            heartbeatTask.cancel(false);
        }
        scheduler.shutdownNow();
        workers.shutdownNow();
        if (events != null) {
            events.close();
        }
        for (String key : agentKeys.values()) {
            try {
                context.deactivateAgent(key);
            } catch (RuntimeException e) {
                log.debug("deactivate {}: {}", key, e.toString());
            }
        }
    }
}
