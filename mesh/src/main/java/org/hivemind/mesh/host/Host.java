package org.hivemind.mesh.host;

import org.hivemind.mesh.wire.Codec;
import org.hivemind.mesh.wire.MeshBus;
import org.hivemind.mesh.wire.Topics;
import org.hivemind.mesh.wire.Wire;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Routes tasks to bees and finalises every task.
 *
 * <p>Several hosts share the "hosts" queue for task.submit, so each task is handled by one host. Each
 * host keeps its own {@link Registry}. Only the host whose hostId matches a TaskDone finalises the task,
 * and it is the only publisher of the TaskReply, so every task is answered exactly once.
 */
public final class Host implements AutoCloseable {

    /** Maximum number of reassignments per task (so at most three attempts in total from the host). */
    public static final int MAX_REASSIGN = 2;

    private static final Logger log = LoggerFactory.getLogger(Host.class);

    public record Config(String hostId, long attemptTimeoutMs, long defaultHeartbeatMs, long announceMs) {
        public static Config defaults(String hostId) {
            return new Config(hostId, 10_000, 5_000, 2_000);
        }
    }

    private static final class Task {
        final String taskId;
        final String tier;
        final List<Wire.Message> messages;
        final String replyTopic;
        final long deadlineMs;
        final long submittedAtMs;
        final Set<String> tried = new HashSet<>();
        int attempt = 1;
        int reassignments;
        String currentBee;

        Task(Wire.TaskSubmit s, long now) {
            this.taskId = s.taskId();
            this.tier = s.tier();
            this.messages = s.messages();
            this.replyTopic = s.replyTopic();
            this.deadlineMs = s.deadlineMs();
            this.submittedAtMs = now;
        }
    }

    private final MeshBus bus;
    private final Topics topics;
    private final Config config;
    private final Registry registry = new Registry();
    private final Random random = new Random();
    private final Map<String, Task> tasks = new HashMap<>(); // guarded by this
    private final ScheduledExecutorService scheduler;

    private final AtomicLong assigned = new AtomicLong();
    private final AtomicLong completed = new AtomicLong();
    private final AtomicLong failed = new AtomicLong();
    private final AtomicLong reassigned = new AtomicLong();

    public Host(MeshBus bus, Config config) {
        this.bus = bus;
        this.topics = bus.topics();
        this.config = config;
        ScheduledThreadPoolExecutor s = new ScheduledThreadPoolExecutor(2, r -> {
            Thread t = new Thread(r, "host-scheduler");
            t.setDaemon(true);
            return t;
        });
        s.setRemoveOnCancelPolicy(true);
        this.scheduler = s;
    }

    public String hostId() {
        return config.hostId();
    }

    public int liveBees() {
        return registry.liveCount(System.currentTimeMillis());
    }

    public synchronized int inFlight() {
        return tasks.size();
    }

    public long completedCount() {
        return completed.get();
    }

    public long failedCount() {
        return failed.get();
    }

    public long reassignedCount() {
        return reassigned.get();
    }

    public void start() throws Exception {
        bus.subscribe(topics.heartbeat(), this::onHeartbeat);
        bus.subscribe(topics.lifecycle(), this::onLifecycle);
        bus.subscribe(topics.taskDone(), this::onTaskDone);
        bus.subscribeQueue(topics.taskSubmit(), Topics.HOSTS_QUEUE, this::onSubmit);
        bus.start();
        scheduler.scheduleAtFixedRate(this::announce, 1000, config.announceMs(), TimeUnit.MILLISECONDS);
        log.info("host {} started", config.hostId());
    }

    private void onHeartbeat(io.amcp.core.Event event) {
        Wire.Heartbeat hb = Codec.decode(event, Wire.Heartbeat.class);
        registry.onHeartbeat(hb, System.currentTimeMillis(), config.defaultHeartbeatMs());
    }

    private void onLifecycle(io.amcp.core.Event event) {
        Wire.Lifecycle l = Codec.decode(event, Wire.Lifecycle.class);
        if ("LEAVING".equals(l.state())) {
            registry.onLeaving(l.beeId());
        }
    }

    private void onSubmit(io.amcp.core.Event event) {
        Wire.TaskSubmit s = Codec.decode(event, Wire.TaskSubmit.class);
        long now = System.currentTimeMillis();
        synchronized (this) {
            if (tasks.containsKey(s.taskId())) {
                return; // duplicate delivery of a task we already own
            }
            Task t = new Task(s, now);
            String bee = registry.choose(t.tier, t.tried, now, random);
            if (bee == null) {
                finish(t, false, null, "no capacity", null, 1);
                return;
            }
            tasks.put(t.taskId, t);
            sendAssignment(t, bee);
        }
    }

    private void onTaskDone(io.amcp.core.Event event) {
        Wire.TaskDone d = Codec.decode(event, Wire.TaskDone.class);
        if (!config.hostId().equals(d.hostId())) {
            return;
        }
        synchronized (this) {
            Task t = tasks.get(d.taskId());
            if (t == null || d.attempt() < t.attempt) {
                return; // already finished, or a stale attempt
            }
            // A handed-off attempt (higher attempt number) is accepted as the current attempt.
            t.attempt = d.attempt();
            if (d.ok()) {
                finish(t, true, d.output(), null, d.beeId(), d.attempt());
            } else {
                reassignOrFail(t, d.error() != null ? d.error() : "no bee answered");
            }
        }
    }

    private void onTimeout(String taskId, int attempt) {
        synchronized (this) {
            Task t = tasks.get(taskId);
            if (t == null || t.attempt != attempt) {
                return; // finished, or already moved on to a later attempt
            }
            reassignOrFail(t, "no bee answered");
        }
    }

    /** Caller holds the lock. */
    private void reassignOrFail(Task t, String error) {
        if (t.reassignments < MAX_REASSIGN) {
            String bee = registry.choose(t.tier, t.tried, System.currentTimeMillis(), random);
            if (bee != null) {
                t.reassignments++;
                t.attempt++;
                reassigned.incrementAndGet();
                sendAssignment(t, bee);
                return;
            }
        }
        finish(t, false, null, error, t.currentBee, t.attempt);
    }

    /** Caller holds the lock. Publishes the assignment and arms the attempt timer. */
    private void sendAssignment(Task t, String bee) {
        t.tried.add(bee);
        t.currentBee = bee;
        registry.noteAssignment(bee);
        assigned.incrementAndGet();
        int attempt = t.attempt;
        bus.publish(topics.inboxFor(bee), bee, new Wire.TaskAssignment(Wire.VERSION, t.taskId, attempt, bee,
                t.tier, t.messages, t.replyTopic, t.deadlineMs, config.hostId()));
        scheduler.schedule(() -> onTimeout(t.taskId, attempt), config.attemptTimeoutMs(), TimeUnit.MILLISECONDS);
    }

    /** Caller holds the lock. Publishes the single TaskReply for this task. */
    private void finish(Task t, boolean ok, String output, String error, String beeId, int attempts) {
        tasks.remove(t.taskId);
        if (ok) {
            completed.incrementAndGet();
        } else {
            failed.incrementAndGet();
        }
        long e2e = System.currentTimeMillis() - t.submittedAtMs;
        bus.publish(t.replyTopic, t.taskId, new Wire.TaskReply(Wire.VERSION, t.taskId, ok, output, error,
                beeId, attempts, e2e));
    }

    private void announce() {
        try {
            long now = System.currentTimeMillis();
            bus.publish(topics.hostAnnounce(), config.hostId(), new Wire.HostAnnounce(Wire.VERSION,
                    config.hostId(), registry.liveCount(now), registry.liveByTier(now), inFlight(),
                    assigned.get(), completed.get(), failed.get(), reassigned.get(), now));
        } catch (RuntimeException e) {
            log.warn("announce failed: {}", e.toString());
        }
    }

    @Override
    public void close() {
        scheduler.shutdownNow();
    }
}
