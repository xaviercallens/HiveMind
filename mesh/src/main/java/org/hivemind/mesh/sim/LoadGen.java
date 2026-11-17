package org.hivemind.mesh.sim;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.hivemind.mesh.wire.Codec;
import org.hivemind.mesh.wire.MeshBus;
import org.hivemind.mesh.wire.Topics;
import org.hivemind.mesh.wire.Wire;

import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;

/**
 * Open-loop load generator for scale runs. Publishes TaskSubmit events directly on the task topic at a
 * steady rate (task i is submitted at start + i / rate, never in bursts), consumes its reply topic and
 * the task.done topic, and writes one JSON line per task plus one summary line.
 *
 * <p>A task with no reply within {@link #LOST_AFTER_MS} after the last submission is "lost". The
 * output is written only after the wait, so serviceMs (from the winning TaskDone) is joined in memory.
 */
public final class LoadGen {

    /** No reply within this long after the end of the load means the task is lost. */
    public static final long LOST_AFTER_MS = 30_000;
    /** Task deadline, as the gateway uses (30 s). */
    public static final long TASK_DEADLINE_MS = 30_000;
    /** Extra time after the last reply for TaskDone records to arrive (serviceMs join). */
    private static final long SERVICE_GRACE_MS = 2_000;
    /** Harness probes carry this taskId prefix; they are never tasks (MESH_SCALE_prereg_amendment_3.json). */
    static final String PROBE_PREFIX = "probe-";
    /** hostId of probe TaskDone records: no host matches it, so no host finalises it. */
    static final String PROBE_HOST = "__probe__";
    private static final long BARRIER_LIMIT_MS = 60_000;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private record Sent(String taskId, long submitMs) {
    }

    private record Got(Wire.TaskReply reply, long replyMs) {
    }

    private LoadGen() {
    }

    /**
     * Runs one load. Returns after the JSONL file is written.
     *
     * @param rate           target submissions per second
     * @param durationS      length of the submission window in seconds
     * @param tier           task tier
     * @param out            JSONL output file
     * @param gatewayId      suffix of the reply topic root.user.reply.GATEWAY_ID
     */
    public static int run(String bootstrap, Topics topics, double rate, long durationS, String tier, Path out,
                          String gatewayId) throws Exception {
        if (rate <= 0 || durationS <= 0) {
            throw new IllegalArgumentException("rate and duration must be positive");
        }
        long total = Math.round(rate * durationS);
        String replyTopic = topics.userReply(gatewayId);
        MeshBus bus = new MeshBus(bootstrap, topics, "lg-" + UUID.randomUUID());

        Map<String, Got> replies = new ConcurrentHashMap<>();
        Map<String, Long> serviceMs = new ConcurrentHashMap<>();   // key taskId|beeId|attempt
        AtomicLong duplicates = new AtomicLong();
        CountDownLatch replyProbeSeen = new CountDownLatch(1);
        CountDownLatch doneProbeSeen = new CountDownLatch(1);

        bus.subscribe(replyTopic, e -> {
            Wire.TaskReply r = Codec.decode(e, Wire.TaskReply.class);
            if (r.taskId().startsWith(PROBE_PREFIX)) {
                replyProbeSeen.countDown();
                return;
            }
            if (replies.putIfAbsent(r.taskId(), new Got(r, System.currentTimeMillis())) != null) {
                duplicates.incrementAndGet();
            }
        });
        bus.subscribe(topics.taskDone(), e -> {
            Wire.TaskDone d = Codec.decode(e, Wire.TaskDone.class);
            if (d.taskId().startsWith(PROBE_PREFIX)) {
                doneProbeSeen.countDown();
                return;
            }
            if (d.ok()) {
                serviceMs.putIfAbsent(d.taskId() + "|" + d.beeId() + "|" + d.attempt(), d.serviceMs());
            }
        });
        bus.start();

        // Readiness barrier (MESH_SCALE_prereg_amendment_3.json): probes are not tasks and are not counted.
        long barrierStart = System.currentTimeMillis();
        String probeId = PROBE_PREFIX + UUID.randomUUID();
        Wire.TaskReply probeReply = new Wire.TaskReply(Wire.VERSION, probeId, false, null, "probe", null, 0, 0);
        Wire.TaskDone probeDone = new Wire.TaskDone(Wire.VERSION, probeId, 0, "probe", false, null, "probe", 0,
                PROBE_HOST);
        while (replyProbeSeen.getCount() > 0 || doneProbeSeen.getCount() > 0) {
            if (System.currentTimeMillis() - barrierStart > BARRIER_LIMIT_MS) {
                throw new IllegalStateException("readiness barrier: own subscriptions not live after "
                        + BARRIER_LIMIT_MS + " ms");
            }
            bus.publish(replyTopic, probeId, probeReply).get(10, TimeUnit.SECONDS);
            bus.publish(topics.taskDone(), probeId, probeDone).get(10, TimeUnit.SECONDS);
            replyProbeSeen.await(500, TimeUnit.MILLISECONDS);
            doneProbeSeen.await(500, TimeUnit.MILLISECONDS);
        }
        long barrierMs = System.currentTimeMillis() - barrierStart;

        long startMs = System.currentTimeMillis();
        bus.publish(topics.control(), gatewayId, new Wire.LoadStart(Wire.VERSION, gatewayId, startMs))
                .get(30, TimeUnit.SECONDS);

        List<Sent> sent = new ArrayList<>((int) Math.min(total, Integer.MAX_VALUE));
        AtomicLong submitFailures = new AtomicLong();
        Map<String, String> submitErrors = new ConcurrentHashMap<>();
        long t0 = System.nanoTime();
        for (long i = 0; i < total; i++) {
            long target = t0 + (long) (i * 1e9 / rate);
            long now;
            while ((now = System.nanoTime()) < target) {
                LockSupport.parkNanos(target - now);
            }
            String taskId = UUID.randomUUID().toString();
            long submitMs = System.currentTimeMillis();
            sent.add(new Sent(taskId, submitMs));
            Wire.TaskSubmit submit = new Wire.TaskSubmit(Wire.VERSION, taskId, tier,
                    List.of(new Wire.Message("user", "loadgen task " + i)), replyTopic,
                    submitMs + TASK_DEADLINE_MS, null);
            bus.publish(topics.taskSubmit(), taskId, submit).whenComplete((v, err) -> {
                if (err != null) {
                    submitFailures.incrementAndGet();
                    submitErrors.put(taskId, err.toString());
                }
            });
        }
        long loopEnd = System.nanoTime();
        double loopS = (loopEnd - t0) / 1e9;

        // Wait for every task to get a reply, at most LOST_AFTER_MS after the end of the load.
        long lastSubmitEnd = System.currentTimeMillis();
        long giveUpAt = lastSubmitEnd + LOST_AFTER_MS;
        while (System.currentTimeMillis() < giveUpAt && replies.size() + submitFailures.get() < total) {
            Thread.sleep(100);
        }
        // Short grace so TaskDone records for the replied tasks can be joined.
        long graceEnd = System.currentTimeMillis() + SERVICE_GRACE_MS;
        while (System.currentTimeMillis() < graceEnd && !allServiceKnown(replies, serviceMs)) {
            Thread.sleep(100);
        }

        long endMs = System.currentTimeMillis();
        long replied = 0;
        long ok = 0;
        long lost = 0;
        long serviceMissing = 0;
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            for (Sent s : sent) {
                Got g = replies.get(s.taskId());
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("taskId", s.taskId());
                row.put("submitMs", s.submitMs());
                if (g == null) {
                    lost++;
                    row.put("replyMs", null);
                    row.put("e2eMs", null);
                    row.put("ok", false);
                    row.put("beeId", null);
                    row.put("attempts", null);
                    row.put("serviceMs", null);
                    row.put("error", null);
                    row.put("lost", true);
                    row.put("submitError", submitErrors.get(s.taskId()));
                } else {
                    replied++;
                    Wire.TaskReply r = g.reply();
                    Long svc = r.ok() ? serviceMs.get(r.taskId() + "|" + r.beeId() + "|" + r.attempts()) : null;
                    if (r.ok()) {
                        ok++;
                        if (svc == null) {
                            serviceMissing++;
                        }
                    }
                    row.put("replyMs", g.replyMs());
                    row.put("e2eMs", g.replyMs() - s.submitMs());
                    row.put("ok", r.ok());
                    row.put("beeId", r.beeId());
                    row.put("attempts", r.attempts());
                    row.put("serviceMs", svc);
                    row.put("error", r.error());
                    row.put("lost", false);
                    row.put("submitError", null);
                    row.put("hostE2eMs", r.e2eMs());
                }
                w.write(MAPPER.writeValueAsString(row));
                w.write('\n');
            }
            Map<String, Object> summary = new LinkedHashMap<>();
            summary.put("summary", true);
            summary.put("gatewayId", gatewayId);
            summary.put("tier", tier);
            summary.put("rateTarget", rate);
            summary.put("durationS", durationS);
            summary.put("submitLoopS", loopS);
            summary.put("achievedRateTps", sent.size() / loopS);
            summary.put("startMs", startMs);
            summary.put("readinessBarrierMs", barrierMs);
            summary.put("endMs", endMs);
            summary.put("submitted", sent.size());
            summary.put("submitFailures", submitFailures.get());
            summary.put("replied", replied);
            summary.put("ok", ok);
            summary.put("notOk", replied - ok);
            summary.put("lost", lost);
            summary.put("duplicates", duplicates.get());
            summary.put("serviceMsMissing", serviceMissing);
            w.write(MAPPER.writeValueAsString(summary));
            w.write('\n');
        }
        bus.close();
        System.out.println("loadgen done: submitted=" + sent.size() + " replied=" + replied + " ok=" + ok
                + " lost=" + lost + " duplicates=" + duplicates.get() + " achievedTps="
                + String.format("%.1f", sent.size() / loopS));
        return 0;
    }

    private static boolean allServiceKnown(Map<String, Got> replies, Map<String, Long> serviceMs) {
        for (Got g : replies.values()) {
            Wire.TaskReply r = g.reply();
            if (r.ok() && !serviceMs.containsKey(r.taskId() + "|" + r.beeId() + "|" + r.attempts())) {
                return false;
            }
        }
        return true;
    }
}
