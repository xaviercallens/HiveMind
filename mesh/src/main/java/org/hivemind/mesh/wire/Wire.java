package org.hivemind.mesh.wire;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.List;
import java.util.Map;

/**
 * Message payloads of the hivemind-mesh/1 wire protocol. Every payload carries {@code v} (always 1).
 * See WIRE_PROTOCOL.md for the topics and the event flow.
 */
public final class Wire {

    public static final int VERSION = 1;

    private Wire() {
    }

    /** Implemented by every payload so the codec can check the protocol version. */
    public interface Versioned {
        int v();
    }

    /** One chat message (OpenAI-style). */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Message(String role, String content) {
    }

    /** hivemind.bee.heartbeat (fan-out). {@code heartbeatMs} is an extension: the bee's interval. */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Heartbeat(int v, String beeId, List<String> tiers, int inFlight, int maxConcurrent,
                            int queueDepth, String region, long startedAtMs, long sentAtMs,
                            long heartbeatMs) implements Versioned {
    }

    /** hivemind.bee.lifecycle (fan-out). state is JOINING or LEAVING. */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Lifecycle(int v, String beeId, String state, long atMs) implements Versioned {
    }

    /** hivemind.bee.inbox.sNN (fan-out, dispatched by targetBeeId). */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record TaskAssignment(int v, String taskId, int attempt, String targetBeeId, String tier,
                                 List<Message> messages, String replyTopic, long deadlineMs,
                                 String hostId) implements Versioned {
    }

    /** hivemind.task.submit (queue "hosts": one host per task). */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record TaskSubmit(int v, String taskId, String tier, List<Message> messages, String replyTopic,
                             long deadlineMs, String userId) implements Versioned {
    }

    /** hivemind.task.done (fan-out to all hosts; only the host whose hostId matches finalises). */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record TaskDone(int v, String taskId, int attempt, String beeId, boolean ok, String output,
                           String error, long serviceMs, String hostId) implements Versioned {
    }

    /** hivemind.user.reply.&lt;gatewayId&gt; (fan-out). */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record TaskReply(int v, String taskId, boolean ok, String output, String error, String beeId,
                            int attempts, long e2eMs) implements Versioned {
    }

    /** hivemind.bee.mesh.help (fan-out among bees). */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record HelpRequest(int v, String fromBeeId, String taskId, String tier, String reason)
            implements Versioned {
    }

    /** hivemind.bee.mesh.offer (fan-out among bees). */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record HelpOffer(int v, String fromBeeId, String toBeeId, String taskId) implements Versioned {
    }

    /** hivemind.host.announce (fan-out, every 2 s). */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record HostAnnounce(int v, String hostId, int liveBees, Map<String, Integer> liveBeesByTier,
                               int inFlight, long assigned, long completed, long failed, long reassigned,
                               long atMs) implements Versioned {
    }

    /**
     * hivemind.control.load (fan-out). Published by a load generator when its load starts. Bee JVMs
     * started with --kill-after-s T schedule their kill at startMs + T seconds, so kill time is measured
     * from load start and not from JVM start.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record LoadStart(int v, String loadgenId, long startMs) implements Versioned {
    }
}
