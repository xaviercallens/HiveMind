package org.hivemind.mesh.wire;

import java.util.regex.Pattern;

/**
 * Topic names of the mesh, all under a configurable root (default "hivemind").
 * Tests and scale runs use a unique root so they cannot collide with each other.
 */
public record Topics(String root) {

    /** Number of inbox shards (hivemind.bee.inbox.s00 .. s31). */
    public static final int SHARDS = 32;

    private static final Pattern ROOT_OK = Pattern.compile("[A-Za-z0-9_-]{1,64}");

    public Topics {
        if (root == null || !ROOT_OK.matcher(root).matches()) {
            throw new IllegalArgumentException("topic root must match [A-Za-z0-9_-]{1,64}: " + root);
        }
    }

    public static Topics defaults() {
        return new Topics("hivemind");
    }

    /** Shard of a bee: Math.floorMod(beeId.hashCode(), 32). Stable across JVMs and versions of this code. */
    public static int shardOf(String beeId) {
        return Math.floorMod(beeId.hashCode(), SHARDS);
    }

    public String heartbeat() {
        return root + ".bee.heartbeat";
    }

    public String lifecycle() {
        return root + ".bee.lifecycle";
    }

    public String inboxShard(int shard) {
        if (shard < 0 || shard >= SHARDS) {
            throw new IllegalArgumentException("shard out of range: " + shard);
        }
        return String.format("%s.bee.inbox.s%02d", root, shard);
    }

    public String inboxFor(String beeId) {
        return inboxShard(shardOf(beeId));
    }

    public String taskSubmit() {
        return root + ".task.submit";
    }

    public String taskDone() {
        return root + ".task.done";
    }

    public String userReply(String gatewayId) {
        return root + ".user.reply." + gatewayId;
    }

    public String help() {
        return root + ".bee.mesh.help";
    }

    public String offer() {
        return root + ".bee.mesh.offer";
    }

    public String hostAnnounce() {
        return root + ".host.announce";
    }

    /** Load-start marker published by a load generator (scale runs only). */
    public String control() {
        return root + ".control.load";
    }

    /** Kafka consumer-group base shared by every JVM of this mesh root. */
    public String consumerGroup() {
        return root + ".mesh";
    }

    /** Name of the competing-consumer queue that every host joins for task.submit. */
    public static final String HOSTS_QUEUE = "hosts";
}
