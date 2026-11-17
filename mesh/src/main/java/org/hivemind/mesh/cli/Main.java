package org.hivemind.mesh.cli;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.hivemind.mesh.bee.Backend;
import org.hivemind.mesh.bee.BeeRuntime;
import org.hivemind.mesh.bee.EventLog;
import org.hivemind.mesh.bee.OpenAiBackend;
import org.hivemind.mesh.gateway.Gateway;
import org.hivemind.mesh.host.Host;
import org.hivemind.mesh.sim.LoadGen;
import org.hivemind.mesh.sim.SimBackend;
import org.hivemind.mesh.wire.Codec;
import org.hivemind.mesh.wire.MeshBus;
import org.hivemind.mesh.wire.Topics;
import org.hivemind.mesh.wire.Wire;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * hivemind-bee-mesh command line. Every subcommand accepts --bootstrap (default 127.0.0.1:9092)
 * and --root (topic root, default "hivemind").
 *
 * <pre>
 * host    [--host-id ID] [--attempt-timeout-ms 10000] [--heartbeat-ms 5000]
 * bees    --count N [--tiers coder,general] [--max-concurrent 4] [--sim-latency-ms 50]
 *         [--sim-jitter-ms 10] [--fail-rate 0] [--backend-url URL --backend-model M]
 *         [--region local] [--help-wait-ms 500] [--queue-bound 16]
 * gateway [--port 8097] [--timeout-s 30]
 * ask     --tier TIER [--timeout-s 30] "prompt"
 *         [--kill-after-s T --kill-fraction F --kill-seed S --event-log path]   (scale runs)
 * status  [--wait-bees N --hosts K --timeout S]
 * loadgen --rate R --duration S [--tier coder] --out file.jsonl [--gateway-id lg1]
 * record  --out file.jsonl [--seconds S]   (every host.announce, one JSON line each)
 * </pre>
 */
public final class Main {

    private static final String DEFAULT_BOOTSTRAP = "127.0.0.1:9092";
    private static final String DEFAULT_ROOT = "hivemind";
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private Main() {
    }

    public static void main(String[] args) {
        if (args.length == 0 || "help".equals(args[0]) || "--help".equals(args[0])) {
            System.out.println("usage: host | bees | gateway | ask | status  (see README.md)");
            System.exit(args.length == 0 ? 2 : 0);
        }
        int code;
        try {
            code = run(args[0], new Options(Arrays.copyOfRange(args, 1, args.length)));
        } catch (IllegalArgumentException e) {
            System.err.println("error: " + e.getMessage());
            code = 2;
        } catch (Exception e) {
            System.err.println("error: " + e);
            code = 1;
        }
        System.exit(code);
    }

    static int run(String command, Options o) throws Exception {
        return switch (command) {
            case "host" -> host(o);
            case "bees" -> bees(o);
            case "gateway" -> gateway(o);
            case "ask" -> ask(o);
            case "status" -> status(o);
            case "loadgen" -> loadgen(o);
            case "record" -> record(o);
            default -> throw new IllegalArgumentException("unknown command: " + command);
        };
    }

    private static int host(Options o) throws Exception {
        String hostId = o.get("host-id", "host-" + randomHex(6));
        MeshBus bus = new MeshBus(o.get("bootstrap", DEFAULT_BOOTSTRAP), topics(o), UUID.randomUUID().toString());
        Host host = new Host(bus, new Host.Config(hostId, o.getLong("attempt-timeout-ms", 10_000),
                o.getLong("heartbeat-ms", 5_000), 2_000));
        host.start();
        System.out.println("host " + hostId + " running on root " + topics(o).root());
        onShutdown(() -> {
            host.close();
            closeQuietly(bus);
        });
        return block();
    }

    private static int bees(Options o) throws Exception {
        int count = o.getInt("count", 1);
        List<String> tiers = Arrays.asList(o.get("tiers", "coder,general").split(","));
        BeeRuntime.Config cfg = new BeeRuntime.Config(tiers, o.getInt("max-concurrent", 4),
                o.get("region", "local"), o.getLong("heartbeat-ms", 5_000), o.getLong("help-wait-ms", 500),
                o.getInt("queue-bound", 16));
        java.util.function.IntFunction<Backend> backends = backendFactory(o);
        String prefix = "bee-" + randomHex(6);
        MeshBus bus = new MeshBus(o.get("bootstrap", DEFAULT_BOOTSTRAP), topics(o), UUID.randomUUID().toString());
        BeeRuntime runtime = new BeeRuntime(bus, cfg, count, prefix, backends);
        runtime.configureKill(o.getLong("kill-after-s", 0), Double.parseDouble(o.get("kill-fraction", "0")),
                o.getLong("kill-seed", 20261027L));
        if (o.has("event-log")) {
            runtime.setEventLog(new EventLog(java.nio.file.Path.of(o.require("event-log"))));
        }
        runtime.start();
        System.out.println(count + " bees joined (" + prefix + "-*) tiers=" + tiers
                + " max-concurrent=" + cfg.maxConcurrent());
        onShutdown(() -> {
            runtime.close();
            closeQuietly(bus);
        });
        return block();
    }

    private static java.util.function.IntFunction<Backend> backendFactory(Options o) {
        if (o.has("backend-url")) {
            OpenAiBackend shared = new OpenAiBackend(o.get("backend-url", null), o.get("backend-model", null));
            return i -> shared;
        }
        long latency = o.getLong("sim-latency-ms", 50);
        long jitter = o.getLong("sim-jitter-ms", 10);
        double failRate = Double.parseDouble(o.get("fail-rate", "0"));
        long seedBase = System.nanoTime();
        return i -> new SimBackend(latency, jitter, failRate, seedBase + i);
    }

    private static int gateway(Options o) throws Exception {
        MeshBus bus = new MeshBus(o.get("bootstrap", DEFAULT_BOOTSTRAP), topics(o), UUID.randomUUID().toString());
        Gateway gateway = new Gateway(bus, "gw-" + randomHex(6), o.getInt("port", 8097),
                o.getLong("timeout-s", 30) * 1000);
        gateway.start();
        System.out.println("gateway " + gateway.gatewayId() + " on http://127.0.0.1:" + gateway.boundPort());
        onShutdown(() -> {
            gateway.close();
            closeQuietly(bus);
        });
        return block();
    }

    private static int ask(Options o) throws Exception {
        String tier = o.require("tier");
        String prompt = String.join(" ", o.positional());
        if (prompt.isBlank()) {
            throw new IllegalArgumentException("ask needs a prompt");
        }
        long timeoutMs = o.getLong("timeout-s", 30) * 1000;
        MeshBus bus = new MeshBus(o.get("bootstrap", DEFAULT_BOOTSTRAP), topics(o), UUID.randomUUID().toString());
        String replyId = "ask-" + randomHex(8);
        CompletableFuture<Wire.TaskReply> reply = new CompletableFuture<>();
        bus.subscribe(bus.topics().userReply(replyId), e -> {
            Wire.TaskReply r = Codec.decode(e, Wire.TaskReply.class);
            reply.complete(r);
        });
        bus.start();
        try {
            String taskId = UUID.randomUUID().toString();
            bus.publish(bus.topics().taskSubmit(), taskId, new Wire.TaskSubmit(Wire.VERSION, taskId, tier,
                    List.of(new Wire.Message("user", prompt)), bus.topics().userReply(replyId),
                    System.currentTimeMillis() + timeoutMs, null)).get(10, TimeUnit.SECONDS);
            Wire.TaskReply r = reply.get(timeoutMs + 5_000, TimeUnit.MILLISECONDS);
            if (r.ok()) {
                System.out.println(r.output());
                System.err.println("bee=" + r.beeId() + " attempts=" + r.attempts() + " e2eMs=" + r.e2eMs());
                return 0;
            }
            System.err.println("error: " + r.error());
            return 1;
        } finally {
            closeQuietly(bus);
        }
    }

    private static int status(Options o) throws Exception {
        boolean waiting = o.has("wait-bees");
        int want = o.getInt("wait-bees", 0);
        int hostsWanted = o.getInt("hosts", 0);
        long timeoutMs = o.getLong("timeout", waiting ? 30 : 5) * 1000;
        MeshBus bus = new MeshBus(o.get("bootstrap", DEFAULT_BOOTSTRAP), topics(o), UUID.randomUUID().toString());
        Map<String, Wire.HostAnnounce> latest = new ConcurrentHashMap<>();
        bus.subscribe(bus.topics().hostAnnounce(), e -> {
            Wire.HostAnnounce a = Codec.decode(e, Wire.HostAnnounce.class);
            latest.merge(a.hostId(), a, (old, neu) -> neu.atMs() >= old.atMs() ? neu : old);
        });
        bus.start();
        try {
            long start = System.currentTimeMillis();
            while (true) {
                Thread.sleep(200);
                long elapsed = System.currentTimeMillis() - start;
                int maxLive = latest.values().stream().mapToInt(Wire.HostAnnounce::liveBees).max().orElse(-1);
                // --hosts K: at least K hosts must be seen and every one of them must count >= N live bees.
                int minLive = latest.values().stream().mapToInt(Wire.HostAnnounce::liveBees).min().orElse(-1);
                boolean hostsOk = latest.size() >= hostsWanted && minLive >= want;
                if (waiting && (hostsWanted > 0 ? hostsOk : maxLive >= want)) {
                    print(latest);
                    return 0;
                }
                if (!waiting && !latest.isEmpty() && elapsed >= 1500) {
                    print(latest);
                    return 0;
                }
                if (elapsed >= timeoutMs) {
                    print(latest);
                    System.err.println(waiting ? "timeout: liveBees < " + want : "no host.announce seen");
                    return 1;
                }
            }
        } finally {
            closeQuietly(bus);
        }
    }

    private static int loadgen(Options o) throws Exception {
        double rate = Double.parseDouble(o.require("rate"));
        long duration = o.getLong("duration", 0);
        return LoadGen.run(o.get("bootstrap", DEFAULT_BOOTSTRAP), topics(o), rate, duration,
                o.get("tier", "coder"), java.nio.file.Path.of(o.require("out")), o.get("gateway-id", "lg1"));
    }

    /**
     * Writes every hivemind.host.announce as one JSON line with the local receive time, until SIGTERM
     * (or --seconds S). Lines are flushed one by one, so a killed recorder keeps everything it saw.
     */
    private static int record(Options o) throws Exception {
        java.io.BufferedWriter w = java.nio.file.Files.newBufferedWriter(
                java.nio.file.Path.of(o.require("out")), java.nio.charset.StandardCharsets.UTF_8);
        MeshBus bus = new MeshBus(o.get("bootstrap", DEFAULT_BOOTSTRAP), topics(o), UUID.randomUUID().toString());
        bus.subscribe(bus.topics().hostAnnounce(), e -> {
            Wire.HostAnnounce a = Codec.decode(e, Wire.HostAnnounce.class);
            Map<String, Object> row = new java.util.LinkedHashMap<>();
            row.put("recvMs", System.currentTimeMillis());
            row.put("hostId", a.hostId());
            row.put("liveBees", a.liveBees());
            row.put("inFlight", a.inFlight());
            row.put("atMs", a.atMs());
            row.put("completed", a.completed());
            row.put("failed", a.failed());
            row.put("reassigned", a.reassigned());
            try {
                String line = MAPPER.writeValueAsString(row);
                synchronized (w) {
                    w.write(line);
                    w.write('\n');
                    w.flush();
                }
            } catch (java.io.IOException ex) {
                System.err.println("record: " + ex);
            }
        });
        bus.start();
        onShutdown(() -> {
            synchronized (w) {
                try {
                    w.close();
                } catch (java.io.IOException ex) {
                    System.err.println("record close: " + ex);
                }
            }
            closeQuietly(bus);
        });
        long seconds = o.getLong("seconds", 0);
        if (seconds > 0) {
            Thread.sleep(seconds * 1000);
            System.exit(0);
        }
        return block();
    }

    private static void print(Map<String, Wire.HostAnnounce> latest) throws Exception {
        for (Wire.HostAnnounce a : latest.values()) {
            System.out.println(MAPPER.writeValueAsString(a));
        }
    }

    private static Topics topics(Options o) {
        return new Topics(o.get("root", DEFAULT_ROOT));
    }

    private static String randomHex(int bytes) {
        byte[] b = new byte[bytes];
        new java.security.SecureRandom().nextBytes(b);
        return HexFormat.of().formatHex(b);
    }

    private static void onShutdown(Runnable cleanup) {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                cleanup.run();
            } catch (RuntimeException e) {
                System.err.println("shutdown: " + e);
            }
        }, "mesh-shutdown"));
    }

    /** Blocks until the process is signalled; the shutdown hook does the cleanup. */
    private static int block() throws InterruptedException {
        new CountDownLatch(1).await();
        return 0;
    }

    private static void closeQuietly(MeshBus bus) {
        try {
            bus.close();
        } catch (Exception e) {
            System.err.println("close: " + e);
        }
    }

    /** Minimal "--key value" parser; a bare "--flag" is stored as "true". */
    static final class Options {
        private final Map<String, String> values = new java.util.HashMap<>();
        private final List<String> positional = new ArrayList<>();

        Options(String[] args) {
            for (int i = 0; i < args.length; i++) {
                String a = args[i];
                if (a.startsWith("--") && a.length() > 2) {
                    String key = a.substring(2);
                    if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                        values.put(key, args[++i]);
                    } else {
                        values.put(key, "true");
                    }
                } else {
                    positional.add(a);
                }
            }
        }

        boolean has(String key) {
            return values.containsKey(key);
        }

        String get(String key, String dflt) {
            return values.getOrDefault(key, dflt);
        }

        String require(String key) {
            String v = values.get(key);
            if (v == null) {
                throw new IllegalArgumentException("missing --" + key);
            }
            return v;
        }

        int getInt(String key, int dflt) {
            return has(key) ? Integer.parseInt(values.get(key)) : dflt;
        }

        long getLong(String key, long dflt) {
            return has(key) ? Long.parseLong(values.get(key)) : dflt;
        }

        List<String> positional() {
            return positional;
        }
    }
}
