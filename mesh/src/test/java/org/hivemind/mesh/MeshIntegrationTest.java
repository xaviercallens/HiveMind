package org.hivemind.mesh;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.hivemind.mesh.bee.Backend;
import org.hivemind.mesh.bee.BeeRuntime;
import org.hivemind.mesh.gateway.Gateway;
import org.hivemind.mesh.host.Host;
import org.hivemind.mesh.sim.SimBackend;
import org.hivemind.mesh.wire.Codec;
import org.hivemind.mesh.wire.MeshBus;
import org.hivemind.mesh.wire.Topics;
import org.hivemind.mesh.wire.Wire;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end tests on a real Kafka broker. They run only when HIVEMIND_MESH_KAFKA is set to the
 * bootstrap address (for example 127.0.0.1:9092); otherwise they are skipped. Each test uses its own
 * random topic root, so tests and reruns never see each other's events.
 */
@EnabledIfEnvironmentVariable(named = "HIVEMIND_MESH_KAFKA", matches = ".+")
class MeshIntegrationTest {

    private static final String BOOTSTRAP = System.getenv("HIVEMIND_MESH_KAFKA");
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient HTTP = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();

    private final List<AutoCloseable> closeables = new ArrayList<>();

    @AfterEach
    void tearDown() {
        for (int i = closeables.size() - 1; i >= 0; i--) {
            try {
                closeables.get(i).close();
            } catch (Exception ignored) {
                // best effort: a hung backend is interrupted by BeeRuntime.close()
            }
        }
        closeables.clear();
    }

    @Test
    void oneHostFiveBeesAndGatewayAnswerTwentyRequests() throws Exception {
        Topics t = newRoot();
        MeshBus hostBus = bus(t);
        MeshBus beeBus = bus(t);
        MeshBus gwBus = bus(t);
        Host host = host(hostBus, "host-1", 10_000);
        BeeRuntime bees = bees(beeBus, List.of("coder"), 4, 500, 16, 5, "t1", i -> new SimBackend(50, 10, 0, i));
        Gateway gw = gateway(gwBus, 30_000);
        waitUntil(() -> host.liveBees() >= 5, 30_000);

        int port = gw.boundPort();
        for (int i = 0; i < 20; i++) {
            HttpResponse<String> r = chat(port, "coder", "request " + i, 60_000);
            assertEquals(200, r.statusCode(), r.body());
            JsonNode body = MAPPER.readTree(r.body());
            assertTrue(body.path("choices").path(0).path("message").path("content").asText().contains("request " + i));
            assertTrue(body.path("hivemind").path("beeId").asText().startsWith("t1-"), r.body());
        }
        assertEquals(5, bees.beeCount());
    }

    @Test
    void busyBeeHandsSecondTaskOffToAPeer() throws Exception {
        Topics t = newRoot();
        MeshBus hostBus = bus(t);
        MeshBus aBus = bus(t);
        MeshBus bBus = bus(t);
        MeshBus gwBus = bus(t);
        Host host = host(hostBus, "host-h", 30_000);
        // Bee A: one slot, slow model, and a long help window so a peer has time to join.
        BeeRuntime a = bees(aBus, List.of("coder"), 1, 20_000, 4, 1, "ta", i -> new SimBackend(400, 0, 0, 1));
        Gateway gw = gateway(gwBus, 60_000);
        waitUntil(() -> host.liveBees() >= 1, 30_000);
        String beeA = a.beeIds().get(0);
        // Warm-up: the host's task.submit consumer joins its group only after a few seconds. Routing the first
        // real request after this proves the host is consuming, so the scenario below is deterministic.
        warmUp(gw.boundPort(), "coder");

        CompletableFuture<HttpResponse<String>> first =
                CompletableFuture.supplyAsync(() -> chatUnchecked(gw.boundPort(), "coder", "first", 90_000));
        Thread.sleep(300);
        CompletableFuture<HttpResponse<String>> second =
                CompletableFuture.supplyAsync(() -> chatUnchecked(gw.boundPort(), "coder", "second", 90_000));

        // Bee B joins only now. It hears the HelpRequest from A and offers; A hands the task to B.
        Thread.sleep(500);
        BeeRuntime b = bees(bBus, List.of("coder"), 4, 20_000, 4, 1, "tb", i -> new SimBackend(50, 0, 0, 2));
        String beeB = b.beeIds().get(0);

        HttpResponse<String> r1 = first.get(90, java.util.concurrent.TimeUnit.SECONDS);
        HttpResponse<String> r2 = second.get(90, java.util.concurrent.TimeUnit.SECONDS);
        assertEquals(200, r1.statusCode(), r1.body());
        assertEquals(200, r2.statusCode(), r2.body());
        assertEquals(beeA, MAPPER.readTree(r1.body()).path("hivemind").path("beeId").asText());
        JsonNode hm2 = MAPPER.readTree(r2.body()).path("hivemind");
        assertEquals(beeB, hm2.path("beeId").asText(), r2.body());
        assertEquals(2, hm2.path("attempts").asInt(), r2.body());
        // Attempt 2 must come from the bee-to-bee handoff, not from a host reassignment (the attempt timeout is
        // 30 s and this test finishes well before it).
        assertEquals(0, host.reassignedCount(), "handoff, not host reassignment, must produce attempt 2");
    }

    @Test
    void hungBackendIsFailedOverToAnotherBee() throws Exception {
        Topics t = newRoot();
        MeshBus hostBus = bus(t);
        MeshBus hungBus = bus(t);
        MeshBus goodBus = bus(t);
        MeshBus gwBus = bus(t);
        Host host = host(hostBus, "host-f", 5_000);
        // The backend never returns: the task must be reassigned after attemptTimeoutMs.
        Backend hung = (tier, messages) -> {
            Thread.sleep(Long.MAX_VALUE);
            return "unreachable";
        };
        BeeRuntime hungRuntime = bees(hungBus, List.of("coder"), 4, 500, 4, 1, "th", i -> hung);
        // A separate bee on another tier only exists to prove the host is consuming task.submit.
        MeshBus warmBus = bus(t);
        bees(warmBus, List.of("warm"), 4, 500, 4, 1, "tw", i -> new SimBackend(10, 0, 0, 9));
        Gateway gw = gateway(gwBus, 60_000);
        waitUntil(() -> host.liveBees() >= 2, 30_000);
        warmUp(gw.boundPort(), "warm");

        CompletableFuture<HttpResponse<String>> pending =
                CompletableFuture.supplyAsync(() -> chatUnchecked(gw.boundPort(), "coder", "failover", 90_000));
        Thread.sleep(500);
        // The good bee joins while the hung attempt is still outstanding.
        BeeRuntime good = bees(goodBus, List.of("coder"), 4, 500, 4, 1, "tg", i -> new SimBackend(50, 0, 0, 3));
        String goodId = good.beeIds().get(0);

        HttpResponse<String> r = pending.get(90, java.util.concurrent.TimeUnit.SECONDS);
        assertEquals(200, r.statusCode(), r.body());
        JsonNode hm = MAPPER.readTree(r.body()).path("hivemind");
        assertEquals(goodId, hm.path("beeId").asText(), r.body());
        assertEquals(2, hm.path("attempts").asInt(), r.body());
        assertEquals(1, host.reassignedCount());
        assertTrue(hungRuntime.beeCount() == 1);
    }

    @Test
    void twoHostsSharingTaskSubmitAnswerEveryTaskExactlyOnce() throws Exception {
        Topics t = newRoot();
        MeshBus hostA = bus(t);
        MeshBus hostB = bus(t);
        MeshBus beeBus = bus(t);
        MeshBus client = bus(t);
        Host ha = host(hostA, "host-A", 10_000);
        Host hb = host(hostB, "host-B", 10_000);
        BeeRuntime bees = bees(beeBus, List.of("general"), 8, 500, 16, 3, "t4", i -> new SimBackend(20, 5, 0, i));

        Map<String, AtomicInteger> repliesByTask = new ConcurrentHashMap<>();
        String replyTopic = t.userReply("client-" + UUID.randomUUID().toString().substring(0, 8));
        client.subscribe(replyTopic, e -> {
            Wire.TaskReply r = Codec.decode(e, Wire.TaskReply.class);
            repliesByTask.computeIfAbsent(r.taskId(), k -> new AtomicInteger()).incrementAndGet();
        });
        client.start();

        waitUntil(() -> ha.liveBees() >= 3 && hb.liveBees() >= 3, 30_000);

        int tasks = 30;
        List<String> ids = new ArrayList<>();
        for (int i = 0; i < tasks; i++) {
            String taskId = UUID.randomUUID().toString();
            ids.add(taskId);
            client.publish(t.taskSubmit(), taskId, new Wire.TaskSubmit(Wire.VERSION, taskId, "general",
                    List.of(new Wire.Message("user", "task " + i)), replyTopic,
                    System.currentTimeMillis() + 60_000, null)).get(10, java.util.concurrent.TimeUnit.SECONDS);
        }
        waitUntil(() -> repliesByTask.size() >= tasks, 90_000);
        Thread.sleep(2_000); // give any duplicate reply time to show up

        for (String id : ids) {
            assertEquals(1, repliesByTask.getOrDefault(id, new AtomicInteger(0)).get(), "task " + id);
        }
        assertEquals(tasks, repliesByTask.size());
        assertEquals(tasks, ha.completedCount() + hb.completedCount());
        assertEquals(0, ha.failedCount() + hb.failedCount());
        assertEquals(3, bees.beeCount());
    }

    // ---- helpers ----

    private static Topics newRoot() {
        String alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";
        StringBuilder sb = new StringBuilder("hmit");
        java.util.Random rnd = new java.util.Random();
        for (int i = 0; i < 10; i++) {
            sb.append(alphabet.charAt(rnd.nextInt(alphabet.length())));
        }
        return new Topics(sb.toString());
    }

    private MeshBus bus(Topics t) {
        MeshBus bus = new MeshBus(BOOTSTRAP, t, UUID.randomUUID().toString());
        closeables.add(bus::close);
        return bus;
    }

    private Host host(MeshBus bus, String hostId, long attemptTimeoutMs) throws Exception {
        Host host = new Host(bus, new Host.Config(hostId, attemptTimeoutMs, 5_000, 2_000));
        host.start();
        closeables.add(host::close);
        return host;
    }

    private BeeRuntime bees(MeshBus bus, List<String> tiers, int maxConcurrent, long helpWaitMs, int queueBound,
                            int count, String prefix, java.util.function.IntFunction<Backend> backends)
            throws Exception {
        BeeRuntime runtime = new BeeRuntime(bus, new BeeRuntime.Config(tiers, maxConcurrent, "test", 5_000,
                helpWaitMs, queueBound), count, prefix, backends);
        runtime.start();
        closeables.add(runtime::close);
        return runtime;
    }

    private Gateway gateway(MeshBus bus, long timeoutMs) throws Exception {
        Gateway gw = new Gateway(bus, "gw-" + UUID.randomUUID().toString().substring(0, 8), 0, timeoutMs);
        gw.start();
        closeables.add(gw::close);
        return gw;
    }

    private static HttpResponse<String> chatUnchecked(int port, String tier, String prompt, long timeoutMs) {
        try {
            return chat(port, tier, prompt, timeoutMs);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private static void warmUp(int port, String tier) throws Exception {
        HttpResponse<String> r = chat(port, tier, "warm-up", 60_000);
        assertEquals(200, r.statusCode(), r.body());
    }

    private static HttpResponse<String> chat(int port, String tier, String prompt, long timeoutMs) throws Exception {
        Map<String, Object> body = new HashMap<>();
        body.put("model", tier);
        body.put("messages", List.of(Map.of("role", "user", "content", prompt)));
        HttpRequest req = HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port + "/v1/chat/completions"))
                .timeout(Duration.ofMillis(timeoutMs))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(MAPPER.writeValueAsString(body)))
                .build();
        return HTTP.send(req, HttpResponse.BodyHandlers.ofString());
    }

    private static void waitUntil(BooleanSupplier condition, long timeoutMs) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (!condition.getAsBoolean()) {
            if (System.currentTimeMillis() > deadline) {
                throw new AssertionError("condition not met within " + timeoutMs + " ms");
            }
            Thread.sleep(100);
        }
    }
}
