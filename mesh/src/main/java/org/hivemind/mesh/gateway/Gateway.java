package org.hivemind.mesh.gateway;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import io.amcp.core.Event;
import org.hivemind.mesh.wire.Codec;
import org.hivemind.mesh.wire.MeshBus;
import org.hivemind.mesh.wire.Topics;
import org.hivemind.mesh.wire.Wire;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * OpenAI-compatible HTTP front door. Bound to 127.0.0.1 only; requests whose Host header is not
 * 127.0.0.1 or localhost get 403. Bodies above 64 KiB get 413.
 *
 * <ul>
 *   <li>POST /v1/chat/completions: "model" is the tier. Publishes a TaskSubmit and waits for the
 *       TaskReply (504 on timeout, 503 when the mesh answers ok=false).</li>
 *   <li>GET /mesh/health: the latest host.announce of every host seen.</li>
 * </ul>
 */
public final class Gateway implements AutoCloseable {

    public static final int MAX_BODY_BYTES = 64 * 1024;

    private static final Logger log = LoggerFactory.getLogger(Gateway.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final MeshBus bus;
    private final Topics topics;
    private final String gatewayId;
    private final int port;
    private final long timeoutMs;
    private final Map<String, CompletableFuture<Wire.TaskReply>> pending = new ConcurrentHashMap<>();
    private final Map<String, Wire.HostAnnounce> hosts = new ConcurrentHashMap<>();
    private HttpServer server;

    public Gateway(MeshBus bus, String gatewayId, int port, long timeoutMs) {
        this.bus = bus;
        this.topics = bus.topics();
        this.gatewayId = gatewayId;
        this.port = port;
        this.timeoutMs = timeoutMs;
    }

    public String gatewayId() {
        return gatewayId;
    }

    /** The port actually bound (useful when constructed with port 0). */
    public int boundPort() {
        return server.getAddress().getPort();
    }

    public void start() throws Exception {
        bus.subscribe(topics.userReply(gatewayId), this::onReply);
        bus.subscribe(topics.hostAnnounce(), this::onAnnounce);
        bus.start();
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        server.createContext("/", this::handle);
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        server.start();
        log.info("gateway {} listening on 127.0.0.1:{}", gatewayId, boundPort());
    }

    private void onReply(Event event) {
        Wire.TaskReply reply = Codec.decode(event, Wire.TaskReply.class);
        CompletableFuture<Wire.TaskReply> f = pending.remove(reply.taskId());
        if (f != null) {
            f.complete(reply);
        }
    }

    private void onAnnounce(Event event) {
        Wire.HostAnnounce a = Codec.decode(event, Wire.HostAnnounce.class);
        hosts.put(a.hostId(), a);
    }

    private void handle(HttpExchange ex) throws IOException {
        try {
            if (!allowedHost(ex.getRequestHeaders().getFirst("Host"))) {
                json(ex, 403, error("forbidden: gateway accepts 127.0.0.1 and localhost only", "forbidden"));
                return;
            }
            String path = ex.getRequestURI().getPath();
            String method = ex.getRequestMethod();
            if ("/v1/chat/completions".equals(path)) {
                if (!"POST".equals(method)) {
                    json(ex, 405, error("use POST", "method_not_allowed"));
                    return;
                }
                chat(ex);
            } else if ("/mesh/health".equals(path)) {
                if (!"GET".equals(method)) {
                    json(ex, 405, error("use GET", "method_not_allowed"));
                    return;
                }
                health(ex);
            } else {
                json(ex, 404, error("not found", "not_found"));
            }
        } catch (RuntimeException e) {
            log.warn("request failed: {}", e.toString());
            json(ex, 500, error("internal error", "internal"));
        } finally {
            ex.close();
        }
    }

    static boolean allowedHost(String hostHeader) {
        if (hostHeader == null) {
            return false;
        }
        int colon = hostHeader.lastIndexOf(':');
        String name = colon >= 0 ? hostHeader.substring(0, colon) : hostHeader;
        return "127.0.0.1".equals(name) || "localhost".equals(name);
    }

    private void chat(HttpExchange ex) throws IOException {
        byte[] body = readLimited(ex.getRequestBody(), ex.getRequestHeaders().getFirst("Content-Length"));
        if (body == null) {
            json(ex, 413, error("request body above 64 KiB", "payload_too_large"));
            return;
        }
        JsonNode root;
        try {
            root = MAPPER.readTree(body);
        } catch (IOException e) {
            json(ex, 400, error("body is not JSON", "invalid_request"));
            return;
        }
        String tier = root == null ? null : root.path("model").asText(null);
        JsonNode messagesNode = root == null ? null : root.path("messages");
        if (tier == null || tier.isBlank() || messagesNode == null || !messagesNode.isArray()
                || messagesNode.isEmpty()) {
            json(ex, 400, error("need \"model\" (the tier) and a non-empty \"messages\" array", "invalid_request"));
            return;
        }
        List<Wire.Message> messages = new ArrayList<>();
        for (JsonNode m : messagesNode) {
            messages.add(new Wire.Message(m.path("role").asText("user"), m.path("content").asText("")));
        }

        String taskId = UUID.randomUUID().toString();
        CompletableFuture<Wire.TaskReply> future = new CompletableFuture<>();
        pending.put(taskId, future);
        long deadline = System.currentTimeMillis() + timeoutMs;
        bus.publish(topics.taskSubmit(), taskId, new Wire.TaskSubmit(Wire.VERSION, taskId, tier, messages,
                topics.userReply(gatewayId), deadline, null));
        Wire.TaskReply reply;
        try {
            reply = future.get(timeoutMs, TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            pending.remove(taskId);
            json(ex, 504, error("no reply within " + timeoutMs + " ms", "gateway_timeout"));
            return;
        } catch (Exception e) {
            pending.remove(taskId);
            json(ex, 500, error("wait failed: " + e.getMessage(), "internal"));
            return;
        }
        ObjectNode hivemind = MAPPER.createObjectNode();
        hivemind.put("beeId", reply.beeId());
        hivemind.put("attempts", reply.attempts());
        hivemind.put("e2eMs", reply.e2eMs());
        if (!reply.ok()) {
            ObjectNode out = error(reply.error(), "mesh_unavailable");
            out.set("hivemind", hivemind);
            json(ex, 503, out);
            return;
        }
        ObjectNode out = MAPPER.createObjectNode();
        out.put("id", "chatcmpl-" + taskId);
        out.put("object", "chat.completion");
        out.put("created", System.currentTimeMillis() / 1000);
        out.put("model", tier);
        ObjectNode choice = out.putArray("choices").addObject();
        choice.put("index", 0);
        ObjectNode message = choice.putObject("message");
        message.put("role", "assistant");
        message.put("content", reply.output());
        choice.put("finish_reason", "stop");
        out.set("hivemind", hivemind);
        json(ex, 200, out);
    }

    private void health(HttpExchange ex) throws IOException {
        ObjectNode out = MAPPER.createObjectNode();
        out.put("v", Wire.VERSION);
        out.put("gatewayId", gatewayId);
        ArrayNode list = out.putArray("hosts");
        for (Wire.HostAnnounce a : hosts.values()) {
            list.add(MAPPER.valueToTree(a));
        }
        json(ex, 200, out);
    }

    /** Reads at most MAX_BODY_BYTES; returns null when the body is larger. */
    static byte[] readLimited(InputStream in, String contentLength) throws IOException {
        if (contentLength != null) {
            try {
                if (Long.parseLong(contentLength.trim()) > MAX_BODY_BYTES) {
                    return null;
                }
            } catch (NumberFormatException ignored) {
                // fall through to the streaming limit
            }
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = in.read(buf)) != -1) {
            if (out.size() + n > MAX_BODY_BYTES) {
                return null;
            }
            out.write(buf, 0, n);
        }
        return out.toByteArray();
    }

    private static ObjectNode error(String message, String type) {
        ObjectNode root = MAPPER.createObjectNode();
        ObjectNode err = root.putObject("error");
        err.put("message", message);
        err.put("type", type);
        return root;
    }

    private static void json(HttpExchange ex, int status, JsonNode body) throws IOException {
        byte[] bytes = MAPPER.writeValueAsBytes(body);
        ex.getResponseHeaders().set("Content-Type", "application/json");
        ex.sendResponseHeaders(status, bytes.length);
        ex.getResponseBody().write(bytes);
    }

    @Override
    public void close() {
        if (server != null) {
            server.stop(0);
        }
    }
}
