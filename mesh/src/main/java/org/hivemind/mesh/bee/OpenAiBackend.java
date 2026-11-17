package org.hivemind.mesh.bee;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.hivemind.mesh.wire.Wire;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;

/**
 * Calls an OpenAI-compatible {@code /v1/chat/completions} endpoint. There is no default URL:
 * the caller must supply one explicitly (CLI: --backend-url).
 */
public final class OpenAiBackend implements Backend {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final URI endpoint;
    private final String model;
    private final HttpClient client;

    public OpenAiBackend(String baseUrl, String model) {
        if (baseUrl == null || baseUrl.isBlank()) {
            throw new IllegalArgumentException("OpenAiBackend requires an explicit base URL");
        }
        String base = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.endpoint = URI.create(base + "/v1/chat/completions");
        this.model = model;
        this.client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
    }

    @Override
    public String complete(String tier, List<Wire.Message> messages) throws Exception {
        ObjectNode body = MAPPER.createObjectNode();
        body.put("model", model != null ? model : tier);
        ArrayNode msgs = body.putArray("messages");
        for (Wire.Message m : messages) {
            ObjectNode n = msgs.addObject();
            n.put("role", m.role());
            n.put("content", m.content());
        }
        HttpRequest request = HttpRequest.newBuilder(endpoint)
                .timeout(Duration.ofMinutes(5))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(MAPPER.writeValueAsString(body)))
                .build();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() / 100 != 2) {
            throw new IllegalStateException("backend HTTP " + response.statusCode());
        }
        JsonNode content = MAPPER.readTree(response.body()).path("choices").path(0).path("message").path("content");
        if (!content.isTextual()) {
            throw new IllegalStateException("backend response has no choices[0].message.content");
        }
        return content.asText();
    }
}
