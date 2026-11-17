package org.hivemind.mesh.wire;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.amcp.core.Event;

/** JSON codec for hivemind-mesh/1 payloads. Rejects any payload whose {@code v} is not 1. */
public final class Codec {

    private static final ObjectMapper MAPPER = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    private Codec() {
    }

    public static byte[] encode(Object payload) {
        try {
            return MAPPER.writeValueAsBytes(payload);
        } catch (Exception e) {
            throw new IllegalStateException("cannot encode " + payload.getClass().getSimpleName(), e);
        }
    }

    public static <T extends Wire.Versioned> T decode(byte[] json, Class<T> type) {
        if (json == null) {
            throw new IllegalArgumentException("empty payload for " + type.getSimpleName());
        }
        try {
            T value = MAPPER.readValue(json, type);
            if (value.v() != Wire.VERSION) {
                throw new IllegalArgumentException(
                        "unsupported protocol version " + value.v() + " for " + type.getSimpleName());
            }
            return value;
        } catch (java.io.IOException e) {
            throw new IllegalArgumentException("malformed " + type.getSimpleName() + ": " + e.getMessage(), e);
        }
    }

    public static <T extends Wire.Versioned> T decode(Event event, Class<T> type) {
        return decode(event.getData(), type);
    }
}
