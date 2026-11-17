package org.hivemind.mesh.sim;

import org.hivemind.mesh.bee.Backend;
import org.hivemind.mesh.wire.Wire;

import java.util.List;
import java.util.Random;

/**
 * Simulated model: sleeps latency + random jitter (on the calling virtual thread), fails with
 * probability failRate, and otherwise returns a deterministic text derived from the tier and the
 * last message.
 */
public final class SimBackend implements Backend {

    private final long latencyMs;
    private final long jitterMs;
    private final double failRate;
    private final Random random;

    public SimBackend(long latencyMs, long jitterMs, double failRate, long seed) {
        if (latencyMs < 0 || jitterMs < 0 || failRate < 0 || failRate > 1) {
            throw new IllegalArgumentException("invalid sim parameters");
        }
        this.latencyMs = latencyMs;
        this.jitterMs = jitterMs;
        this.failRate = failRate;
        this.random = new Random(seed);
    }

    @Override
    public String complete(String tier, List<Wire.Message> messages) throws Exception {
        long delay = latencyMs + (jitterMs > 0 ? random.nextLong(jitterMs + 1) : 0);
        if (delay > 0) {
            Thread.sleep(delay);
        }
        if (failRate > 0 && random.nextDouble() < failRate) {
            throw new IllegalStateException("simulated backend failure");
        }
        String last = messages.isEmpty() ? "" : messages.get(messages.size() - 1).content();
        return "sim[" + tier + "]: " + last;
    }
}
