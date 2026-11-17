package org.hivemind.mesh.bee;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Plain-text event log of one bee JVM, used by scale runs. Lines are space separated:
 * <pre>
 * ASSIGN taskId attempt beeId ms     every assignment routed to a bee of this JVM (before any drop)
 * DROP   taskId beeId attempt phase ms   a killed bee suppressed work (phase: assign, queued, help, inflight)
 * KILL   beeId ms                    a bee was killed (stops heartbeats, answers and LEAVING)
 * </pre>
 * Writes are synchronized and buffered; the runtime flushes once per second and on close.
 */
public final class EventLog implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(EventLog.class);

    private final BufferedWriter writer;

    public EventLog(Path path) throws IOException {
        this.writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8, StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
    }

    public synchronized void line(String text) {
        try {
            writer.write(text);
            writer.write('\n');
        } catch (IOException e) {
            log.warn("event log write failed: {}", e.toString());
        }
    }

    public synchronized void flush() {
        try {
            writer.flush();
        } catch (IOException e) {
            log.warn("event log flush failed: {}", e.toString());
        }
    }

    @Override
    public synchronized void close() {
        try {
            writer.close();
        } catch (IOException e) {
            log.warn("event log close failed: {}", e.toString());
        }
    }
}
