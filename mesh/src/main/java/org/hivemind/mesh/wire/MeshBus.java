package org.hivemind.mesh.wire;

import io.amcp.broker.kafka.KafkaEventBroker;
import io.amcp.core.AgentContext;
import io.amcp.core.Event;
import io.cloudevents.CloudEvent;
import io.cloudevents.core.builder.CloudEventBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.time.OffsetDateTime;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * One AMCP event bus per JVM: a single {@link KafkaEventBroker} and a single {@link AgentContext}.
 * Every bee, host and gateway of the JVM shares it. Subscriptions are per topic, never per bee.
 */
public final class MeshBus implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(MeshBus.class);

    private final Topics topics;
    private final String instanceId;
    private final KafkaEventBroker broker;
    private final AgentContext context;

    public MeshBus(String bootstrap, Topics topics, String instanceId) {
        this.topics = topics;
        this.instanceId = instanceId;
        this.broker = new KafkaEventBroker(bootstrap, topics.consumerGroup(), instanceId);
        this.context = new AgentContext("mesh-" + instanceId, broker);
    }

    public Topics topics() {
        return topics;
    }

    public String instanceId() {
        return instanceId;
    }

    public AgentContext context() {
        return context;
    }

    /** Starts the broker. Call after the subscriptions of this JVM are registered. */
    public void start() throws Exception {
        context.start().get(60, TimeUnit.SECONDS);
    }

    /** Publishes a payload as a CloudEvent with the "partitionkey" extension set to the given key. */
    public CompletableFuture<Void> publish(String topic, String partitionKey, Object payload) {
        CloudEvent ce = CloudEventBuilder.v1()
                .withId(UUID.randomUUID().toString())
                .withType("io.amcp.event." + topic)
                .withSource(URI.create("https://hivemind.org/mesh/" + instanceId))
                .withTime(OffsetDateTime.now())
                .withDataContentType("application/json")
                .withData(Codec.encode(payload))
                .withExtension("partitionkey", partitionKey)
                .build();
        return context.getEventBroker().publish(new Event(ce));
    }

    /** Fan-out subscription: every JVM of the mesh receives every matching event. */
    public String subscribe(String topic, Consumer<Event> handler) {
        return context.getEventBroker().subscribe(topic, guard(topic, handler));
    }

    /** Competing-consumer subscription: each event goes to one subscriber of the named queue. */
    public String subscribeQueue(String topic, String queue, Consumer<Event> handler) {
        return context.getEventBroker().subscribeQueue(topic, queue, guard(topic, handler));
    }

    private Consumer<Event> guard(String topic, Consumer<Event> handler) {
        return event -> {
            try {
                handler.accept(event);
            } catch (RuntimeException e) {
                log.warn("handler for {} failed: {}", topic, e.toString());
            }
        };
    }

    @Override
    public void close() throws Exception {
        context.stop().get(30, TimeUnit.SECONDS);
    }
}
