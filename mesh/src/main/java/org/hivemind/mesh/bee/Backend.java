package org.hivemind.mesh.bee;

import org.hivemind.mesh.wire.Wire;

import java.util.List;

/** The model a bee calls to answer a task. Implementations may block; bees call them on virtual threads. */
public interface Backend {

    String complete(String tier, List<Wire.Message> messages) throws Exception;
}
