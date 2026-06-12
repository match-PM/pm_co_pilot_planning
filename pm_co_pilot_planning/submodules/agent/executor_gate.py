"""
ExecutorGate: serialize the ROS executor against the agent's LLM calls.

The node is spun by a background thread while the agent runs LLM (HTTP) calls on a
separate worker thread. During an LLM call the GIL is released (C socket read), letting
the executor run rosidl/DDS C deserialization concurrently with the worker. That
concurrent message construction corrupts interpreter state and segfaults (observed in
`ObjectScene` / `FrConstraints` deserialization).

ExecutorGate replaces the raw `executor.spin()` background loop with a pausable
`spin_once` loop. `GatePauseCallback` pauses the executor for the duration of every LLM
call (`on_llm_start` → pause, `on_llm_end` / `on_llm_error` → resume), with a
paused-ack handshake that guarantees no `spin_once` (deserialization/callback) is
in-flight when the worker begins its HTTP call.
"""

import time
import threading

import rclpy
from langchain_core.callbacks import BaseCallbackHandler


class ExecutorGate:
    """Drive a ROS executor via `spin_once`, with a pause/resume handshake."""

    def __init__(self):
        self._pause_request = threading.Event()
        self._paused_ack = threading.Event()
        self._stop = False

    def spin(self, executor) -> None:
        """Background spin loop. Run in a dedicated thread: Thread(target=gate.spin, args=(executor,))."""
        while rclpy.ok() and not self._stop:
            if self._pause_request.is_set():
                # Reached the top of the loop with no spin_once in flight: acknowledge,
                # then idle until the worker resumes us.
                self._paused_ack.set()
                while self._pause_request.is_set() and rclpy.ok() and not self._stop:
                    time.sleep(0.005)
                self._paused_ack.clear()
                continue
            executor.spin_once(timeout_sec=0.05)

    def pause(self) -> None:
        """Block the calling (worker) thread until the executor is idle and paused.

        Returns only once the spinner has finished any in-flight `spin_once` and parked
        at the top of its loop, so no message deserialization overlaps what the caller
        does next (the LLM HTTP call).
        """
        if self._stop:
            return
        self._paused_ack.clear()
        self._pause_request.set()
        # Wait for the spinner to acknowledge. Bounded waits keep us from hanging
        # forever if the spin thread is tearing down during shutdown.
        while not self._paused_ack.wait(timeout=0.5):
            if self._stop or not rclpy.ok():
                return

    def resume(self) -> None:
        """Let the executor resume spinning."""
        self._pause_request.clear()

    def stop(self) -> None:
        """Signal the spin loop to exit (clean shutdown)."""
        self._stop = True
        self._pause_request.clear()


class GatePauseCallback(BaseCallbackHandler):
    """Pause the node's ExecutorGate around each LLM call.

    Reads `node.executor_gate` lazily because the gate is attached to the node after the
    agent (and this handler) are constructed. No-op if no gate is present.
    """

    def __init__(self, node):
        self._node = node

    def _gate(self):
        return getattr(self._node, "executor_gate", None)

    def on_llm_start(self, *args, **kwargs) -> None:
        gate = self._gate()
        if gate is not None:
            gate.pause()
            self._node.get_logger().debug("ExecutorGate: paused for LLM call")

    def on_llm_end(self, *args, **kwargs) -> None:
        gate = self._gate()
        if gate is not None:
            gate.resume()
            self._node.get_logger().debug("ExecutorGate: resumed after LLM call")

    def on_llm_error(self, *args, **kwargs) -> None:
        gate = self._gate()
        if gate is not None:
            gate.resume()
            self._node.get_logger().debug("ExecutorGate: resumed after LLM error")
