"""
Run the co-pilot's ROS node on one controllable executor thread.

The planning window has two launch modes:

* as its own executable; and
* embedded in the ROS Sequential Action Programmer (RSAP).

In the embedded mode the node already belongs to RSAP's ``MultiThreadedExecutor``.
Merely constructing an :class:`ExecutorGate` in the standalone entry point therefore
does nothing for the mode used most often.  In particular, ObjectScene
deserialisation can continue on several rclpy worker threads while a LangGraph worker
executes an RSAP service action.  The generated rosidl Python bindings have been
observed to segfault in that situation.

``ExecutorGate.acquire`` migrates the node to a ``SingleThreadedExecutor`` owned by
the co-pilot, after waiting for callbacks already submitted by the previous executor
to finish.  The previous executor is restored when the last co-pilot window releases
the gate.  LLM callbacks can also temporarily pause this executor with an
acknowledged pause handshake.
"""

import threading
from typing import Optional, Tuple

import rclpy
from langchain_core.callbacks import BaseCallbackHandler
from rclpy.executors import SingleThreadedExecutor


class ExecutorGate:
    """Own and pause a single-threaded executor for one ROS node."""

    _NODE_ATTRIBUTE = "executor_gate"

    def __init__(self, node):
        self._node = node
        self._executor = None
        self._previous_executor = None
        self._thread: Optional[threading.Thread] = None
        self._condition = threading.Condition()
        self._pause_count = 0
        self._paused = False
        self._stopped = False
        self._lease_count = 0

    @classmethod
    def acquire(cls, node) -> Tuple["ExecutorGate", bool]:
        """Attach (or reuse) a gate and acquire one window-lifetime lease.

        Returns ``(gate, created)``.  Every successful acquire must be paired with
        :meth:`release`.
        """
        gate = getattr(node, cls._NODE_ATTRIBUTE, None)
        created = gate is None
        if created:
            gate = cls(node)
            gate._attach()
            setattr(node, cls._NODE_ATTRIBUTE, gate)

        with gate._condition:
            gate._lease_count += 1
        return gate, created

    def _attach(self) -> None:
        """Move the node off its current executor and start the controlled executor."""
        previous_executor = self._node.executor
        if previous_executor is not None:
            previous_executor.remove_node(self._node)
            # remove_node prevents new callbacks from being selected, but a
            # MultiThreadedExecutor may still have handlers in its worker pool.  Waiting
            # here is essential before the same node is added to another executor.
            work_tracker = getattr(previous_executor, "_work_tracker", None)
            if work_tracker is not None:
                work_tracker.wait()

        executor = SingleThreadedExecutor(context=self._node.context)
        executor.add_node(self._node)

        self._previous_executor = previous_executor
        self._executor = executor
        self._thread = threading.Thread(
            target=self.spin,
            name="pm-co-pilot-ros-executor",
            daemon=True,
        )
        self._thread.start()
        self._node.get_logger().info(
            "Planning co-pilot attached a controlled single-threaded ROS executor."
        )

    def spin(self) -> None:
        """Drive the owned executor, honouring acknowledged pause requests."""
        while rclpy.ok(context=self._node.context):
            with self._condition:
                if self._stopped:
                    return
                if self._pause_count:
                    self._paused = True
                    self._condition.notify_all()
                    self._condition.wait_for(
                        lambda: self._pause_count == 0 or self._stopped,
                        timeout=0.5,
                    )
                    if self._pause_count == 0:
                        self._paused = False
                        self._condition.notify_all()
                    continue

            # The short timeout bounds how long pause() waits for an in-flight
            # spin_once to return.
            self._executor.spin_once(timeout_sec=0.05)

    def pause(self) -> None:
        """Wait until no executor callback or message deserialisation is in flight."""
        with self._condition:
            if self._stopped:
                return
            self._pause_count += 1
            self._condition.notify_all()
            self._condition.wait_for(
                lambda: self._paused or self._stopped
            )

    def resume(self) -> None:
        """Release one matching pause request."""
        with self._condition:
            if self._pause_count:
                self._pause_count -= 1
            self._condition.notify_all()

    def release(self) -> None:
        """Release one window lease and restore the previous executor at zero."""
        with self._condition:
            if self._lease_count == 0:
                return
            self._lease_count -= 1
            should_detach = self._lease_count == 0

        if should_detach:
            self._detach()

    def _detach(self) -> None:
        """Stop the controlled executor and return the node to its former executor."""
        with self._condition:
            if self._stopped:
                return
            self._stopped = True
            self._pause_count = 0
            self._condition.notify_all()

        if self._executor is not None:
            self._executor.wake()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

        if self._executor is not None:
            self._executor.remove_node(self._node)
            self._executor.shutdown(timeout_sec=2.0)

        if getattr(self._node, self._NODE_ATTRIBUTE, None) is self:
            delattr(self._node, self._NODE_ATTRIBUTE)

        if self._previous_executor is None:
            # remove_node does not clear Node.executor itself.  Avoid leaving the
            # standalone node pointing at a stopped executor.
            self._node.executor = None
        elif rclpy.ok(context=self._node.context):
            try:
                self._previous_executor.add_node(self._node)
                self._node.get_logger().info(
                    "Planning co-pilot restored the previous ROS executor."
                )
            except Exception as exc:
                # Shutdown may destroy the node before Qt closes child windows.
                self._node.get_logger().warning(
                    f"Could not restore the previous ROS executor during shutdown: {exc}"
                )


class GatePauseCallback(BaseCallbackHandler):
    """Pause the node's attached gate for the duration of each LLM call."""

    def __init__(self, node):
        self._node = node

    def _gate(self):
        return getattr(self._node, ExecutorGate._NODE_ATTRIBUTE, None)

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
