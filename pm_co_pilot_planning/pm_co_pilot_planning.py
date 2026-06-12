import faulthandler
faulthandler.enable()

import os
os.environ.setdefault("LANGSMITH_DISABLE_RUN_COMPRESSION", "true")

import rclpy
from rclpy.node import Node
from PyQt6.QtWidgets import QApplication
import sys
from pm_co_pilot_planning.submodules.PmCoPilotPlanningApp import PmCoPilotPlanningApp
from pm_co_pilot_planning.submodules.agent.executor_gate import ExecutorGate
from rclpy.executors import SingleThreadedExecutor
from rosidl_runtime_py.convert import message_to_ordereddict, get_message_slot_types
from rosidl_runtime_py.set_message import set_message_fields
from rosidl_runtime_py.utilities import get_message, get_service, get_interface
from rqt_py_common import message_helpers
from threading import Thread 

class PmCoPilotNode(Node):

    def __init__(self):
        super().__init__('gpt_co_pilot')
        self.get_logger().info('PM Co-Pilot started!')

        self.qt_window = PmCoPilotPlanningApp(self)
        
def main(args=None):
    rclpy.init(args=args)
    # Single-threaded executor driven via spin_once by the ExecutorGate, so the node's
    # message deserialization can be paused for the duration of each LLM call (it would
    # otherwise race the worker thread's GIL-releasing HTTP call and segfault inside
    # rosidl/DDS message construction).
    executor = SingleThreadedExecutor()

    app = QApplication(sys.argv)

    co_pilot_node = PmCoPilotNode()
    executor.add_node(co_pilot_node)

    gate = ExecutorGate()
    co_pilot_node.executor_gate = gate

    thread = Thread(target=gate.spin, args=(executor,))
    thread.start()

    try:
        co_pilot_node.qt_window.show()
        sys.exit(app.exec())

    finally:
        gate.stop()
        co_pilot_node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
    
if __name__ == '__main__':
    main()
    