import rclpy
from rclpy.node import Node
from PyQt6.QtWidgets import QApplication
import sys
from pm_co_pilot_planning.submodules.PmCoPilotProgrammingApp import PmCoPilotProgrammingApp
from rclpy.executors import MultiThreadedExecutor
from rosidl_runtime_py.convert import message_to_ordereddict, get_message_slot_types
from rosidl_runtime_py.set_message import set_message_fields
from rosidl_runtime_py.utilities import get_message, get_service, get_interface
from rqt_py_common import message_helpers
from threading import Thread 

class PmCoPilotNode(Node):

    def __init__(self):
        super().__init__('gpt_co_pilot')
        self.get_logger().info('PM Co-Pilot started!')

        self.qt_window = PmCoPilotProgrammingApp(self)
        
def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor(num_threads=6) 

    app = QApplication(sys.argv)

    co_pilot_node = PmCoPilotNode()
    executor.add_node(co_pilot_node)

    thread = Thread(target=executor.spin)
    thread.start()
    
    try:
        co_pilot_node.qt_window.show()
        sys.exit(app.exec())

    finally:
        co_pilot_node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
    
if __name__ == '__main__':
    main()
    