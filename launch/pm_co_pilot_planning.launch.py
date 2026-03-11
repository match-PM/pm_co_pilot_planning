import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # Define a ROS 2 node
    pm_co_pilot_app_node = Node(
        package='pm_co_pilot_planning',
        executable='pm_co_pilot_planning',
        name='PM_Co_Pilot_Planning',
        #parameters=[{'param_name': LaunchConfiguration('my_param')}]
        emulate_tty=True
    )
    
    ld = LaunchDescription()
    ld.add_action(pm_co_pilot_app_node)

    return ld