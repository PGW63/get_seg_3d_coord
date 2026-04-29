from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    package_share = get_package_share_directory("get_seg_3d_coord")
    params_file = os.path.join(package_share, "config", "params.yaml")

    table_person_association = Node(
        package="get_seg_3d_coord",
        executable="table_person_association",
        name="table_person_association",
        output="screen",
        parameters=[params_file],
    )

    return LaunchDescription([
        table_person_association,
    ])
