import rclpy
from rclpy.node import Node
from messages_fr3.srv import SetPose, SetParam
from geometry_msgs.msg import Pose
import math

class UserInputClient(Node):

    def __init__(self):
        super().__init__('user_input_client')
        self.pose_client = self.create_client(SetPose, 'set_pose')
        self.param_client = self.create_client(SetParam, 'set_param')

        self.pose_subscriber = self.create_subscription(
            Pose,
            'current_pose',
            self.pose_callback,
            10
        )
        self.get_logger().info('Pose subscriber created')

    def pose_callback(self, msg):
        self.get_logger().info(f'Received Pose: {msg.position.x}, {msg.position.y}, {msg.position.z}')

    def send_pose_request(self, x, y, z, roll, pitch, yaw):
        pose_request = SetPose.Request()
        pose_request.x = x
        pose_request.y = y
        pose_request.z = z
        pose_request.roll = roll
        pose_request.pitch = pitch
        pose_request.yaw = yaw
        
        if self.pose_client.wait_for_service(timeout_sec=1.0):
            future = self.pose_client.call_async(pose_request)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                self.get_logger().info(f'Pose set successfully: {future.result().success}')
            else:
                self.get_logger().error('Failed to call service setPose')

    def send_param_request(self, a, b, c, d, e, f):
        param_request = SetParam.Request()
        param_request.a = a
        param_request.b = b
        param_request.c = c
        param_request.d = d
        param_request.e = e
        param_request.f = f

        if self.param_client.wait_for_service(timeout_sec=1.0):
            future = self.param_client.call_async(param_request)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                self.get_logger().info(f'Inertia parameters set: {future.result().success}')
            else:
                self.get_logger().error('Failed to call service setParam')

def main(args=None):
    rclpy.init(args=args)
    user_input_client = UserInputClient()

    while rclpy.ok():
        print("Enter the next task: \n [1] --> Change position \n [2] --> Change impedance parameters")
        task_selection = int(input())

        if task_selection == 1:
            print("Enter new goal position: \n [1] --> 0.5, -0.4, 0.5 \n [2] --> DO NOT USE \n [3] --> 0.5, 0.4, 0.5")
            pose_selection = int(input())
            if pose_selection == 1:
                user_input_client.send_pose_request(0.5, -0.4, 0.5, math.pi, 0.0, math.pi/2)
            elif pose_selection == 3:
                user_input_client.send_pose_request(0.5, 0.4, 0.5, math.pi, 0.0, math.pi/2)
            else:
                print("Enter your desired position and orientation")
                pose = [float(input()) for _ in range(6)]
                user_input_client.send_pose_request(*pose)

        elif task_selection == 2:
            print("Enter new inertia: \n [1] --> N/A \n [2] --> N/A \n [3] --> N/A")
            param_selection = int(input())
            if param_selection == 1:
                user_input_client.send_param_request(2, 0.5, 0.5, 2, 0.5, 0.5)
            else:
                user_input_client.send_param_request(1, 1, 1, 1, 1, 1)

        else:
            print("Invalid selection, please try again")

    user_input_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
