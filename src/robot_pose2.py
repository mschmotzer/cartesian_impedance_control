import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Wrench, Pose
from sensor_msgs.msg import JointState  # Replace with correct message type for franka_robot_state
from franka_msgs.msg import FrankaRobotState
from std_srvs.srv import Trigger
import numpy as np
import json
from datetime import datetime
import os
import time
from scipy.spatial.transform import Rotation as R
import cv2
import mediapipe as mp
import pyzed.sl as sl


def quaternion_to_euler(quaternion):
     # Create a Rotation object from the quaternion
    rotation = R.from_quat(quaternion)
    # Convert to Euler angles (roll, pitch, yaw)
    euler_angles = rotation.as_euler('xyz', degrees=False)  # radians
    return euler_angles

class RobotTrajectoryLogger(Node):

    def __init__(self):
        super().__init__('robot_trajectory_logger')

        # Add the Pose publisher
        self.pose_publisher = self.create_publisher(Pose, "cartesian_impedance_control/reference_pose", 10)

        # Timer for logging robot state at 100Hz
        self.timer_log_data = self.create_timer(1.0 / 10.0, self.log_data)

        #Timer for Tracking:
        self.timer_tracking = self.create_timer(1.0 / 25.0, self.run_tracking)

        # Subscribe to the robot state
        self.subscription = self.create_subscription(
            FrankaRobotState,  # Replace with the correct message type for franka_robot_state
            '/franka_robot_state_broadcaster/robot_state',
            self.robot_state_callback,
            10)

        # Initialize state and variables
        self.first = True
        self.time_start = time.time()
        self.reference_pose = Pose()
        self.ee_pose = Pose()
        self.f_ext = Wrench()
        self.results=None
        np.set_printoptions(precision=8)
        self.transformation_matrix=np.eye(4)
        self.zero_pos =[0.2, 0.2, 0.3] #[0.4, 0.0, 0.4] 
        self.zero_orientation = [0.7071,    0.7071, 0.0, 0.0] 
        self.reference_pose.position.x =  self.zero_pos[0] #0.4
        self.reference_pose.position.y =  self.zero_pos[1] #0.7
        self.reference_pose.position.z =  self.zero_pos[2] #0.4
        self.reference_pose.orientation.w = self.zero_orientation[0] 
        self.reference_pose.orientation.x = self.zero_orientation[1] 
        self.reference_pose.orientation.y = self.zero_orientation[2] 
        self.reference_pose.orientation.z = self.zero_orientation[3] 
        self.q = self.zero_pos
        self.ee_euler_angles = quaternion_to_euler([self.ee_pose.orientation._x, self.ee_pose.orientation._y, self.ee_pose.orientation._z, self.ee_pose.orientation._w])
        self.reference_euler_angles = quaternion_to_euler([self.reference_pose.orientation._x, self.reference_pose.orientation._y, self.reference_pose.orientation._z, self.reference_pose.orientation._w])
        self.start = True   
        self.memory_points = 5
        self.last_hand_points = np.zeros([3,self.memory_points])
        self.replacement_time = 3 # Seconds
        self.prev_case = 0   # 0 if hand was tracked, 1 if hand is lost, 2 if lost on the way back to zero
        self.tracking_success = False
        self.min_z = 0
        self.min_d = 0.3 # Distance from last known point to camera position.
        self.trajectory = []
        self.transf_matrix_0_Cam=np.eye(4)
        self.transf_matrix_EE_Cam=np.array([[0,-1, 0, 0],[1,0 , 0,-0.15],[0, 0, 1, 0.122],[0, 0, 0, 1]])
        self.last_success_pos = np.zeros(3)
        self.target = np.zeros(3)
        self.ball_Radius_back = 0.05 #m 
        self.ball_Radius_static = 0.02
        self.hand_image_ratio = 0.07
        self.wait_time = time.time()

        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
        self.log_file = (f"robot_state_log_{timestamp}.json")
        print(self.log_file)

        #Tracking Attribute
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        if self.zed.open(self.init_params) != sl.ERROR_CODE.SUCCESS:
            print("Failed to open ZED camera.")
            exit()
        # Initialize MediaPipe Hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.runtime_parameters = sl.RuntimeParameters()
        self.image_zed = sl.Mat()
        self.depth_map = sl.Mat()
        self.left_cam_info = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    
    def calculate_transfM_from_pose(self):
        transf_matrix_0_EE=np.eye(4)
        rotation = R.from_quat([self.quaternion[0], self.quaternion[1], self.quaternion[2], self.quaternion[3]])  # [x, y, z, w]
        rotation_matrix = rotation.as_matrix()
        transf_matrix_0_EE[:3, :3] = rotation_matrix
        transf_matrix_0_EE[:3, 3] = self.q
        self.transf_matrix_0_Cam = transf_matrix_0_EE


    def calculate_traj(self, target):
        t0 = np.array([1, 0, 0, 0, 0, 0])  # Position at t=0
        t0_v = np.array([0, 1, 0, 0, 0, 0])  # Velocity at t=0
        t0_a = np.array([0, 0, 2, 0, 0, 0])  # Acceleration at t=0
        
        # Time powers for t = t_end
        t_end = self.replacement_time
        t_end_vec = np.array([1, t_end, t_end**2, t_end**3, t_end**4, t_end**5])
        t_end_v = np.array([0, 1, 2*t_end, 3*t_end**2, 4*t_end**3, 5*t_end**4])
        t_end_a = np.array([0, 0, 2, 6*t_end, 12*t_end**2, 20*t_end**3])

        A = np.vstack([t0, t0_v, t0_a, t_end_vec, t_end_v, t_end_a])


        v0 = 0.0
        v_end = 0.0
        a0 = 0.2
        a_end = 0.2
        x_sign = np.sign(target[0]- self.q[0])
        y_sign = np.sign(target[1]- self.q[1])
        z_sign = np.sign(target[2]- self.q[2])
        b_x = np.array([self.q[0], x_sign*v0, x_sign*a0, target[0], x_sign*v_end, x_sign*a_end])
        b_y = np.array([self.q[1], y_sign*v0, y_sign*a0, target[1], y_sign*v_end, y_sign*a_end])
        b_z = np.array([self.q[2], z_sign*v0, z_sign*a0, target[2], z_sign*v_end, z_sign*a_end])
        self.trajectory=np.array([np.linalg.solve(A, np.transpose(b_x)),np.linalg.solve(A, np.transpose(b_y)),np.linalg.solve(A, np.transpose(b_z))])

    def find_hand(self):
        self.prev_case=1

        if np.any(np.isnan(self.last_hand_points)) or np.any(np.isinf(self.last_hand_points)):
            print("Invalid values in self.last_hand_points!")

        # recalculate 
        mean_point = np.mean(self.last_hand_points, axis=1, keepdims=True)
        centered_points = self.last_hand_points - mean_point

        # Compute the covariance matrix
        covariance_matrix = centered_points @ centered_points.T

        # Eigen decomposition of the covariance matrix
        eigvals, eigvecs = np.linalg.eig(covariance_matrix)

        # The optimal direction is the eigenvector corresponding to the largest eigenvalue
        d_norm1 = eigvecs[:, np.argmax(eigvals)]
        
        d_norm = self.last_hand_points[:3,-1] - self.last_hand_points[:3,self.memory_points - 2]
        d_norm /= np.linalg.norm(d_norm)

        if d_norm1[2] < 0:  # z-Komponente überprüfen
            d_norm1 *= -1

            
        target_point_1 = self.last_hand_points[:3,-1] + self.min_d * d_norm1

        print("PRINT TARGET: ", target_point_1)
        print("Last Hand Point", self.last_hand_points[:,-1])
        print("Direction d: ", d_norm)
        print("Direction d1: ", d_norm1)
        print("Last Hand Points: ", self.last_hand_points)

        t = np.dot((self.q[:3]- self.last_hand_points[:3,-1]), d_norm)/np.dot(d_norm, d_norm)
        target_point_2 = self.last_hand_points[:3,-1] + t*d_norm

        R_Transp= np.transpose(self.transf_matrix_EE_Cam[:3,:3])
        t = self.transf_matrix_EE_Cam[:3,3]

        #target = target_point_1 if target_point_1[2] > target_point_2[2] else target_point_2
        self.last_t_point=target_point_1
        
        self.calculate_traj(self.last_t_point)
        self.time_start=time.time()
        

    def run_tracking(self):
        if self.first:
            self.first = False
            self.pose_publisher.publish(self.reference_pose)

            while True:
                self.pose_publisher.publish(self.reference_pose)
                self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
                self.frame = self.image_zed.get_data()
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGRA2BGR)
                self.results = self.hands.process(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
                cv2.imshow("Hand Tracking with ZED Depth", self.frame)
                cv2.waitKey(1)

                if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    if self.results.multi_hand_landmarks:
                        print('Hand detected')
                        break
               
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:

            # Retrieve image and depth data
            self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
            self.frame = self.image_zed.get_data()
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGRA2BGR)

            # Process the image with MediaPipe
            self.results = self.hands.process(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
            if self.results.multi_hand_landmarks:
                
                self.start=False
                self.hand = self.results.multi_hand_landmarks[0]
                self.wrist = self.hand.landmark[0]
                index = self.hand.landmark[5]
                confidence = self.results.multi_handedness[0].classification[0].score
                print(f"Hand detection confidence: {confidence}")
                self.wrist_x = int(self.wrist.x * self.frame.shape[1])
                self.wrist_y = int(self.wrist.y * self.frame.shape[0])
                index_x = int(index.x * self.frame.shape[1])
                index_y = int(index.y * self.frame.shape[0])

                if self.wrist_x > 0 and self.wrist_y > 0 and index_x > 0 and index_y > 0:
                    self.tracking_success= True
                    self.depth_value = self.depth_map.get_value(self.wrist_x, self.wrist_y)[1]      
                    self.z=self.depth_value/1000
                    self.x=(self.wrist_x-self.left_cam_info.cx)*self.z/self.left_cam_info.fx
                    self.y=(self.wrist_y-self.left_cam_info.cy)*self.z/self.left_cam_info.fy

                    if not np.isnan(self.depth_value) and not np.isinf(self.depth_value):
                        current_hand_point= np.dot(self.transf_matrix_0_Cam,np.transpose([self.x, self.y, self.z, 1]))
                        self.last_hand_points=np.roll(self.last_hand_points, -1)
                        self.last_hand_points[:,-1] = current_hand_point[:3]
                        self.last_success_pos = self.q[:3]
                        cv2.putText(self.frame, f'({current_hand_point[0]},{current_hand_point[1]},{current_hand_point[2]})', (self.wrist_x, self.wrist_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if not self.start and (time.time() - self.wait_time > 1.5*self.replacement_time):
                            if self.prev_case == 2 and  (time.time() - self.time_start) <= self.replacement_time :
                                self.send_trajectory(time.time() - self.time_start)
                            elif self.prev_case == 1 and self.inside_ball(self.ball_Radius_back):
                                self.calculate_traj(self.zero_pos)
                                print("here:", self.zero_pos)
                                self.time_start=time.time()
                                self.send_trajectory(time.time() - self.time_start)
                                self.prev_case = 2
                            elif (time.time() - self.time_start) <= self.replacement_time + 1:
                                self.reference_pose.position.x =  self.zero_pos[0] 
                                self.reference_pose.position.y =  self.zero_pos[1] 
                                self.reference_pose.position.z =  self.zero_pos[2]
                                self.reference_pose.orientation.w = self.zero_orientation[0] 
                                self.reference_pose.orientation.x = self.zero_orientation[1] 
                                self.reference_pose.orientation.y = self.zero_orientation[2] 
                                self.reference_pose.orientation.z = self.zero_orientation[3] 
                                self.pose_publisher.publish(self.reference_pose)
                                self.prev_case = 0
                        else:
                            if np.linalg.norm(np.array([index.x -self.wrist.x, index.y - self.wrist.y])) < self.hand_image_ratio  :
                                direction = current_hand_point[:3] - self.q  # Vector pointing to the hand
                                distance = np.linalg.norm(direction)  # Calculate the distance
                        
                                step = 0.01 * (direction / distance)  # Normalize and scale by step size
                                self.reference_pose.position.x += step[0]
                                self.reference_pose.position.y += step[1]
                                self.reference_pose.position.z += step[2]
                                self.calculate_orientation()
                                self.pose_publisher.publish(self.reference_pose)
                            print("waiting....")

                else:
                    self.wait_time = time.time() 
                    # analog wie unten dann anpassen
                    self.tracking_success= False

                    if  not self.inside_ball(self.ball_Radius_static):
                        self.find_hand()
                        self.send_trajectory(time.time() - self.time_start)

                    elif self.prev_case == 2:

                        self.calculate_traj(self.last_success_pos)
                        print(self.last_success_pos)
                        self.send_trajectory(time.time() - self.time_start)
                        self.prev_case = 1
                    
                    else:
                        target = self.q
                        target[:2] = self.q[:2] + 0.3*(self.last_hand_points[:2,-1] - self.q[:2])
                        target[2] += 0.3
                        print(target)
                        self.calculate_traj(target)
                        self.send_trajectory(time.time() - self.time_start)
                        self.prev_case = 1
    
            else:
                self.tracking_success= False
                self.wait_time = time.time() 

               

                if self.prev_case == 2 and self.inside_ball(self.ball_Radius_back):

                    self.calculate_traj(self.last_success_pos)
                    self.time_start=time.time()
                    self.send_trajectory(time.time() - self.time_start)
                    print("Lost on the way back")
                    self.prev_case = 1

                elif self.prev_case == 0 and self.inside_ball(self.ball_Radius_static):

                    target = self.q
                    target[:2] = self.q[:2] + 0.3*(self.last_hand_points[:2,-1] - self.q[:2])
                    target[2] += 0.3
                    self.calculate_traj(target)
                    self.time_start=time.time()
                    self.send_trajectory(time.time() - self.time_start)
                    self.prev_case = 1
                    print("static cas") 

                elif self.prev_case == 0 or self.prev_case == 2:

                    self.find_hand()
                    self.send_trajectory(time.time() - self.time_start)
                    print("mov case")   

                elif (time.time() - self.time_start) <= self.replacement_time:
                    self.send_trajectory(time.time() - self.time_start)

                elif (time.time() - self.time_start) <= self.replacement_time + 2:
                    #just wait
                    print("wait")
                else:
                    print('Couldnt find hand')
                    exit()

            cv2.imshow("Hand Tracking with ZED Depth", self.frame)
            cv2.waitKey(1)


    def calculate_orientation(self):
        ref_orient=[self.reference_pose.position.x, self.reference_pose.position.y, self.reference_pose.position.z]
        z = self.last_hand_points[:3,-1] - ref_orient
        z /= np.linalg.norm(z)  # Normalize z to ensure it's a unit vector.

        # Use the current x-axis to find the closest x direction while maintaining orthogonality to z.
        current_x_axis = self.transf_matrix_0_Cam[:3,0]  # Replace with your actual current x-axis if different.
        x = current_x_axis - np.dot(current_x_axis, z) * z  # Remove the component of current_x_axis along z.
        x /= np.linalg.norm(x)  # Normalize to ensure x is a unit vector.

        # Compute y as orthogonal to both z and x.
        y = np.cross(z, x)  # Cross product to get the third orthogonal vector.

        # Construct the rotation matrix.
        rotation_matrix = np.column_stack((x, y, z))

        # Convert the rotation matrix to quaternion.
        rotation = R.from_matrix(rotation_matrix)
        ref_quaternion = rotation.as_quat() 
        self.reference_pose.orientation._x = ref_quaternion[0]
        self.reference_pose.orientation._y = ref_quaternion[1]
        self.reference_pose.orientation._z = ref_quaternion[2]
        self.reference_pose.orientation._w = ref_quaternion[3]
        self.reference_euler_angles = rotation.as_euler('xyz', degrees=False)
        
        

    def send_trajectory(self,current_time):
        time_vec = np.array([1, current_time, current_time**2, current_time**3, current_time**4, current_time**5])

        self.reference_pose.position.x = np.dot(time_vec, self.trajectory[0])
        self.reference_pose.position.y = np.dot(time_vec, self.trajectory[1])
        self.reference_pose.position.z = np.dot(time_vec, self.trajectory[2])
        #print(self.reference_pose.position)
        self.calculate_orientation()
        #self.get_logger().info(f'Sent trajectory Point at time: {current_time:.2f} seconds \n Point: ({self.reference_pose.position.x,self.reference_pose.position.y,self.reference_pose.position.z})')
 
        self.pose_publisher.publish(self.reference_pose)
    def inside_ball(self, radius):

        # Calculate the centroid (center) of the data
        centroid = np.mean(self.last_hand_points, axis=1)
        
        # Calculate distances of all points from the centroid
        distances = np.linalg.norm(self.last_hand_points - centroid[:, np.newaxis], axis=0)
        
        # Check if all distances are within the radius
        return np.all(distances <= radius)

    def robot_state_callback(self, msg: FrankaRobotState):
        self.f_ext = msg._o_f_ext_hat_k._wrench  # Assuming this is the correct attribute
        self.ee_pose.position = msg.o_t_ee._pose._position
        self.ee_pose.orientation = msg.o_t_ee._pose._orientation
        self.q = np.array([self.ee_pose.position._x, self.ee_pose.position._y,self.ee_pose.position._z]) 
        self.quaternion = [self.ee_pose.orientation._x, self.ee_pose.orientation._y, self.ee_pose.orientation._z, self.ee_pose.orientation._w]
        self.ee_euler_angles = quaternion_to_euler(self.quaternion)
        self.calculate_transfM_from_pose()

    def log_data(self):
            # Ensure data is available before logging
        if self.f_ext is None or self.reference_pose is None or self.ee_euler_angles is None:
            self.get_logger().warn("f_ext is None. Skipping log entry.")
            return
        
        # Prepare the data to log
        data_to_log = {
            "reference_position": {
                "x": self.reference_pose.position._x,
                "y": self.reference_pose.position._y,
                "z": self.reference_pose.position._z
            },
            "euler_angles": {
                "roll": self.reference_euler_angles[0],
                "pitch": self.reference_euler_angles[1],
                "yaw": self.reference_euler_angles[2]
            },
            "ee_pose":
            {
                "position": {
                    "x": self.ee_pose.position._x,
                    "y": self.ee_pose.position._y,
                    "z": self.ee_pose.position._z
                },
                "orientation": {
                    "roll": self.ee_euler_angles[0],
                    "pitch": self.ee_euler_angles[1],
                    "yaw": self.ee_euler_angles[2]
                }
            },
            "Tracking": {
                    "Success": self.tracking_success 
            }
            
        }

        # Log data to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data_to_log) + '\n')

def main(args=None):
    rclpy.init(args=args)
    node = RobotTrajectoryLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
