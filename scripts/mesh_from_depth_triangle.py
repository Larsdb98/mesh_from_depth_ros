#!/usr/bin/env python3
#################################################################################

import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
import os
import cv_bridge
# import cv2
import numpy as np
import open3d as o3d 
import math
import copy
import tf
# from skimage.io import imread # this will be removed since we're using images coming from ROS messages
from tqdm import tqdm # Not necessary as well, but may be useful for dev and debugging

# NOTE: This script used the following repo to compute the mesh from depth images:
# https://github.com/hesom/depth_to_mesh/tree/master
# This node allows us to create a triangle mesh from depth image messages from ROS


class triangle_mesh_from_depth:
    def __init__(self):
        rospy.init_node('triangle_mesh_from_depth', anonymous = True)
        self.bridge = cv_bridge.CvBridge()
        self.tf_listener = tf.TransformListener()
        
        # Placeholder for variables for camera intrinsics
        self.depth_camera_intrinsics = None
        self.depth_camera_intrinsics_header = None
        self.depth_image = None
        self.depth_image_header = None
        self.rgb_image = None
        self.rgb_image_header = None

        # Placeholder for tf recorded when depth image was acquired
        self.tf_translation = None
        self.tf_rotation = None

        # default camera intrinsics: 
        # These are the default camera intrinsics for the Intel Realsense D435 
        # mounted on the MoMa robot arm. The camera is parametrized to publish 640x480 images (rgb & depth)
        self.DEFAULT_CAMERA = o3d.camera.PinholeCameraIntrinsic(
            width=640, height=480,
            fx=613.7755737304688, fy=613.7672119140625,
            cx=319.5116271972656, cy=250.294189453125
        )


        # ROS params
        self.__depth_image_topic = rospy.get_param("~depth_image_topic", "/wrist_camera/aligned_depth_to_color/image_raw")
        self.__camera_intrinsics_topic = rospy.get_param("~depth_intrinsics", "/wrist_camera/aligned_depth_to_color/camera_info")
        self.__rgb_image_topic = rospy.get_param("~rgb_image_topic", "/wrist_camera/color/image_raw")
        self.__export_directory = rospy.get_param("~export_directory", "/home/lars/Master_Thesis_workspaces/VIS4ROB_Vulkan_Glasses/catkin_ws/src/mapping_pipeline_packages/mesh_from_depth/output_mesh")
        self.__export_success_topic = rospy.get_param("~export_success_topic", "mesh_from_depth_success")
        
        self.__depth_scale = rospy.get_param("~depth_scale", 1000.0)
        
        self.__camera_frame_id = rospy.get_param("~camera_frame_id", "wrist_camera_color_optical_frame")
        self.__target_frame_id = rospy.get_param("~target_frame_id", "world")

        # Setting up subscribers
        rospy.Subscriber(self.__depth_image_topic, Image, self.depth_image_callback)
        rospy.Subscriber(self.__camera_intrinsics_topic, CameraInfo, self.depth_intrinsics_callback)
        rospy.Subscriber(self.__rgb_image_topic, Image, self.rgb_image_callback)

        # Setting up publishers: export flag
        self.export_success_pub = rospy.Publisher(self.__export_success_topic, Bool, queue_size=10)


    # The depth callback is essentially the highlevel method here.
    def depth_image_callback(self, img_msg):
        try:
            # Get the transform fro the depth frame to world
            (trans, rot) = self.tf_listener.lookupTransform(self.__target_frame_id, self.__camera_frame_id, rospy.Time(0))
            # (trans, rot) = self.tf_listener.lookupTransform(self.__camera_frame_id, self.__target_frame_id, rospy.Time(0))
            # Create a 3x3 rotation matrix from the transform
            self.tf_rotation = np.array(self.tf_listener.fromTranslationRotation(trans, rot)[:3, :3])
            self.tf_translation = np.array(trans)

            # print("tf_rotation matrix: {}".format(self.tf_rotation))
            # print("tf_translation: {}".format(self.tf_translation))

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("Error looking up transform ! Ignoring depth frame... %s", str(e))
            return
        

        rospy.loginfo("depth image recieved ! Processing...")

        self.depth_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
        self.depth_image_header = img_msg.header

        camera_matrix = self.DEFAULT_CAMERA

        if self.depth_camera_intrinsics is None:
            rospy.logwarn("Camera intrinsics not yet recieved ! Using default camera model...")
        else:
            camera_matrix = self.depth_camera_intrinsics


        mesh = self.depth_image_to_mesh(self.depth_image, 
                                            cameraMatrix = camera_matrix, 
                                            minAngle=3.0,
                                            depthScale=self.__depth_scale, 
                                            rotation_matrix=self.tf_rotation,
                                            translation=self.tf_translation)
        
        # print("DEBUG PRE mesh object: {}".format(mesh))

        self.save_mesh_as_obj(mesh=mesh, export_dir = self.__export_directory)

        # print("Depth image shape: {}".format(self.depth_image.shape))



    def depth_intrinsics_callback(self, intrinsics_msg):
        self.depth_camera_intrinsics = np.array(intrinsics_msg.K).reshape((3, 3))


    def rgb_image_callback(self, img_msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        self.rgb_image_header = img_msg.header


    def save_mesh_as_obj(self, mesh, export_dir):
        rospy.logdebug("Exporting mesh as .obj file")
        # Filename for .obj mesh file:
        obj_filename = os.path.join(export_dir, "workspace_mesh.obj")
        o3d.io.write_triangle_mesh(obj_filename, mesh, 
                                    write_ascii = True, 
                                    write_vertex_normals = False) # can't be exported into .obj files anyways
        rospy.loginfo("Mesh file successfully exported.")


    ########################################################################
    ### METHODS COPIED AND MODIFIED FROM THE GITHUB REPO MENTIONED ABOVE ###
    ########################################################################

    def _pixel_coord_np(self, width, height):
        """
        Pixel in homogenous coordinate
        Returns:
            Pixel coordinate:       [3, width * height]
        """
        x = np.linspace(0, width - 1, width).astype(np.int32)
        y = np.linspace(0, height - 1, height).astype(np.int32)
        [x, y] = np.meshgrid(x, y)
        return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

    def depth_image_to_mesh(self, image, cameraMatrix=None, minAngle=3.0, depthScale=1000.0, rotation_matrix = np.eye(3), translation = np.zeros(3)):
        """
        Converts a depth image file into a open3d TriangleMesh object

        :param image: depth image as a numpy array, luckily that is the same as opencv type
        :param cameraMatrix: numpy array of the intrinsic camera matrix
        :param minAngle: Minimum angle between viewing rays and triangles in degrees
        :param sun3d: Specify if the depth file is in the special SUN3D format
        :returns: an open3d.geometry.TriangleMesh containing the converted mesh
        """
        # Default value for camera intrinsics
        if cameraMatrix is None:
            cameraMatrix = self.DEFAULT_CAMERA



        depth_raw = copy.copy(image) # transfer depth image to "depth_raw" variable.
        width = depth_raw.shape[1]
        height = depth_raw.shape[0]

        # if sun3d:
        #     depth_raw = np.bitwise_or(depth_raw>>3, depth_raw<<13)
        depth_raw = depth_raw.astype('float32')
        depth_raw /= depthScale

        # logger.debug('Image dimensions:%s x %s', width, height)
        # logger.debug('Camera Matrix:%s', cameraMatrix)

        if cameraMatrix is None:
            camera = self.DEFAULT_CAMERA
        else:
            camera = o3d.camera.PinholeCameraIntrinsic(
                width=width, height=height,
                fx=cameraMatrix[0,0], fy=cameraMatrix[1,1],
                cx=cameraMatrix[0,2], cy=cameraMatrix[1,2]
            )
        return self.depth_to_mesh(depth_raw.astype('float32'), camera, minAngle, rotation_matrix = rotation_matrix, translation = translation)

    def depth_to_mesh(self, depth, camera=None, minAngle=3.0, rotation_matrix = np.eye(3), translation = np.zeros(3)):
        """
        Converts an open3d.geometry.Image depth image into a open3d.geometry.TriangleMesh object

        :param depth: np.array of type float32 containing the depth image
        :param camera: open3d.camera.PinholeCameraIntrinsic
        :param minAngle: Minimum angle between viewing rays and triangles in degrees
        :returns: an open3d.geometry.TriangleMesh containing the converted mesh
        """

        # Default value for camera intrinsics
        if camera is None:
            camera = self.DEFAULT_CAMERA

        # logger.info('Reprojecting points...')
        K = camera.intrinsic_matrix
        K_inv = np.linalg.inv(K)
        pixel_coords = self._pixel_coord_np(depth.shape[1], depth.shape[0])
        cam_coords = K_inv @ pixel_coords * depth.flatten()

        # print("TYPE OF CAM_COORDS: {}".format(type(cam_coords)))
        # print("SHAPE OF CAM_COORDS: {}".format(cam_coords.shape))

        # print("cam_coords dimensions: {}".format(cam_coords.shape))

        rospy.logdebug("rotation matrix: {}".format(rotation_matrix))
        # print("cam_coords: {}".format(cam_coords[:, 0:10]))

        # Rotate points based on TF information
        cam_coords = rotation_matrix @ cam_coords

        # Translate points based on TF information

        # print("cam_coords dimensions: {}".format(cam_coords.shape))
        # print("cam_coords[0, :] dimensions: {}".format(cam_coords[0, :].shape))
        # print("np.ones_like(cam_coords): {}".format(np.ones_like(cam_coords).shape))

        # cam_coords[0, :] = cam_coords[0, :] + translation[0] * np.ones_like(cam_coords[0, :])
        # cam_coords[1, :] = cam_coords[1, :] + translation[1] * np.ones_like(cam_coords[1, :])
        # cam_coords[2, :] = cam_coords[2, :] + translation[2] * np.ones_like(cam_coords[2, :])


        # print("cam_coords: {}".format(cam_coords[:, 0:10]))
        # print("SHAPE OF CAM_COORDS: {}".format(cam_coords.shape))

        indices = o3d.utility.Vector3iVector()
        w = camera.width
        h = camera.height
        
        # The following few lines are experimmental, trying to figure out how to map
        # the indices (or triangle vertex-indices) to a corresponding UV coordinate.
        # This might make me figure out how the hell I can create a texture file (png)
        # to texture this mesh with a separate texture file with the Vulkan rendering engine.
        # 
        # It's important to point out that mesh.triangle_uvs expects an array of normalized UV coords
        # defined between [0, 1] and has a dimension of (3 * num_triangles, 2). Each vertex that forms
        # the triangles need to be appended to the list below:
        uv_mapping = []


        with tqdm(total=(h-1)*(w-1)) as pbar:
            for i in range(0, h-1):
                for j in range(0, w-1):
                    verts = [
                        cam_coords[:, w*i+j],
                        cam_coords[:, w*(i+1)+j],
                        cam_coords[:, w*i+(j+1)],
                    ]
                    if [0,0,0] in map(list, verts):
                        continue
                    v1 = verts[0] - verts[1]
                    v2 = verts[0] - verts[2]
                    n = np.cross(v1, v2)
                    n /= np.linalg.norm(n)
                    center = (verts[0] + verts[1] + verts[2]) / 3.0
                    u = center / np.linalg.norm(center)
                    angle = math.degrees(math.asin(abs(np.dot(n, u))))
                    if angle > minAngle:
                        indices.append([w*i+j, w*(i+1)+j, w*i+(j+1)])

                        u1 = float(i)/h
                        v1 = float(j)/w
                        u2 = float(i+1)/h
                        v2 = float(j)/w
                        u3 = float(i)/h
                        v3 = float(j+1)/w
                        uv_mapping.append([u1, v1]) # Lars: added this for the uv mapping
                        uv_mapping.append([u2, v2])
                        uv_mapping.append([u3, v3])

                    verts = [
                        cam_coords[:, w*i+(j+1)],
                        cam_coords[:, w*(i+1)+j],
                        cam_coords[:, w*(i+1)+(j+1)],
                    ]
                    if [0,0,0] in map(list, verts):
                        continue
                    v1 = verts[0] - verts[1]
                    v2 = verts[0] - verts[2]
                    n = np.cross(v1, v2)
                    n /= np.linalg.norm(n)
                    center = (verts[0] + verts[1] + verts[2]) / 3.0
                    u = center / np.linalg.norm(center)
                    angle = math.degrees(math.asin(abs(np.dot(n, u))))
                    if angle > minAngle:
                        indices.append([w*i+(j+1),w*(i+1)+j, w*(i+1)+(j+1)])
                        
                        u1 = float(i)/h
                        v1 = float(j+1)/w
                        u2 = float(i+1)/h
                        v2 = float(j)/w
                        u3 = float(i+1)/h
                        v3 = float(j+1)/w
                        uv_mapping.append([u1, v1]) # Lars: added this for the uv mapping
                        uv_mapping.append([u2, v2])
                        uv_mapping.append([u3, v3])
                    pbar.update(1)

        uv_mapping_np = np.array(uv_mapping)

        points_np = cam_coords.transpose()
        indices_np = np.array(indices)

        points = o3d.utility.Vector3dVector(cam_coords.transpose())

        triangle_uv_vector = o3d.utility.Vector2dVector(uv_mapping_np)

        mesh = o3d.geometry.TriangleMesh(points, indices)

        triangle_uv = mesh.triangle_uvs

        print("Size of mesh.triangle_uvs before adding the uv coords: {}".format(triangle_uv))

        mesh.triangle_uvs = triangle_uv_vector

        triangle_uv = mesh.triangle_uvs

        print("Size of mesh.triangle_uvs before adding the uv coords: {}".format(triangle_uv))

        print("")

        # points = o3d.utility.Vector3dVector(cam_coords.transpose())

        print("Mesh debug: {}".format(mesh))

        # print("DEBUG: points: {}".format(points))
        # print("DEBUG: indices: {}".format(indices))

        # device  = o3d.core.Device("CPU:0")
        # dtype_f = o3d.core.float32
        # dtype_i = o3d.core.int32

        # tensor_points = o3d.core.Tensor(points_np, dtype_f, device)
        # tensor_triangle_indices = o3d.core.Tensor(indices_np, dtype_i, device)

        # print("Tensor_points: {}".format(tensor_points))
        # print("Tensor_triangles: {}".format(tensor_triangle))

        # print("MESH CREATE DEBUG: {}".format(mesh))

        # tensor_mesh = o3d.t.geometry.TriangleMesh(device)
        # tensor_mesh.vertex.positions = tensor_points
        # tensor_mesh.triangle.indices = tensor_triangle_indices

        # print("TENSOR TRIANGLEMESH DEBUG: {}".format(tensor_mesh))

        return mesh # tensor_mesh
    



    # For the actual ROS looping
    def run(self):
        rospy.spin()


###########################################################
###########################################################
def main():
    node = triangle_mesh_from_depth()
    node.run()




if __name__ == "__main__":
    main()