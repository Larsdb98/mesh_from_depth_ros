# Depth to Mesh ROS Package

Warning: This package, although functional, is still in development.

ROS distro: Noetic (Ubuntu 20.04)

This ROS package allows to convert depth images (sensor_msgs/Image) from RGBD sensors into meshes. It uses the uniform (2D image-like) structure of depth maps for its triangulation. Degenerate triangles and triangles that are likely wrong (i.e. connecting foreground and background surfaces) are filtered out.

The meshing process uses Open3D's TriangleMesh class to export the mesh as OBJ files. Moreover, the exported OBJ files contain UV coordinates on each computed triangle. It is therefore possible to texture the exported mesh using an RGB image from the same sensor. 

## ROS Nodes

- **mesh_from_depth_triangle** : When this ROS node receives a depth, RGB and grayscale image message with the same timestamps, it will mesh the depth map and export it as a OBJ file with UV coordinates. It will also create a 4-channel RGBA image using the RGB and grayscale images and export it as a PNG file. The alpha channel (4-th channel) of this exported image contains the grayscale image.
- **mesh_from_depth_simple** : NOT IMPLEMENTED YET

## Parameters

- **~depth_image_topic** : Topic where depth images are being published
- **~depth_intrinsics** : Topic where depth camera intrinsics are being published (sensor_msgs/CameraInfo).
- **~rgb_image_topic** : Topic where RGB images are being published. This parameter is used to create the 4 Channel texture images.
- **~semantic_image** : Topic where grayscale images of the same height & width as the RGB images of the previous parameter get published. These are fused with RGB images to create and export a 4 channel texture map.
- **~export_directory** : Mesh and texture file export directory.
- **~export_success_topic** : Topic where to broadcast Boolean message when a mesh and / or texture file has been successfully exported. 
- **~depth_scale** : Scale correction for depth value messages.
- **~camera_frame_id** : Camera's current coordinate frame ID.
- **~target_frame_id** : When computing a mesh from depth images, the coordinates of the triangle points will be defined with respect to the sensor's optical frame. In case the mesh points need to be defined with respect to another coordinate frame, indicate the frame ID on this parameter. Make sure however that the transformation between **~camera_frame_id** and **~target_frame_id** is exists. 

## Example

See the [demo.launch](launch/demo.launch) launchfile.

## Installation

1. To install, clone this repository into your catkin ```src``` folder with the following command lines:
    ```bash
    cd ~/catkin_ws
    git clone git@github.com:Larsdb98/mesh_from_depth_ros.git
    ```

2. Build and source your workspace:
    ```bash
    cd ~/catkin_ws
    catkin build
    source devel/setup.bash
    ```

3. Launch the node:
    ```bash
    roslaunch mesh_from_depth demo.launch
    ```

# Credit

[mesh_from_depth](https://github.com/hesom/depth_to_mesh/): Code used for the triangulation process. 

# License

This project is licensed under the MIT license - see the [LICENSE.md](LICENSE.md) file for details.
