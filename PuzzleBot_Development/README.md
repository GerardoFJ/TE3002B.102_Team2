# Puzzebot Development

## Sota state
The last modification was the stable version of the traffic lights stop with waypoints
the jetson_nano have multiple problems, the slowness of the processing cause the 
traffic lights to be detected like 3 seconds after, the scripts should be optimized
and changed to c++ for better performance, also have more plots and benchmarks about
the performance of the robot

## How to run
The last state consider this setup on jetson_nano 


```
source ~/ros2_packages_ws/install/local_setup.bash
source ~/ros2_ws/install/local_setup.bash

```
This setup considers the unorganized setup of multiple ws with packages in differents parts


Steps to run the demo

Microros agent (Vel_L, Vel_R, Enc_L, Enc_R)
```
ros2 launch puzzlebot_ros micro_ros_agent.launch.py

```

Odom and cmd_vel generator (cmd_vel, odom)
```
ros2 run cmd_vel_to_wheels cmd_vel_to_wheels --ros-args -p publish_tf:=true

```

Camera publisher (compressed and uncompressed)
```

ros2 launch puzzlebot_ros camera_jetson.launch.py

```

Traffic light detector
```
ros2 run traffic_light detect --ros-args -p confirm_frames:=2

```

Path controller
```
ros2 run square_controller square_controller --ros-args   -p waypoints_x:='[1.67, 1.67, 0.67, 0.67, 0.0] '   -p waypoints_y:='[0.0, 0.95, 0.95, 0.0, 0.0]' -p max_linear_vel:=0.25 -p max_angular_vel:=0.8

```
