# UFACTORY xArm6 Description (MJCF)

Requires MuJoCo 2.3.3 or later.

## Overview

This package contains a MuJoCo description of the [xArm6](https://www.ufactory.cc/#/en/xarm) collaborative robot arm manufactured by [UFACTORY](https://www.ufactory.cc/). The model is derived from the open-source ROS description shipped in [`xarm_ros/xarm_description`](https://github.com/xArm-Developer/xarm_ros/tree/master/xarm_description).

## URDF → MJCF derivation steps

1. Used `xacro` to instantiate `xarm_description/urdf/xarm_device.urdf.xacro` with `dof:=6`,
   producing an intermediate URDF that contains the base robot only.
2. Copied the visual meshes from `xarm_ros/xarm_description/meshes/xarm6/visual` and the
   flange collision mesh from `xarm_ros/xarm_description/meshes/end_tool/collision` into
   this folder's `assets/` directory.
3. Loaded the URDF with MuJoCo's Python bindings and exported the resulting MJCF as a
   starting point.
4. Rebuilt the hierarchy by hand to clean up defaults, add materials, introduce
   controllers, and expose a tool attachment site.
5. Re-used the visual meshes as simplified collision geometry (the original URDF does not
   ship dedicated collision meshes for all links).
6. Added `scene.xml` which instantiates the arm together with a textured plane, skybox,
   and tuned visualization defaults.

## License

The upstream ROS description is distributed under the BSD-3-Clause License shown in
[`../../../../xarm_ros/LICENSE`](../../../../xarm_ros/LICENSE). The MJCF assets in this
folder inherit the same license. A copy is provided in [`LICENSE`](LICENSE).
