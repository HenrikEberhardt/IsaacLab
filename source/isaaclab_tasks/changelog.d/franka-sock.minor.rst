Added
^^^^^

* Added ``IsaacContrib-Lift-Sock-Franka``, a Franka Panda task that lifts a sock off the table. The sock is
  modelled as a surface deformable tube with a closed toe and an open cuff. The task reuses the
  observations, actions, commands, terminations, and curriculum of ``Isaac-Lift-Cloth-Franka``, and
  drops that task's support rails because the tube keeps a graspable cross-section on its own.
* Placed each environment of ``IsaacContrib-Lift-Sock-Franka`` in its own walled 3 m x 3 m room. The
  environment spacing is 3 m, and the four walls run from the floor to 1 m above the table. All four
  walls of one room draw the same NVIDIA Nucleus MDL material, drawn independently per environment
  through heterogeneous cloning, so a room's background stays fixed and consistent for the robot
  training in it.
* Rendered the table, floor, and sock of ``IsaacContrib-Lift-Sock-Franka`` with PBR materials
  (marble, industrial concrete, and cloth), projected in world space on the static surfaces so the
  textures keep a real-world scale. Added a directional light (1500 lux) and a partly cloudy dome
  light (250 lux), both dimmer than an initial pass at this lighting.
* Changed success in ``IsaacContrib-Lift-Sock-Franka`` to show as a green rim around the now visible
  table, which appears only once the goal is reached, instead of tinting the whole table. The default
  camera now frames a close 2 x 2 grid of rooms (``--num_envs 4``) so the sock and gripper on the
  table stay legible.
* Added Kit, Newton GL, and Viser visualizer configurations to ``IsaacContrib-Lift-Sock-Franka``. Kit
  and Newton GL run headless and render only the first 16 environments; Viser serves on port 8750.
  They launch whenever ``--viz`` is omitted, so pass ``--viz none`` to train without a visualizer.
* Made the sock of ``IsaacContrib-Lift-Sock-Franka`` drape like fabric: the bending stiffness
  (``edge_ke``) was lowered from 30 to 2 and the mesh resolution raised from 20 to 28 segments.
  ``SOCK_RESTING_HEIGHT`` was updated to the flatter settled shape, and the lift clearance was raised
  from 0.05 m to 0.10 m.
* Added ``IsaacContrib-Lift-Sock-Franka-WristCam`` and ``IsaacContrib-Lift-Sock-Franka-HeadCam``. Each is
  ``IsaacContrib-Lift-Sock-Franka`` plus one 128 x 128 RGB and depth camera for viewing and recording.
  The wrist camera models a RealSense D405 on the Franka hand and follows the hand every step. The head
  camera models a RealSense D455 fixed 0.8 m above the robot base and pitched down toward the table.
  Both variants keep the state task's observations and agent, so its checkpoints still load.
* Added a second Viser server on port 8751 to both camera variants. It streams the camera's RGB and
  depth images. ``--video`` still records the viewport by default; pass ``env.video_recorders=rgb`` or
  ``env.video_recorders=depth`` to record the camera instead.
