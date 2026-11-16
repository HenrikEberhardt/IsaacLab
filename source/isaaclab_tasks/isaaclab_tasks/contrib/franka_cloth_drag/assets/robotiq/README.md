# Robotiq 2F-85 on the Franka Panda

`franka_panda_robotiq_coupled_oak.usdc` comes unchanged from Mike Michelis's `isaaclab_soft` assets
(`mmichelis/IsaacLab-Soft`, branch `liftsoft-robotiq`, commit `1ee1ab3`, `src/isaaclab_soft/assets/data/robotiq/`).
His repository's BSD-3-Clause [LICENSE](LICENSE) covers the layers he authored. The source layers, generator scripts
and tests are not included; regenerate the file there.

It is a standalone, flattened file: the standard Isaac Lab `franka_panda.usda` arm with the `mujoco` physics and
`primitives` collider variants selected, with its hand subtree inactive, and on the flange:

- the Robotiq GRP-CPL-062 coupling, a 7 mm camera spacer and the gripper. The gripper's `base_link` is fixed to
  `panda_link7` 125 mm along its z axis with 0° yaw, so the jaws open along the flange's y axis. That yaw is measured
  from real wrist images. Its 19 analytical collision boxes are the fitted thin-pad boxes of `robotiq_minimal.usda`;
- the printed camera mount and the OAK-1 Lite W housing, with the optical camera (calibrated lens, inferred pose) at
  `base_link/OAKHardware/Mount/CameraHousing/OpticalCamera`.

The coupling and the camera each add one invisible collider to `base_link`: a cylinder (radius 37.5 mm) and a box
around the camera and its mount. Their masses (75 g, 30 g, 53 g) are folded into `base_link`, which weighs 768 g.
The cable is visual only and hidden.

`FRANKA_PANDA_ROBOTIQ_CFG` in [../../constrained_gripper/robotiq.py](../../constrained_gripper/robotiq.py) spawns it
through `franka_panda_robotiq_simplified.usda` (below). It carries only the arm actuator; the task assigns the gripper
actuators. The file needs no NVIDIA asset download: the arm and gripper meshes are embedded.

## Simplified colliders

`franka_panda_robotiq_simplified.usda` is the layer the task loads. It sublayers the unchanged `.usdc` and only
changes colliders, so the kinematics, masses and visuals stay those of the `.usdc`. Newton tests every cloth node,
edge and triangle against every shape of its environment, whether or not that shape can reach the cloth, so the
step time grows with the number of shapes. Measured on 2048 environments (random actions):

| Colliders per environment | ms/step | env steps/s |
|---|---|---|
| `.usdc`: 21 gripper + 10 arm, plus 11 visual meshes with a disabled collision schema | 982 | 2090 |
| this layer: 11 gripper | about 650 | about 3150 |

(Same session: 9 colliders with a single backing box 611 ms, these 11 colliders 645 ms; repeat runs vary by about
5 %.)

The simplified mechanism colliders are conservative: they contain every original gripper and coupling collider, so
every mechanism contact of the original colliders also happens with them (extra contacts are accepted). The camera
mount and housing are intentionally visual-only, like the arm in this task. Only the pads, which grip the cloth,
stay exact. Each other box encloses original colliders of one rigid unit, as axis-aligned bounds in that unit's body
frame, grown by 2 mm on every face:

| Box | Encloses | Size [mm] |
|---|---|---|
| `base_link/simplified_collision/base` | the three base boxes and coupling cylinder; camera excluded | 79.0 x 88.9 x 108.0 |
| `<side>_inner_finger/minimal_collision/pad` | itself, unchanged | 22.0 x 6.5 x 38.0 |
| `<side>_inner_finger/simplified_collision/backing_upper` | the wide upper backing box | 31.0 x 31.9 x 23.0 |
| `<side>_inner_finger/simplified_collision/backing_lower` | the two narrow lower backing boxes | 20.8 x 19.8 x 21.9 |
| `<side>_outer_finger/simplified_collision/outer` | the outer finger's two boxes and the outer knuckle's box (fixed joint) | 31.2 x 59.7 x 64.9 |
| `<side>_inner_knuckle/simplified_collision/inner_knuckle` | its box | 39.0 x 59.2 x 63.4 |

The backing boxes are not grown on their jaw-side face, so they stay 1.4 mm behind the pads' gripping plane and the
cloth meets the pads as before. The backing is split in two because one box around all three backing boxes was 31 mm
wide down to 18 mm above the tip, next to the pad: in the reset-bank generator it caused 80 % more gripper-cube
terminations than the original colliders, the split 29 % more. The outer box reaches up to 26 mm further towards the jaws than the outer finger, 40 to
77 mm above the fingertips, next to the inner knuckle. The lowest points move down by the 2 mm margin; over the
stroke they are at least 34 mm (inner knuckles), 38 mm (outer fingers) and 56 mm (base) above the fingertips. The
cube top is 49 mm above the table.

The arm and camera-assembly colliders are deactivated. The arm was never proxied to the cloth solver, the camera does
not participate in task contact, and MuJoCo self-collision is off. The visual meshes lose their (disabled) collision
schemas and keep their mass schemas, so the body masses do not change. Regenerate the boxes the same way if the
`.usdc` changes.

## Pads

Each inner finger has a thin orange pad box that keeps the original inward gripping plane, width (22.0 mm), and
height (38.0 mm), with its closing depth reduced from 19.2 mm to 6.5 mm so it matches the thin lower pad of the real
finger. The remaining finger material is covered by conservative backing boxes at least 1.3 mm behind the gripping
plane, so the fingertip below that only has the thin pad. The boxes are not shape-exact: rearward overfill can
reach 12 mm, so contacts other than the pads' should not be treated as exact. The simplified layer encloses the
backing boxes in two.

## Licences of the embedded content

- NVIDIA's asset terms apply to the embedded Franka and Robotiq meshes.
- The OAK housing mesh is Luxonis's, under the MIT licence in [LUXONIS-LICENSE](LUXONIS-LICENSE).
- The coupling mesh is tessellated from Robotiq's GRP-CPL-062 STEP file. Its redistribution licence is not
  established, and the camera mount STL's author and licence are unknown. Do not assume the repository's licence
  covers either.
