# Franka cloth drag: visual distillation

DAgger distillation of the state-based cloth-drag teacher (`IsaacContrib-Drag-Cloth-Franka`) into a wrist-camera
student with the joint-delta deploy stack: 7 relative joint targets (0.05 rad per action unit), the binary Robotiq
gripper and the OAK-1 Lite W wrist camera at 160x120.

| Task id | Purpose |
|---|---|
| `IsaacContrib-Drag-Cloth-Franka-Joint` | Student action space without the camera; label conversion checks |
| `IsaacContrib-Drag-Cloth-Franka-Visual` | Wrist camera, visual randomization and the DAgger runner |

## Layout

| Path | What it does |
|---|---|
| `franka_cloth_drag_visual_env_cfg.py` | Puts it together: teacher task + adaptive resets + student actions and inputs + camera + randomization + action mapping (`ACTION_MAPPING_CFG`) |
| `teacher/` | Everything pinned to the teacher checkpoint: checkpoint path, frozen IK arm action (`ik_action.py`), actor network and terminations (`teacher_cfg.py`), `load_teacher` for scripts |
| `student/` | What is deployed: OAK camera (`camera.py`), joint-delta actions with latency and drops (`actions.py`), proprio and image groups (`observations.py`), network (`model.py`), runner config (`agent_cfg.py`), LEAPP export detection (`export.py`) |
| `deploy/` | Robot-side tools, no `__init__.py` (never imported by `isaaclab_tasks`): `send_target.py` publishes the cloth target |
| `action_mapping/` | Teacher action space to student action space: interface (`base.py`), identity, IK to joint delta (`ik_to_joint_delta.py`, including the observations it needs), and the RSL-RL algorithm that applies it (`distillation.py`) |
| `domain_randomization/` | All camera-side randomization: mount, light, colors, cloth recolor, photometric augmentation; all values in `visual_randomization_cfg.py` |
| `tests/` | `test_action_mapping.py` (pytest, no sim), `check_label_conversion.py`, `save_image_strip.py` and `eval_student.py` (sim) |

## Teacher

Pinned in `teacher/teacher_cfg.py`: run `2026-11-22_18-53-59_gentle_contact_adaptive` (commit `1fb4d0c93` with adaptive
resets), `model_999.pt`. Its arm action is the task's `TopDownDeltaEEAction`: scale 0.015 m / 0.2 rad, an EE target
floor at 0.0193 m, no joint-speed cap and no joint-range scaling, and a 0–4 physics-step delay.

The teacher is trained with `AvoidTableCollisionCurriculum`. Once the success gate opens, its progress p ramps:

- the z step from 0.015 to 0.005 m;
- the yaw step from 0.2 to 0.05 rad (one 0.05 rad joint-delta label on the wrist);
- the joint speed stop from 0.6 to 0.3 of the joint velocity limits (0.65 rad/s on joints 1–4, the real servo limit),
  with its penalty band;
- the gripper force stop from 40 to 15 N (off before).

The distillation env has no curriculum and keeps two kinds of values apart:

- **Teacher arm action** (labels): pinned to the checkpoint's p with `TEACHER_CURRICULUM_PROGRESS`. `TEACHER_YAW_RAMP`
  says whether the checkpoint was trained with the yaw ramp (`model_450`: no, it kept 0.2 rad).
- **Student terminations and penalties**: the curriculum's final values from the start (`apply_final_curriculum_values`:
  15 N force stop, 0.3 joint speed stop), read from the task's `CurriculumCfg`.

When you change the checkpoint, read its p first:

```bash
uv run python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_cloth_drag_visual/tests/read_teacher_curriculum.py \
    --run logs/rsl_rl/cloth_drag_task/2026-11-22_18-53-59_gentle_contact_adaptive --iteration 999
```

## Data flow per step

1. The student acts on `student_proprio` + `student_image` (Gaussian noise, std 0.05, for state coverage).
2. The frozen teacher acts on `teacher`, its unchanged actor group, so the PPO checkpoint loads strictly.
3. The action mapping turns the teacher's deterministic action into a label in the student's action space.
4. After each 64-step rollout the student regresses onto the labels (MSE) and the rollout is discarded.

## Observation groups

| Group | Used by | Contents |
|---|---|---|
| `teacher` | teacher | The task's actor group; `last_action` is the executed student action expressed in the teacher's action space |
| `ik_state` | action mapping | Noise-free joint positions, EE pose, EE Jacobian and joint bounds (added by the IK mapping) |
| `student_proprio` | student | Arm joint pos and vel, noisy Robotiq width, last action, target point; last 5 steps, flattened |
| `student_image` | student | Last 5 wrist RGB frames, uint8, oldest first, `[5, 3, 120, 160]` (group history) |

Both student groups use the observation manager's history (`history_length=5`), which a LEAPP export keeps as graph
state: the robot feeds one frame and one robot state per tick. The image stays in uint8 like the camera. That keeps the
rollout storage at about 4.7 GB at 256 envs (float would be about 19 GB). The student's CNN starts with a layer that
folds the history into 15 channels and divides by 255 (`student/model.py`), so exports take 0–255 images.

## Network

Stock RSL-RL `CNNModel`. The CNN (32-64-64-64 channels, kernels 5-3-3-3, stride 2, flattened) encodes the
15-channel image history. The normalized proprio history is concatenated with the CNN latent, and an MLP
(256-256-128) maps the result to the 8 actions.

## Action mapping

The teacher acts in `(dx, dy, dz, dyaw, gripper)` through a top-down DLS IK; the student acts in joint deltas.
`IKToJointDeltaMapping` makes the label as follows:

1. Compute the teacher's first-substep joint target step from `ik_state`. This uses the same pose target, DLS step,
   and speed and range scaling as the frozen arm action.
2. Divide it by the student's joint-delta scale. Steps beyond [-1, 1] are scaled down uniformly
   (`preserve_direction`). With the old, speed-capped teacher this gave 96.7 % through the labels vs 99.5 % with its
   own IK. With the new, uncapped teacher (`model_100`, p = 0) it gives 41.0 % vs 68.7 %, with 8x more
   `ee_below_table` terminations.
3. Pass the gripper action through unchanged.

The label scale is read from the student's arm action at start (`env.actions.arm_action.scale`), so one override
changes both.

The student's joint target is anchored once per action (`anchor_per_policy_step`): `q0 + scale * a` is set when the
action arrives and held over the 4 physics steps, as the IK teacher's target is. Re-anchoring on the current q every
physics step, as Joran's `RelativeJointPositionAction` does, overshoots the teacher's labels. Teacher `model_150`
(p = 0) through its labels, 128 envs × 900 steps:

| Scale [rad] | Re-anchored | Anchored | Teacher with its own IK |
|---|---|---|---|
| 0.05 | 85.2 % (into the table 3.5 %) | 87.0 % (0.4 %, slow) | 93.5 % (1.1 %) |
| 0.1 | 84.7 % (6.4 %) | 94.3 % (0.3 %) | |
| 0.2 | 81.2 % (11.1 %) | 94.5 % (1.4 %) | |

The runner logs `label_clip_fraction`, `rollout_arm_mse` and `rollout_gripper_mse` next to the behavior loss.

**Same controller in teacher and student (later):** set the student's `actions` to the teacher's arm action and
`ACTION_MAPPING_CFG = IdentityMappingCfg()` in the env cfg. The `ik_state` group and the `teacher_last_action`
override then drop out, because only the IK mapping's `apply_to_env` adds them.

## Resets

Distillation uses the task's `AdaptiveResetsCfg`: the same bank as teacher training
(`logs/reset_banks/bank_v13_one_raised_node.pt`), with start-stage ratios that adapt to the student's success. The bank
check accepts extra sensors such as the wrist camera, so banks made without the camera still load. Play mode uses
task-start resets.

## Randomization

The teacher env's physics randomization is kept.

| Requested | Implemented | Why |
|---|---|---|
| Camera position ±1 cm, rotation ±1° | `randomize_camera_offset` per reset | Calibration error of the mount |
| Focal length / FOV | No | The calibrated OpenCV lens takes no new intrinsics |
| Texture vs. color | Solid per-episode colors: beech-like table, tinted robot, Bambu PETG Basic Orange cube (#FF671F, gloss approximated by a washed-out palette entry), light-green microfiber cloth (recolor of the white-rendered cloth) | No PBR materials in the Warp renderer |
| Roughness, metallic, specular | No | Not in the Warp renderer |
| HDRI, light intensity | Shadow-casting light re-pointed every 2–5 s (elevation 35–88°) | No HDRI or light intensity; exposure and white balance cover it |
| Color jitter, blur, noise, grayscale | `RgbAugmentation` (Joran's ranges) + grayscale p = 0.05, once per frame before stacking | Camera vs. renderer |
| Frame lag | 2–3 policy steps + repeated frame p = 0.1 | Camera latency and dropped frames |
| Action lag | 0–4 physics steps + kept command p = 0.1 | Command latency and drops |

The cloth recolor and the light and color events use private renderer handles (`Camera._renderer`, the Newton
sensor's shape-index buffer).

Play mode (and the LEAPP export, which resolves the task in play mode) uses nominal values instead: no physics
randomization, the nominal camera mount, a fixed light (elevation 70°, not measured), the first palette entry of every
color (green cloth, beech, official cube orange, untinted robot), no augmentation, a 2-step image latency, no dropped
frames or commands, and the gripper's 5-step delay. `--train_env_cfg` plays with the training randomization. The
second Viser server (port 8757) streams the raw wrist render; the cloth shows grey there, only the policy input is
recolored.

## Deploy

The student deploys like Joran's and Mike's policies: LEAPP 0.7.2 export, then Isaac ROS Deploy.

| Graph input | LEAPP kind | Robot source |
|---|---|---|
| `wrist_camera_rgb` | `state/camera/image` | OAK frame, raw distorted, resized to 160x120, rgb8/bgr8 (fed as float32 0–255 HWC) |
| robot joint positions, velocities | state | Arm joints and `finger_joint` (the graph converts the angle to the pad gap) |
| `cloth_target` | `command/body/target_position` | `BodyCommand.pose[0].position` [m, robot base frame], from `deploy/send_target.py` |

The 5-step histories and the last action are graph state. The outputs are the arm joint target
`q + 0.05 * a`, computed once per policy step and held until the next, and the open/close step target of
`finger_joint`. Training adds latency and drops on top; the export leaves them out, since on the robot they are
physical. The cloth-node lock is simulation-only.

```bash
# export (writes the LEAPP yaml, ONNX and initial state next to the checkpoint)
uv run --extra leapp isaaclab leapp export --task IsaacContrib-Drag-Cloth-Franka-Visual \
    --checkpoint logs/rsl_rl/cloth_drag_distillation/<run>/model_999.pt
# replay the exported graph in sim (read -> infer -> write -> hold, as on the robot)
uv run --extra leapp isaaclab leapp deploy --task IsaacContrib-Drag-Cloth-Franka-Visual --pipeline <exported yaml>
# on the robot PC: type targets, or publish one directly
python3 deploy/send_target.py --topic /cloth_target
ros2 topic pub --once /cloth_target isaac_ros_deploy_interfaces/msg/BodyCommand \
    "{names: [cloth_target], pose: [{position: {x: 0.42, y: 0.0, z: 0.002}}]}"
```

Map the inputs to topics with the Isaac ROS Deploy launch argument `source_to_topic`, for example
`cloth_target:/cloth_target`.

## Commands

```bash
# unit tests (no sim)
uv run python -m pytest source/isaaclab_tasks/isaaclab_tasks/contrib/franka_cloth_drag_visual/tests/test_action_mapping.py
# label conversion: float-precision check, teacher reference, teacher through its labels (task-start resets)
uv run python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_cloth_drag_visual/tests/check_label_conversion.py --mode check --num_envs 32 --num_steps 150 --viz none
uv run python .../check_label_conversion.py --mode native --num_envs 128 --num_steps 900 --viz none
uv run python .../check_label_conversion.py --mode labels --num_envs 128 --num_steps 900 --viz none
# newest frame of the image stack (rows: envs, columns: time) and ms/step
uv run python .../save_image_strip.py --num_envs 64 --num_steps 300 --out strip.png --viz none
# student, 100 first episodes per variant: success + gripper forces, width-reading lies, raised-node height (0 = flat)
uv run python .../eval_student.py --checkpoint logs/rsl_rl/cloth_drag_distillation/<run>/model_999.pt \
    --test all --out eval.json --viz none
# distillation
uv run isaaclab train --task IsaacContrib-Drag-Cloth-Franka-Visual \
    --checkpoint logs/rsl_rl/cloth_drag_task/2026-11-21_15-56-41_one_node_dr/model_750.pt
```

A new teacher only needs `teacher/teacher_cfg.py` updated: checkpoint, curriculum progress and arm action.
