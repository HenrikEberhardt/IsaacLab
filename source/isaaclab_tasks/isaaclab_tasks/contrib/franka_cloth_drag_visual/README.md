# Franka cloth drag: visual distillation

DAgger distillation of the state-based cloth-drag teacher (`IsaacContrib-Drag-Cloth-Franka`) into an RGB student.
The student uses the joint-delta deploy stack: 7 relative joint targets at 0.05 rad per action unit, the binary
Robotiq gripper, and the OAK-1 Lite W wrist camera at 160x120.

| Task id | Purpose |
|---|---|
| `IsaacContrib-Drag-Cloth-Franka-Joint` | Joint-delta env without the camera. Used to check the teacher-label conversion. |
| `IsaacContrib-Drag-Cloth-Franka-Visual` | Wrist camera and visual randomization. The DAgger runner trains here. |

## Teacher in another action space

The teacher acts in `(dx, dy, dz, dyaw, gripper)` through a top-down DLS IK. RSL-RL's `Distillation` builds the
teacher with the student's action count, so `agents/distillation.py:IKTeacherDistillation` builds it with its own 5
outputs instead. On every visited state it converts the teacher's deterministic action into an 8-D label
(`teacher/ik_label.py`):

1. The teacher's first-substep joint target offset is computed from the noise-free `ik_state` group (joint
   positions, EE pose, EE Jacobian, joint bounds). This uses the same pose target, DLS step, and speed and range scaling
   as the teacher's arm action.
2. The offset is divided by the student's joint-delta scale and clipped to [-1, 1].
3. The teacher's gripper action passes through unchanged.

The teacher also observes its previous action. In this env that is reconstructed from the executed joint offset
through the EE Jacobian (`teacher_last_action`).

`teacher_cfg.py` pins everything that has to match the teacher checkpoint: its arm action (a frozen copy in
`teacher/ik_action.py`), its actor network, and the terminations it was trained with. Update this file when the
teacher is retrained.

## Checks

```bash
# conversion: converted label vs the IK term's applied first step (expect float precision)
uv run python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_cloth_drag_visual/scripts/teacher_joint_replay.py --mode check --num_envs 32 --num_steps 150 --viz none
# reference success of the teacher with its own IK, and success when it acts through its labels
uv run python .../teacher_joint_replay.py --mode native --num_envs 128 --num_steps 900 --viz none
uv run python .../teacher_joint_replay.py --mode labels --num_envs 128 --num_steps 900 --viz none
# wrist image strip (rows: envs, columns: time) and ms/step of the visual env
uv run python .../save_image_strip.py --num_envs 64 --num_steps 300 --out strip.png --viz none
```

## Distillation

```bash
uv run isaaclab train --task IsaacContrib-Drag-Cloth-Franka-Visual \
    --checkpoint logs/rsl_rl/cloth_drag_task/2026-11-21_15-56-41_one_node_dr/model_750.pt
```

The student is `RslRlCNNModelCfg`. The CNN encodes only the image group. The proprioception group (arm joints,
noisy Robotiq width, last action, target point) is normalized and concatenated with the CNN latent, and an MLP head
maps the result to the 8 actions. Next to the behavior loss, the runner logs `label_clip_fraction` and the arm and
gripper MSE of the student's mean on the rollout.

## Randomization

The physics randomization of the teacher env is kept. Resets are the task-start defaults.

| Requested | Implemented | Why |
|---|---|---|
| Wrist camera position ±1 cm, rotation ±1° | `randomize_camera_offset` per reset | Calibration error of the mount |
| Focal length / FOV | No | The calibrated OpenCV lens cannot take new intrinsics; Joran and Mike don't randomize it either. The Newton renderer traces it with its calibrated fx/fy/cx/cy and distortion, scaled to 160x120 |
| Third-person camera, depth | No | Wrist RGB only |
| Texture vs. color, texture maps | Solid per-episode colors: beech-like table, tinted robot, orange cube, cloth from a blue towel palette (recolor); cube and cloth colors are placeholders until the real ones are fixed | The Warp renderer has no PBR materials. Joran's textured table tops can be ported later |
| Roughness, metallic, specular | No | Not in the Warp renderer |
| HDRI map, rotation, intensity | Shadow-casting directional light re-pointed every 2–5 s (elevation 35–88°) | No HDRI and no light intensity or color in the Warp renderer; the augmentation's exposure and white balance cover intensity and color |
| Color jitter, blur, noise, grayscale | `RgbAugmentation` (Joran's ranges) plus grayscale p = 0.05 | Camera vs. renderer differences |
| Frame lag 0.1 | 2–3 policy steps of camera latency, plus a repeated frame with p = 0.1 | Real camera latency and dropped frames |
| Action lag 0.1 | 0–4 physics steps of joint target delay, plus a kept previous command with p = 0.1 | Real command latency and dropped commands |

The cloth recolor and the light and color events use private renderer handles (`Camera._renderer`, the Newton
sensor's shape-index buffer).
