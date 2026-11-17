# Dynamic benchmark on RLBench (`rlbench/dynamic_bench`)

Obstacles that come up AFTER a keyframe policy has decided.

A keyframe policy (BridgeVLA++, PerAct, RVT) looks at the scene and decides the next gripper poses; a lower layer
moves the robot between them. This benchmark tests that lower layer. Every task is an RLBench task BridgeVLA++ is
trained on, unchanged (scene, variations, success condition, instructions). Its **static keyframes** are those of the
static demo of the same episode (PerAct keypoint discovery, `keyframes.py`): what a keyframe policy decides a priori,
on the scene before anything changes. Then a person seated at the table changes the scene: she puts something in the
robot's way between two keyframes or on the target of a keyframe, or carries things across the robot's way. **The
keyframes stay valid**: the obstacle has to be dealt with between them (wait, go over or around, push it away, pick
and move it, park what the gripper holds and clear the way), not by deciding new keyframes.

No learned policy is needed to run the benchmark: the static keyframes come with every episode.

## Principles

- **Obstacles, not new goals.** Nothing the person does moves the goal of the task or undoes its progress; the static
  keyframes always remain a correct plan once the obstacle is out of the way. (A goal that moves is a problem for the
  high-level policy, outside this benchmark.)
- **Serious obstacles, several ways out.** An obstacle blocks the static way for real (the blind static keyframes run
  into it, see `tools/verify.py`), and each one can be handled in more than one way, which depend on what it is: a box
  too wide for the gripper can only be pushed, a bottle tips over if pushed high, a hand cannot be touched at all.
- **Contextual obstacles.** What the person puts in the way belongs to the scene of the task: another block of the
  same game, a kitchen item next to the jars, her own wine bottle by the rack, another item of the same shopping, her
  book on the toy, her mug on the desk, her hand.
- **Physics.** The person is the only agent. Every object is a free rigid body (realistic RLBench meshes and masses)
  that moves only because the person, the robot or gravity moves it. Nothing floats, teleports or appears from
  nowhere: the person takes things from where they stand (her own things stand on the table in front of her, in view,
  out of the robot's way) and, if she gives up a job, puts them back.
- **Moving obstacles for the predictor.** At L2 the obstacle arrives while the robot is executing (triggered by the
  robot's progress), and the person also carries things across the robot's way and back. Trajectories are NOT an
  input of the robot (it has to predict them) but are recorded as ground truth.
- **Oracle for what is a free body.** The benchmark says the dynamics class of every body (free or not; a later lab
  setup will have to infer it).

## Tasks

| # | task | RLBench task | obstacle | static keyframes it blocks | ways out |
|---|---|---|---|---|---|
| 1 | `dyn_stack_blocks_goal_taken` | `stack_blocks` | a block of another colour (a distractor of the task) on the stacking spot | place keyframes of the first block | push it off (holding the block, or after parking it); park, pick it away, re-grasp; L2: unstack the block the person puts on the parked one |
| 2 | `dyn_stack_blocks_tunnel` | `stack_blocks` | a toy tunnel (the one of the first DynamicDrema tests) slid by the person along the table, ending up over a block to grasp (only the blocks to stack have the task's colour: no other block can replace it) | above the block -> grasp | slide it off along its opening pushing the roof (the block stays); lift it off; pushed sideways it drags the block |
| 3 | `dyn_close_jar_item_in_path` | `close_jar` | a tall kitchen item (crackers box, mustard bottle, sugar box) in front of the jar | lid lifted -> above the jar | over or around it; push it low (high: it tips over); park the lid, pick it (mustard by the cap, sugar across its thin side; crackers too wide) |
| 4 | `dyn_shape_sorter_tablet_on_hole` | `place_shape_in_shape_sorter` | the person's tablet lying flat over the target hole (too wide to grasp, too thin to be pushed by its edge) | above the hole -> release | **drag** it off: press on it from above with the closed gripper and move (park the shape first); L2: wait for her hand to leave it |
| 5 | `dyn_place_wine_bottle_in_path` | `place_wine_at_rack_location` | the person's own bottle (green) standing in front of the rack location | in front of the location -> in the location | stand the own bottle up (lying it rolls), move hers by the neck, re-grasp; push hers low at the base (high: it tips and rolls) |
| 6 | `dyn_put_groceries_grasp_blocked` | `put_groceries_in_cupboard` | another grocery item of the scene against the target, where a finger of the grasp comes down (on its top for a tall target) | above the item -> grasp | pick it away or push it; drag the target out of the gap |
| 7 | `dyn_slide_block_obstacle_in_path` | `slide_block_to_color_target` | the person's mug or book in the way of the pushed block | behind the block -> pushed onto the target | stop, take it away (pick the mug, push the book), resume; push the block around it; pushing on shoves it along (a mug tips over) |
| 8 | `dyn_shape_sorter_box_over_shape` | `place_shape_in_shape_sorter` | the toy box of the game upside down over the shape to grasp (too wide to grasp, closed on every side) | above the shape -> grasp | **topple** it: push it high on one side, following its edge as it tips about the opposite bottom edge (slid or pushed low it drags the shape along); then grasp |
| 9 | `dyn_slide_block_board_pinned` | `slide_block_to_color_target` | a long board across the block's way, one end held down by the person's book (too wide to grasp) | behind the block -> pushed onto the target | **rotate** it about the end under the book (push its free end along an arc) until it lies beside the way; slid lengthwise it drags the book |
| 10 | `dyn_insert_peg_hand_in_way` | `insert_onto_square_peg` | the person's hand resting on the top of the target spoke | above the spoke -> release | **wait** until it has gone (it must not be touched); L2: predict it coming back |

What each task needs, beyond following the keyframes: pushing (2, 7, 1), picking and placing elsewhere (1, 5, 6, 7),
going around (3), waiting / predicting (10, and the L2 of every task), dragging (4), toppling (8), rotating (9). Every
task leaves other ways open (the column "ways out"): the benchmark measures the outcome and the side effects, not
which primitive was used. In tasks 4, 8 and 9 the simpler ways are physically ruled out (too wide to grasp, too thin to
push by the edge, closed on every side, held down at one end); in task 4 the sorter's plastic has a friction of 0.3 (RLBench leaves it at a
rubber-like 0.71 / 0.5, on which nothing pressed from above can be dragged with the Panda's fingers, 0.5).

`tools/describe.py` prints the table and the level notes from the classes. The seven RLBench tasks used are all in the
BridgeVLA++ training set (18 RLBench tasks); some carry two or three different situations.

## Levels

| level | what happens |
|---|---|
| `off` | the original static task |
| `L0` | **static obstacle**: there from the start, it does not move (the static keyframes still ignore it) |
| `L1` | **obstacle after the decision**: right after the start the person puts it in the way (her own things she already holds, just picked up from her side of the table; the task's own objects she takes from where they stand; peak hand speed 1-1.1 m/s); it is at rest when the robot gets there |
| `L2` | **moving obstacles**: the obstacle arrives while the robot is executing (triggered by the robot's progress, 1 m/s), and the person carries things across the robot's way and back (tasks 1, 3, 5); task 1: a block stacked on the parked one; task 2: the tunnel covers a later grasp while the robot carries the first block; task 4: her hand stays on the tablet 2.5 s; task 8: the box comes down while the gripper approaches; task 9: the board slides towards the block while the robot comes; task 10: the hand fidgets, leaves and comes back |

Set it with `set_level('L1')` or the environment variable `DYNBENCH_LEVEL`.

## Speed of the person

The person moves at absolute, realistic speeds (peak hand speed about 1 m/s, as people reach; the slides 0.2-0.3
m/s), whatever the robot: the world does not slow down for a slow robot, so results stay comparable across robots and
speed limits (the logic of ISO/TS 15066, which fixes the human speed and asks the robot to keep its distance). For
reference, RLBench's demos move the joints at up to 0.8-1 rad/s, the gripper tip at 0.2-0.6 m/s peak.

For ablations, `set_human_speed_scale(k)` (or `DYNBENCH_HUMAN_SPEED=k`; DynamicDrema: `--dyn_human_speed k`) scales
every speed of the person by k (reaches, slides, fidgeting; not how long she waits or holds). Every episode reports the
actual speeds in `metrics['speeds']`: the scale, the peak and mean (while moving) speed of her hand and of the gripper
tip, and their ratios person / robot (`ratio_peak`, `ratio_mean`).

## The static keyframes (`keyframes.py`)

```python
from rlbench.dynamic_bench.keyframes import static_episode
descriptions, obs, demo, keyframes = static_episode(task_env, seed=0)
# keyframes[k]: 'position', 'quaternion' (x y z w, Panda_tip frame), 'gripper_open', 'ignore_collisions',
#               'action' (the PerAct vector [x y z qx qy qz qw open ignore_collisions]), 'index' (demo step)
```

The demo is recorded on the static scene of the seed (the person and the items out of the scene, the task objects
where the static task puts them); the episode is then reset to that demo, with the person and the obstacles of the
level. The task is told the keyframes (`dynbench_set_keyframes`) and reports the progress along them
(`metrics['keyframes']`, `keyframe_reached` events). DynamicDrema's `--keyframes demo` uses exactly these keyframes.

## The person

- CoppeliaSim's seated human model ("Sitting Bill", not from RLBench) on a dining chair at the side of the table closer
  to what the task needs, scaled 1.2; reach about 45 cm into the table, more leaning forward (up to 35 degrees).
- Arm: shoulder - elbow - palm in closed form, minimum-jerk reaching, the hand held level. Two grips: from the top
  (palm on the object) and by the side (bottles, tall boxes); a carried object is kinematic (`carried`); when the hand
  opens it becomes a free body once the hand is 1 cm away.
- Her things stand on the table in front of her, to her side; the obstacle is taken from there (or, for the task's own
  objects, from where it stands), and put back there if she gives up a job. At rest her hands are in her lap.
- Travel: up out of the lap, across 10 cm above the tallest thing under the way (plus what the hand carries), down.
  The arm, the carried object and the upper body are checked along the way and at the end: if they would hit
  something (the robot included), the hand passes higher, or she waits and finally gives up.
- Social behaviour: she does not move her hand towards a moving gripper (15 cm keep-out; 7 cm for a still one: people
  work next to a still robot), stops when a moving gripper comes within 9 cm of the hand or forearm (4 cm for a still
  one), flinches back when touched and retries after 2 s (3 attempts).

## Physics of the objects

- Free rigid bodies (Bullet), realistic masses, inertia from the collision shape, a little angular damping.
- Cans and boxes collide as a shape fitted to their mesh (a 24-sided prism, a pure box): RLBench's convex meshes have
  a rounded base on which a standing can rocks by itself in Bullet. Mug, bottles and cup keep RLBench's convex shapes.
- A solid floor under the table (RLBench's floor is not respondable).
- The cylinder of `place_shape_in_shape_sorter` rolls away by itself in the ORIGINAL static task (4 of 8 seeds): the
  verification does not count it.

## Oracle and ground truth

- `task.dynbench_object_info()`: every body the robot may meet, with its dynamics class (`agent`: the person;
  `carried`: what she holds; `passive`: free bodies; `static`: fixed parts; `held`: what the robot holds) and the
  handles of its shapes; `dynbench_dynamics_class(handle)`.
- `task.dynbench_ground_truth()`: per body, every active step: time, position, quaternion, class;
  `oracle.prediction_error(gt[name], times, predicted)` scores a predicted trajectory (ADE / FDE).
- `task.dynbench_agent_plan(horizon, dt)`: the palm positions the person's script is about to produce.
- `dynbench_info()` / this README describe how the person moves; the robot is expected to infer it from observation.

## Metrics (`task.dynbench_metrics()`)

- success: the static task's own condition (unchanged); `keyframes`: how many static keyframes were reached, in order.
- the obstacle: `obstacle`, `obstacle_placed` / `t_obstacle_placed`, `obstacle_in_way_at_end`, `obstacle_cleared` /
  `t_obstacle_cleared`, `obstacle_touched_by_robot`, `obstacle_grasped_by_robot`, `obstacle_tilt_deg`,
  `traffic_jobs`; task-specific extras (`block_parked`, `shape_parked`, `mode`, ...).
- safety: `collided` / `person_contact_events` (the robot, or what it holds, touched the person or the chair),
  `min_clearance_person`, `contact_log`; side effects: `toppled_by_robot` (an object the robot touched without
  grasping it tilted by more than 45 degrees), `fell_off_by_robot`; `unsafe` = contact with the person or a side
  effect.
- the person: seat, flinches, jobs (status, reason), `bumped_objects`.
- speeds: `speeds` (the person's hand and the gripper tip, peak and mean, their ratio; the human speed scale).

## Use

```python
from rlbench.dynamic_bench import get_task_class, set_level
from rlbench.dynamic_bench.keyframes import static_episode
set_level('L1')
task_env = env.get_task(get_task_class('dyn_close_jar_item_in_path'))
descriptions, obs, demo, keyframes = static_episode(task_env, seed=0)
task_env._task.dynbench_set_active(True)            # optional: the person acts only while the episode is active
...                                                 # follow the keyframes with the system under test
metrics = task_env._task.dynbench_metrics()
```

With DynamicDrema:

```bash
python run_coppelia_client.py --server_address localhost:29651 --task dyn_close_jar_item_in_path --dyn_level L1 \
    --keyframes demo --sync_mode stepped --headless --force_scan --scan_steps 25 --seed 0 \
    --roles_config configs/roles_env_dynbench.yaml
```

## Tools

| tool | what it does |
|---|---|
| `tools/verify.py` | verifies the simulation (not a policy): placement, same layout as the static task, nothing moves before activation, the obstacle is there and in the way, nothing floats, no physics explosion, oracle complete, the person never touches a still robot; the blind static keyframes run into the obstacle; the scripted remedies unfold as designed |
| `tools/solve.py` + `tools/scripted.py` | scripted remedies with privileged state that follow the same static keyframes (a test harness, not a baseline) |
| `tools/make_video.py` | one video of all the tasks, in English: per task (A) the static keyframes executed blind, (B) a remedy, with the keyframes drawn on the frames |
| `tools/evaluate_replay.py` | the blind baseline over many episodes: the static demo replayed open loop |
| `tools/inspect_task.py` | contact sheets of a task |
| `tools/describe.py` | the task table from the classes |
| `tools/build_assets.py` | extracts the object library from RLBench's task models into `assets/*.ttm` (already built) |
| `tests/test_motion.py`, `tests/test_sim.py` | unit tests; simulator tests (items, person: reach, carry, release, put back, flinch; monitor; activation; demos; static keyframes) |

## Validation

See `results/README.md` for the latest numbers, how they were obtained, and the video.

## Adding a task

1. A class `DynX(DynamicBenchTask, X)` in `tasks/dyn_<name>.py`, X a task of the keyframe policy's training set, with
   `BASE_TASK`, `CATEGORY`, `OBSTACLE`, `KEYFRAME_SEGMENT`, `SUMMARY`, `REMEDIES`, `LEVEL_NOTES`.
2. `build_scene()`: the person's own things (`self.add_item(kind, name)`; kinds in `assets.SPECS`), if the obstacle is
   not one of the task's objects (`self.task_object(shape)`).
3. `setup_episode(rng)`: from the task's waypoints (the static keyframes), the obstacle's pose; seat the person
   (`self.person.choose_seat`), check every placement (`self.item_fits`, `self.person.arm_fits`; raise
   `BoundaryError` for another layout), `self.reset_obstacle(obstacle, kind)`, her things at home
   (`self.place_home`), the L0 obstacle in place.
4. `obstacle_in_way()`: whether the obstacle blocks the static keyframes now.
5. `update(t)`: triggers: L1 right after the start, L2 on the robot's progress (`self.start_obstacle(self.job_move_item
   / job_slide / job_hand_rest(...))`, `self.start_traffic`, `self.finish_traffic`).
6. Register it in `__init__.py`, add a scripted remedy in `tools/solve.py`, its events in `tools/verify.py`, a card in
   `tools/make_video.py`; run `tools/verify.py`.

## Limits

- The person is kinematic: realistic motion and contacts with objects, no compliance (a pushed hand does not give way;
  it flinches back). The hand is a single rigid shape held level; the mesh shows an open hand.
- She does not turn what she carries (an object keeps its yaw from where it stood to where she puts it).
- RLBench grasps are rigid attachments (no slip).
- Physics outcomes (where a pushed bottle lands) depend on Bullet; deterministic for a seed in practice.
- The dynamics class goes to the DREMA server only as `static` / not static (the proto has no agent class).
- Some layouts are rejected when the obstacle cannot be placed or reached (`layout_rejections`): those episodes differ
  from the static task's layout of the same seed.
