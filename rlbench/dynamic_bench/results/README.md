# Validation results (2026-11-16)

What is verified here is the SIMULATION (does every scene behave as designed?), not a policy. Files:
`verify_2026-11-16.json` (static checks, levels L0-L2 x seeds 0-2, and the static keyframes executed blind, L1-L2
seed 0; `tools/verify.py --parts static blind`) and `verify_solve_2026-11-16.json` (the scripted remedies of
`tools/solve.py`, with the checks of `verify.py --parts solve`). Unit tests: `tests/test_motion.py` 5/5,
`tests/test_sim.py` 40/40.

    python -m rlbench.dynamic_bench.tools.verify --parts static blind --seeds 0 1 2 --out verify.json
    python -m rlbench.dynamic_bench.tools.verify --parts solve --seeds 0 --out verify_solve.json

Video: `../videos/dynbench_L2_static_keyframes_and_remedies.mp4` (12.5 min, English): every task at L2, (A) the
static keyframes executed blind, (B) a scripted remedy along the same keyframes.

## Summary

- Static and blind checks: **688/693**. Every episode is placed, nothing moves before activation, nothing
  floats or explodes, the oracle is complete, the person never touches a still robot, the L1/L2 layouts without
  rejections are those of the static task.
- The obstacle is where it should be in all L0/L1 episodes. At L2 the person found no way to carry something across
  the robot's way in 3 of 9 episodes of the tasks with traffic (close_jar seeds 1-2, stack_blocks seed 2).
- Static keyframes executed blind: the obstacle matters (the blind robot, or what it holds or pushes, touches or
  moves it, touches the person, or fails) in 18/20 runs; in the other two the moving obstacle crosses after the blind
  robot (fast: RLBench's demo speed) has passed (groceries L2, slide_block book across L2). Task success alone hides
  most of the effect: the blind robot often "succeeds" while knocking the obstacle or touching the person.
- Scripted remedies (seed 0): 16/20 pass every check. Failures: close_jar L1 (the lid is not let go
  at the end: a harness issue), close_jar L2 and shape_sorter L2 (the person, careful, does not put the obstacle next
  to the moving robot, so the designed events do not happen; the remedy then has nothing to do).
- Video, L2: the remedy completes 9/10 tasks; in stack_blocks (goal taken) the scripted robot runs into the person
  while she carries a block across, and the task is not completed (the same remedy run alone succeeds: the scripted
  robot is timing-sensitive). The scripted robot is a test harness, not a baseline.
- Layout rejections: in 38 of 90 static episodes RLBench had to sample another layout because the obstacle could
  not be placed or reached in the first one; those episodes differ from the static task's layout of the same seed
  (the static keyframes are always those of the episode actually run).

## Static checks (robot still; levels L0, L1, L2 x seeds 0, 1, 2)

| task | placed | same_layout | still_inactive | obstacle | supported | no_explosion | oracle | idle_safe | episodes with a layout rejection |
|---|---|---|---|---|---|---|---|---|---|
| `dyn_close_jar_item_in_path` | 9/9 | 2/2 | 9/9 | 7/9 | 9/9 | 9/9 | 9/9 | 6/6 | 6/9 |
| `dyn_insert_peg_hand_in_way` | 9/9 | 2/2 | 9/9 | 6/6 | 9/9 | 9/9 | 9/9 | 6/6 | 6/9 |
| `dyn_place_wine_bottle_in_path` | 9/9 | 6/6 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 6/6 | 0/9 |
| `dyn_put_groceries_grasp_blocked` | 9/9 | 2/2 | 9/9 | 6/6 | 9/9 | 9/9 | 9/9 | 6/6 | 6/9 |
| `dyn_shape_sorter_book_on_hole` | 9/9 | 4/4 | 9/9 | 6/6 | 9/9 | 9/9 | 9/9 | 6/6 | 3/9 |
| `dyn_slide_block_book_across` | 9/9 | 6/6 | 9/9 | 6/6 | 9/9 | 9/9 | 9/9 | 6/6 | 0/9 |
| `dyn_slide_block_obstacle_in_path` | 9/9 | - | 9/9 | 6/6 | 9/9 | 9/9 | 9/9 | 6/6 | 9/9 |
| `dyn_slide_block_target_taken` | 9/9 | 2/2 | 9/9 | 6/6 | 9/9 | 9/9 | 9/9 | 6/6 | 6/9 |
| `dyn_stack_blocks_goal_taken` | 9/9 | 6/6 | 9/9 | 8/9 | 9/9 | 9/9 | 9/9 | 6/6 | 0/9 |
| `dyn_stack_blocks_tunnel` | 9/9 | 4/4 | 9/9 | 6/6 | 9/9 | 9/9 | 9/9 | 6/6 | 2/9 |
| **total** | **90/90** | **34/34** | **90/90** | **66/69** | **90/90** | **90/90** | **90/90** | **60/60** | 38/90 |

## Static keyframes executed blind (seed 0)

| task | level | static keyframes reached | task success | obstacle touched / moved | contacts with the person | obstacle matters |
|---|---|---|---|---|---|---|
| `dyn_close_jar_item_in_path` | L1 | 6/6 | yes | yes / 0.06 m | 0 | yes |
| `dyn_close_jar_item_in_path` | L2 | 6/6 | no | no / - m | 0 | yes |
| `dyn_insert_peg_hand_in_way` | L1 | 5/5 | yes | no / - m | 1 | yes |
| `dyn_insert_peg_hand_in_way` | L2 | 5/5 | yes | no / - m | 1 | yes |
| `dyn_place_wine_bottle_in_path` | L1 | 5/5 | yes | yes / - m | 3 | yes |
| `dyn_place_wine_bottle_in_path` | L2 | 5/5 | yes | yes / - m | 1 | yes |
| `dyn_put_groceries_grasp_blocked` | L1 | 5/5 | no | yes / 0.0 m | 0 | yes |
| `dyn_put_groceries_grasp_blocked` | L2 | 5/5 | yes | no / 0.0 m | 0 | no |
| `dyn_shape_sorter_book_on_hole` | L1 | 4/4 | no | yes / 0.034 m | 0 | yes |
| `dyn_shape_sorter_book_on_hole` | L2 | 5/5 | yes | yes / - m | 1 | yes |
| `dyn_slide_block_book_across` | L1 | 1/2 | yes | no / 0.081 m | 0 | yes |
| `dyn_slide_block_book_across` | L2 | 1/2 | yes | no / - m | 0 | no |
| `dyn_slide_block_obstacle_in_path` | L1 | 1/2 | yes | no / 0.071 m | 0 | yes |
| `dyn_slide_block_obstacle_in_path` | L2 | 1/2 | yes | no / 0.021 m | 0 | yes |
| `dyn_slide_block_target_taken` | L1 | 1/2 | yes | no / 0.031 m | 0 | yes |
| `dyn_slide_block_target_taken` | L2 | 1/2 | yes | no / 0.032 m | 0 | yes |
| `dyn_stack_blocks_goal_taken` | L1 | 11/11 | no | yes / 0.03 m | 0 | yes |
| `dyn_stack_blocks_goal_taken` | L2 | 11/11 | no | yes / 0.099 m | 1 | yes |
| `dyn_stack_blocks_tunnel` | L1 | 11/11 | no | yes / 0.112 m | 0 | yes |
| `dyn_stack_blocks_tunnel` | L2 | 11/11 | no | yes / 0.096 m | 0 | yes |

## Scripted remedies (seed 0)

| task | level | events in the designed order | task completed | person never touched | static keyframes reached | side effects |
|---|---|---|---|---|---|---|
| `dyn_close_jar_item_in_path` | L1 | yes | no | yes | 6/6 | - |
| `dyn_insert_peg_hand_in_way` | L1 | yes | yes | yes | 5/5 | - |
| `dyn_place_wine_bottle_in_path` | L1 | yes | yes | yes | 5/5 | - |
| `dyn_put_groceries_grasp_blocked` | L1 | yes | yes | yes | 5/5 | - |
| `dyn_shape_sorter_book_on_hole` | L1 | yes | yes | yes | 5/5 | - |
| `dyn_slide_block_book_across` | L1 | yes | yes | yes | 2/2 | - |
| `dyn_slide_block_obstacle_in_path` | L1 | yes | yes | yes | 2/2 | - |
| `dyn_slide_block_target_taken` | L1 | yes | yes | yes | 2/2 | - |
| `dyn_stack_blocks_goal_taken` | L1 | yes | yes | yes | 11/11 | - |
| `dyn_stack_blocks_tunnel` | L1 | yes | yes | yes | 11/11 | - |
| `dyn_close_jar_item_in_path` | L2 | no (missing ['obstacle_placed']) | no | yes | 6/6 | - |
| `dyn_insert_peg_hand_in_way` | L2 | yes | yes | yes | 5/5 | - |
| `dyn_place_wine_bottle_in_path` | L2 | no (missing ['obstacle_placed', 'obstacle_cleared']) | yes | yes | 5/5 | - |
| `dyn_put_groceries_grasp_blocked` | L2 | yes | yes | yes | 5/5 | - |
| `dyn_shape_sorter_book_on_hole` | L2 | no (missing ['obstacle_placed', 'shape_parked', 'obstacle_cleared']) | yes | yes | 5/5 | - |
| `dyn_slide_block_book_across` | L2 | yes | yes | yes | 2/2 | - |
| `dyn_slide_block_obstacle_in_path` | L2 | yes | yes | yes | 2/2 | - |
| `dyn_slide_block_target_taken` | L2 | yes | yes | yes | 2/2 | - |
| `dyn_stack_blocks_goal_taken` | L2 | yes | yes | yes | 11/11 | - |
| `dyn_stack_blocks_tunnel` | L2 | yes | yes | yes | 11/11 | - |

## Video clips (L2, seed 0)

| task | A: success | A: obstacle touched | A: contacts with the person | B: success | B: contacts with the person |
|---|---|---|---|---|---|
| `dyn_close_jar_item_in_path` | no | no | 0 | yes | 0 |
| `dyn_insert_peg_hand_in_way` | yes | no | 1 | yes | 0 |
| `dyn_place_wine_bottle_in_path` | yes | no | 0 | yes | 0 |
| `dyn_put_groceries_grasp_blocked` | yes | no | 0 | yes | 0 |
| `dyn_shape_sorter_book_on_hole` | yes | no | 0 | yes | 0 |
| `dyn_slide_block_book_across` | yes | no | 1 | yes | 0 |
| `dyn_slide_block_obstacle_in_path` | yes | no | 0 | yes | 0 |
| `dyn_slide_block_target_taken` | yes | no | 0 | yes | 0 |
| `dyn_stack_blocks_goal_taken` | no | yes | 1 | no | 4 |
| `dyn_stack_blocks_tunnel` | no | yes | 0 | yes | 0 |

## Known issues

- CoppeliaSim segfaults now and then (signal 11), more often with several instances in parallel: rerun.
- At L2 the person is careful: she does not put the obstacle next to a moving gripper; a robot that is quick and
  already there does not get the obstacle.
- A robot as fast as RLBench's demos can be past the spot before a moving obstacle arrives.
