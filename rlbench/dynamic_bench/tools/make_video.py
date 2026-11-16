"""
One video of the whole benchmark, in English. For every task: a title card, then
  (A) the STATIC keyframes executed blind: the joint trajectory of the static demo of the same episode, replayed open
      loop (what a keyframe policy decided on the static scene, carried out without looking): every obstacle shows;
  (B) one possible remedy: a scripted robot with privileged state that follows the SAME keyframes and deals with the
      obstacle between them (tools/solve.py).
The static keyframes are drawn on the frames (numbered circles; the next one to reach is highlighted), with captions:
task, level, part, time, the events of the scene as they happen, a CONTACT mark when the robot touches the person.

    # render the clips (one process per task, in parallel), then assemble them
    python -m rlbench.dynamic_bench.tools.make_video render dyn_close_jar_item_in_path --level L2 --seed 0 --out clips/
    python -m rlbench.dynamic_bench.tools.make_video assemble --clips clips/ --out dynbench.mp4
"""

import argparse
import glob
import json
import os
import textwrap

import numpy as np

W, H, FPS = 1280, 720, 20
FOV = 58.0
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

# What the events mean, for the captions
EVENT_TEXT = {
    'obstacle_coming': 'the person brings the obstacle (after the keyframes were decided)',
    'obstacle_placed': 'the obstacle is in place',
    'obstacle_in_way': 'the obstacle has settled in the way',
    'obstacle_not_placed': 'the person could not put the obstacle there',
    'obstacle_cleared': 'the obstacle is out of the way',
    'traffic_crossing': 'the person carries something across the robot\'s way',
    'second_obstacle_coming': 'a second item comes',
    'block_parked': 'the robot parks its block',
    'second_blocker_coming': 'the person stacks a block on the parked one',
    'parked_block_covered': 'the parked block is covered: it must be unstacked',
    'shape_parked': 'the robot parks the shape',
    'keyframe_reached': 'static keyframe reached',
    'person_flinches': 'touched by the robot: the person pulls back',
    'person_waits_for_robot': 'the person waits for the robot to move away',
    'person_pauses': 'the gripper is close: the person stops',
    'person_gives_up': 'the person gives up',
}

# Title card per task: title, what happens, ways to deal with it
CARDS = {
    'dyn_stack_blocks_goal_taken': (
        'Stacking spot taken',
        'The person, playing with the same blocks, puts a block of another colour on the stacking spot: the target '
        'of the place keyframes. At L2 she also carries a block across the robot\'s way, and if the robot parks its '
        'block she stacks another one on it.',
        'push the block off the spot (holding the own block, or after parking it); park, pick the block away, '
        'unstack the parked one, re-grasp; then the same keyframes.'),
    'dyn_stack_blocks_tunnel': (
        'Tunnel slid over the block',
        'A toy tunnel of the same game stands on the table. The person, tidying up, slides it along the table and it '
        'ends up right over a block the robot has to grasp: the block is still there, the grasp from above hits the '
        'roof. At L2 it slides over the second block while the robot carries the first one.',
        'slide the tunnel off along its opening by pushing its roof (the block stays); lift it off by a wall; pushing '
        'it sideways drags the block along.'),
    'dyn_slide_block_target_taken': (
        'Mug on the target',
        'The person puts her mug down on the coloured target square, where the push keyframe brings the block. At L2 '
        'it comes while the block is being pushed.',
        'take the mug away (by the rim) or push it off, then push the block on; pushing on regardless shoves the mug '
        'off the target (it tips over).'),
    'dyn_slide_block_book_across': (
        'Book slid across the push',
        'The person slides her book across the table, through the way the block is pushed along: a moving obstacle '
        'on the table, at the robot\'s height. At L1 it stops in the way; at L2 it goes across and back while the '
        'block is being pushed.',
        'predict it: hold the push, let it pass, push on; if it stops in the way, push it aside or push the block '
        'around it.'),
    'dyn_close_jar_item_in_path': (
        'Kitchen item on the lid\'s way',
        'The person sets a tall kitchen item (crackers box, mustard bottle, sugar box) down in front of the jar, on the '
        'way between the keyframes "lid lifted" and "above the jar". At L2 it moves in while the lid travels, and a '
        'soup can is carried across.',
        'carry the lid over or around it; push it low (pushed high it tips over); park the lid and pick it away '
        '(mustard by the cap, sugar box across its thin side; the crackers box is too wide).'),
    'dyn_shape_sorter_book_on_hole': (
        'Book on the hole',
        'The person lays her book flat on the shape sorter, over the target hole: the release keyframe is now above '
        'the book. At L2 she keeps her hand on it for 4 s.',
        'slide the book off (too wide to grasp), holding the shape or after parking it; at L2 wait for the hand '
        'first.'),
    'dyn_place_wine_bottle_in_path': (
        'Bottle in front of the rack',
        'The person stands her own bottle (green) in front of the rack location, where the robot\'s bottle has to be '
        'at the last two keyframes. At L2 it comes while the robot\'s bottle is on its way, and a glass is carried '
        'across.',
        'stand the own bottle up (lying it rolls), move hers by the neck, re-grasp; or push hers low at the base '
        '(pushed high it tips over and rolls).'),
    'dyn_put_groceries_grasp_blocked': (
        'Grasp blocked by groceries',
        'The person, unpacking the same shopping, puts another grocery item right against the target, where a finger '
        'of the grasp keyframe goes down (or, for a tall target, on its top). At L2 it moves in while the gripper '
        'comes down.',
        'pick the item away or push it (a tall box pushed high tips over); drag the target out of the gap; then grasp '
        'as decided.'),
    'dyn_slide_block_obstacle_in_path': (
        'Obstacle on the push',
        'The person puts her mug or her book in the way of the block the robot pushes to the target. At L2 she slides '
        'the book (or carries the mug) into the way while the block is being pushed.',
        'stop, take the obstacle away (pick the mug, push the book aside), resume; or push the block around it; '
        'pushing on regardless shoves it along (a mug tips over).'),
    'dyn_insert_peg_hand_in_way': (
        'Hand next to the spoke',
        'The person rests her hand on the board next to the target spoke, where the ring has to come down. A hand '
        'is not an object: it cannot be pushed and must not be touched. At L2 it fidgets, leaves and comes back.',
        'wait until the hand has gone; at L0 (the hand stays) insert carefully from above, clear of it.'),
}

LEVELS_TEXT = [
    ('off', 'the original static RLBench task (no person, no obstacle).'),
    ('L0', 'static obstacle: it is there from the start and does not move.'),
    ('L1', 'obstacle after the decision: right after the start the person puts it in the way (her own things she '
           'already holds); it is at rest when the robot gets there.'),
    ('L2', 'moving obstacles: the obstacle arrives while the robot is on its way to it (triggered by the robot\'s '
           'progress), and the person also carries things across the robot\'s way.'),
]

PART_TEXT = {'A': '(A) static keyframes, executed blind', 'B': '(B) one possible remedy (scripted, same keyframes)'}


def _font(size, bold=False):
    from PIL import ImageFont
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


# ---------------------------------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------------------------------

def camera_for(task):
    """Eye and look-at: across the table from the person (she faces the camera), the robot at the side."""
    side = task.person.side if not task.person.hidden else -1
    if task.BASE_TASK == 'put_groceries_in_cupboard':
        return (0.05, -1.30 * side, 1.45), (0.30, 0.02 * side, 0.92)
    return (0.25, -1.30 * side, 1.60), (0.30, 0.02 * side, 0.84)


def project(cam_matrix, points):
    """Pixel coordinates of world points in the snapshot camera (None for points behind it)."""
    f = 0.5 * W / np.tan(np.radians(FOV) / 2)
    inv = np.linalg.inv(cam_matrix)
    out = []
    for p in points:
        c = inv @ np.r_[p, 1.0]
        if c[2] <= 0.05:
            out.append(None)
            continue
        out.append((0.5 * W - f * c[0] / c[2], 0.5 * H - f * c[1] / c[2]))
    return out


def _overlay(img, ctx, t, events, contact, kf_px, kf_next):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img).convert('RGBA')
    ov = Image.new('RGBA', im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    # static keyframes
    for k, uv in enumerate(kf_px):
        if uv is None:
            continue
        u, v = uv
        done, nxt = k < kf_next, k == kf_next
        r = 13 if nxt else 10
        colour = (120, 120, 120, 200) if done else ((255, 200, 0, 255) if nxt else (0, 200, 255, 230))
        d.ellipse([u - r, v - r, u + r, v + r], outline=colour, width=3)
        d.text((u + r + 2, v - r - 4), str(k + 1), font=_font(18, True), fill=colour)
    d.rectangle([0, 0, W, 60], fill=(0, 0, 0, 165))
    d.text((18, 6), f'{ctx["index"]}/{ctx["total"]}  {CARDS[ctx["name"]][0]}   -   {PART_TEXT[ctx["part"]]}',
           font=_font(24, True), fill=(255, 255, 255, 255))
    d.text((18, 37), f'{ctx["name"]}   (RLBench task: {ctx["base"]})   level {ctx["level"]}   '
                     f'static keyframe {min(kf_next + 1, len(kf_px))}/{len(kf_px)}', font=_font(15),
           fill=(210, 210, 210, 255))
    d.text((W - 150, 12), f't = {t:5.1f} s', font=_font(24, True), fill=(255, 255, 255, 255))
    if events:
        h = 30 * len(events) + 16
        d.rectangle([0, H - h, W, H], fill=(0, 0, 0, 150))
        for k, (te, text, age) in enumerate(events):
            alpha = 255 if age < 3.0 else 170
            colour = (255, 215, 90, alpha) if age < 3.0 else (220, 220, 220, alpha)
            d.text((18, H - h + 8 + 30 * k), f'{te:5.1f} s   {text}', font=_font(22, age < 3.0), fill=colour)
    if contact:
        d.rectangle([W - 210, 72, W - 18, 114], fill=(200, 30, 30, 220))
        d.text((W - 190, 78), 'CONTACT', font=_font(26, True), fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(im, ov).convert('RGB'))


def _card(lines, sub=None):
    from PIL import Image, ImageDraw
    im = Image.new('RGB', (W, H), (18, 24, 33))
    d = ImageDraw.Draw(im)
    y = 110
    for text, size, bold, colour in lines:
        for line in textwrap.wrap(text, width=int(1950 / size)):
            d.text((80, y), line, font=_font(size, bold), fill=colour)
            y += int(size * 1.32)
        y += int(size * 0.55)
    if sub:
        d.text((80, H - 60), sub, font=_font(17), fill=(150, 160, 175))
    return np.asarray(im)


def _event_caption(e):
    text = EVENT_TEXT.get(e['event'])
    if text is None:
        return None
    if e['event'] == 'keyframe_reached':
        return f'static keyframe {e["index"] + 1}/{e["of"]} reached'
    if e['event'] in ('obstacle_coming', 'obstacle_placed') and e.get('obstacle'):
        return text + f' ({str(e["obstacle"]).replace("_", " ")})'
    return text


def render_part(te, task, demo, kfs, part, ctx, clip_dir, tail, seed):
    from PIL import Image
    from rlbench.dynamic_bench.tools.common import SnapshotCamera
    descriptions, _ = te.reset_to_demo(demo)
    task.dynbench_set_keyframes(kfs)
    eye, at = camera_for(task)
    cam = SnapshotCamera(eye, at, resolution=(W, H), fov_deg=FOV)
    cam_m = cam.sensor.get_matrix()
    kf_pts = [np.asarray(k['position']) for k in kfs]
    frames = []
    ctx = dict(ctx, part=part)

    def shoot():
        t = task.t
        evs = []
        for e in task.event_log:
            text = _event_caption(e)
            if text is not None:
                evs.append((e['t'], text, t - e['t']))
        img = _overlay(cam.capture(), ctx, t, evs[-3:], task.monitor._in_contact, project(cam_m, kf_pts),
                       task.kf_next)
        path = os.path.join(clip_dir, f'{part}{len(frames):05d}.jpg')
        Image.fromarray(img).save(path, quality=90)
        frames.append(path)

    orig = task.step

    def step():
        orig()
        shoot()
    task.step = step
    q0 = np.concatenate([task.robot.arm.get_joint_positions(), [1.0]])
    for _ in range(int(1.0 / task._dt) if part == 'A' else 0):    # (A) the scene before the robot moves
        te.step(q0)
    success = False
    if part == 'A':
        arm = task.robot.arm
        for obs in demo:
            action = np.concatenate([obs.joint_positions, [1.0 if obs.gripper_open > 0.5 else 0.0]])
            for _ in range(4):
                te.step(action)
                if np.max(np.abs(np.asarray(arm.get_joint_positions()) - obs.joint_positions)) < 0.01:
                    break
            success = success or bool(task.success()[0])
        hold = np.concatenate([arm.get_joint_positions(), [1.0 if demo[-1].gripper_open > 0.5 else 0.0]])
    else:
        from rlbench.dynamic_bench.tools.scripted import ScriptedRobot, Timeout
        from rlbench.dynamic_bench.tools.solve import SOLVERS
        rb = ScriptedRobot(te, max_steps=3000)
        try:
            SOLVERS[ctx['name']](rb, task, kfs, np.random.default_rng(seed))
        except Timeout:
            pass
        success = rb.success
        hold = np.concatenate([task.robot.arm.get_joint_positions(), [rb.open]])
    for _ in range(int(tail / task._dt)):
        te.step(hold)
        success = success or bool(task.success()[0])
    del task.step
    cam.remove()
    m = task.dynbench_metrics()
    return {'frames': len(frames), 'success': bool(success), 'contacts': int(m['person_contact_events']),
            'unsafe': bool(m['unsafe']), 'obstacle_touched': bool(m.get('obstacle_touched_by_robot', False)),
            'toppled': m['toppled_by_robot'], 'keyframes': m['keyframes'],
            'obstacle_cleared': bool(m.get('obstacle_cleared')), 'description': descriptions[0],
            'events': [e['event'] for e in task.event_log]}


def render(name, level, seed, out_dir, tail):
    from rlbench.dynamic_bench import TASKS, get_task_class, set_level
    from rlbench.dynamic_bench.keyframes import keyframes_from_demo, record_static_demo
    from rlbench.dynamic_bench.tools.common import make_env
    clip_dir = os.path.join(out_dir, name)
    os.makedirs(clip_dir, exist_ok=True)
    for f in glob.glob(os.path.join(clip_dir, '*.jpg')):
        os.remove(f)
    env = make_env()
    set_level(level)
    te = env.get_task(get_task_class(name))
    task = te._task
    np.random.seed(seed)
    te.reset()
    np.random.seed(seed)
    demo = record_static_demo(te)
    kfs = keyframes_from_demo(demo)
    ctx = {'name': name, 'base': task.BASE_TASK, 'level': level, 'index': list(TASKS).index(name) + 1,
           'total': len(TASKS)}
    info = {'name': name, 'level': level, 'seed': seed, 'keyframes': kfs}
    for part in ('A', 'B'):
        info[part] = render_part(te, task, demo, kfs, part, ctx, clip_dir, tail, seed)
        print(part, json.dumps({k: v for k, v in info[part].items() if k != 'events'}), flush=True)
    info['description'] = info['A']['description']
    with open(os.path.join(clip_dir, 'clip.json'), 'w') as f:
        json.dump(info, f, indent=1)
    env.shutdown()


# ---------------------------------------------------------------------------------------------------------------------
# assembling
# ---------------------------------------------------------------------------------------------------------------------

def assemble(clips_dir, out, card_seconds):
    import imageio
    from PIL import Image
    from rlbench.dynamic_bench import TASKS, get_task_class
    writer = imageio.get_writer(out, fps=FPS, codec='libx264', quality=8, pixelformat='yuv420p', macro_block_size=8)
    n_card = int(card_seconds * FPS)
    infos = {}
    for name in TASKS:
        clip = os.path.join(clips_dir, name, 'clip.json')
        if os.path.exists(clip):
            infos[name] = json.load(open(clip))
    level = next(iter(infos.values()))['level'] if infos else 'L2'
    intro = _card([
        ('Dynamic benchmark on RLBench', 46, True, (255, 255, 255)),
        ('Obstacles that come up AFTER a keyframe policy has decided.', 28, True, (255, 215, 90)),
        ('The tasks are RLBench tasks BridgeVLA++ is trained on, unchanged. Their static keyframes are those of the '
         'static demo of the same episode (PerAct keypoint discovery): what a keyframe policy decides a priori, on the '
         'scene before anything changes. Then a person seated at the table puts things in the robot\'s way: between '
         'two keyframes, on a keyframe target, or carried across the way. The keyframes stay valid; what is tested is '
         'the layer that moves the robot between them.', 23, False, (210, 215, 225)),
        ('The person (CoppeliaSim\'s seated human model) is the only agent. Every object obeys physics: it moves only '
         'when the person, the robot or gravity moves it. Obstacles belong to the scene of the task (another block of '
         'the game, a kitchen item, the person\'s own bottle or book, another grocery item, her hand).', 21, False,
         (190, 200, 215))], sub='rlbench/dynamic_bench')
    levels = _card([('Levels', 44, True, (255, 255, 255))] +
                   [(f'{k}:  {v}', 25, k == level, (255, 215, 90) if k == level else (215, 220, 230))
                    for k, v in LEVELS_TEXT] +
                   [(f'This video: level {level}, real simulation speed. For every task:', 24, True, (255, 255, 255)),
                    ('(A) the static keyframes executed blind: the static demo\'s joint trajectory replayed open loop, '
                     'as planned on the static scene. Numbered circles: the static keyframes (yellow: the next one).',
                     22, False, (210, 215, 225)),
                    ('(B) one possible remedy: a scripted robot with privileged state follows the SAME keyframes and '
                     'deals with the obstacle between them. It shows the scenario can be solved; it is not a '
                     'baseline.', 22, False, (210, 215, 225))])
    for card in (intro, levels):
        for _ in range(int(1.8 * n_card)):
            writer.append_data(card)
    summary = []
    for name in TASKS:
        if name not in infos:
            print(f'missing clip: {name}')
            continue
        info = infos[name]
        cls = get_task_class(name)
        title, what, ways = CARDS[name]
        i = list(TASKS).index(name) + 1
        card = _card([(f'{i}/{len(TASKS)}  {title}', 42, True, (255, 255, 255)),
                      (f'{name}   (RLBench task: {cls.BASE_TASK}, level {info["level"]}, '
                       f'{len(info["keyframes"])} static keyframes)', 19, False, (150, 170, 200)),
                      (what, 26, False, (225, 228, 235)),
                      ('Ways to deal with it: ' + ways, 24, True, (255, 215, 90)),
                      (f'Level {info["level"]}: ' + cls.LEVEL_NOTES.get(info['level'], ''), 20, False,
                       (190, 200, 215)),
                      ('Instruction (as in the static task): "' + info['description'] + '"', 19, False,
                       (170, 180, 195))], sub=f'{cls.CATEGORY}  -  keyframes: {cls.KEYFRAME_SEGMENT}')
        for _ in range(n_card):
            writer.append_data(card)
        for part in ('A', 'B'):
            files = sorted(glob.glob(os.path.join(clips_dir, name, f'{part}*.jpg')))
            for f in files:
                writer.append_data(np.asarray(Image.open(f)))
        summary.append((i, title, info))

    def yn(b):
        return 'yes' if b else 'no'
    lines = [(f'Summary (level {level})', 40, True, (255, 255, 255)),
             ('A = static keyframes blind: success / obstacle touched by the robot / contacts with the person.   '
              'B = scripted remedy: success / contacts.', 17, False, (160, 170, 185))]
    for i, title, info in summary:
        a, b = info['A'], info['B']
        lines.append((f'{i}. {title}:  A {yn(a["success"])} / {yn(a["obstacle_touched"])} / {a["contacts"]}    '
                      f'B {yn(b["success"])} / {b["contacts"]}', 19, False, (220, 225, 232)))
    end = _card(lines, sub='Scripted remedies check that the scenarios unfold and can be solved; they are not a '
                           'baseline.')
    for _ in range(int(1.8 * n_card)):
        writer.append_data(end)
    writer.close()
    print('written', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('render')
    r.add_argument('task')
    r.add_argument('--level', default='L2')
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--tail', type=float, default=3.0)
    r.add_argument('--out', default='dynbench_clips')
    a = sub.add_parser('assemble')
    a.add_argument('--clips', default='dynbench_clips')
    a.add_argument('--out', default='dynbench.mp4')
    a.add_argument('--card_seconds', type=float, default=6.0)
    args = ap.parse_args()
    if args.cmd == 'render':
        render(args.task, args.level, args.seed, args.out, args.tail)
    else:
        assemble(args.clips, args.out, args.card_seconds)


if __name__ == '__main__':
    main()
