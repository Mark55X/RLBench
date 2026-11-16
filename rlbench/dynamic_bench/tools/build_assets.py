"""
Builds the object library of the benchmark (rlbench/dynamic_bench/assets/*.ttm) from the models of RLBench's own tasks:
    python -m rlbench.dynamic_bench.tools.build_assets

Each asset is one everyday object as RLBench models it: a respondable dynamic body (a simple convex shape tuned for
Bullet) with its realistic visual mesh as a child. The objects are taken from the task .ttm files, detached, turned into
models and saved one per file, so a benchmark task loads only what it uses.
"""

import argparse
import os

from pyrep import PyRep
from pyrep.const import ObjectType
from pyrep.objects.shape import Shape

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_TTMS = os.path.join(HERE, '..', '..', 'task_ttms')
ASSETS = os.path.join(HERE, '..', 'assets')

# asset name: (source task .ttm, object name in it)
SOURCES = {
    'mug': ('place_cups', 'mug0'),
    'cup': ('pick_up_cup', 'cup1'),
    'wine_bottle': ('stack_wine', 'wine_bottle'),
    'soup': ('put_groceries_in_cupboard', 'soup'),
    'tuna': ('put_groceries_in_cupboard', 'tuna'),
    'spam': ('put_groceries_in_cupboard', 'spam'),
    'sugar': ('put_groceries_in_cupboard', 'sugar'),
    'coffee': ('put_groceries_in_cupboard', 'coffee'),
    'mustard': ('put_groceries_in_cupboard', 'mustard'),
    'crackers': ('put_groceries_in_cupboard', 'crackers'),
    'chocolate_jello': ('put_groceries_in_cupboard', 'chocolate_jello'),
    'strawberry_jello': ('put_groceries_in_cupboard', 'strawberry_jello'),
    'book': ('put_books_on_bookshelf', 'book0'),
    'small_book': ('put_books_on_bookshelf', 'book1'),
    'tunnel': ('dynamic_drema_test_3', 'tunnel_obstacle'),     # the toy tunnel of the first DynamicDrema tests
}


def build(names=None) -> None:
    os.makedirs(ASSETS, exist_ok=True)
    pr = PyRep()
    pr.launch(headless=True)
    try:
        by_source = {}
        for name, (src, obj) in SOURCES.items():
            if names and name not in names:
                continue
            by_source.setdefault(src, []).append((name, obj))
        for src, items in by_source.items():
            base = pr.import_model(os.path.join(TASK_TTMS, src + '.ttm'))
            for name, obj in items:
                shape = Shape(obj)
                shape.set_parent(None, keep_in_place=True)
                shape.set_position([0.0, 0.0, 0.5])
                shape.set_orientation([0.0, 0.0, 0.0]) if name not in ('book', 'small_book') else None
                shape.set_name('asset_' + name)
                # only the body and its meshes: the source task's waypoints and grasp points must not come along
                # (RLBench would take a 'waypoint3' dummy for one of the waypoints of the task the asset is used in)
                for o in shape.get_objects_in_tree(exclude_base=True):
                    if o.get_type() != ObjectType.SHAPE:
                        o.remove()
                shape.set_model(True)
                path = os.path.abspath(os.path.join(ASSETS, name + '.ttm'))
                shape.save_model(path)
                n = len(shape.get_objects_in_tree(object_type=ObjectType.SHAPE))
                print(f'{name:18s} <- {src}/{obj}: mass {shape.get_mass():.3f} kg, {n} child shapes -> {path}')
                shape.remove()
            base.remove()
    finally:
        pr.shutdown()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*', help='assets to build (default: all)')
    build(ap.parse_args().names)
