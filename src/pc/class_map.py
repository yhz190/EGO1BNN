#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""class_map.py -- the ONLY executable definition of the class encoding.

Single source of truth is src/weights_layout_spec.md §9. This module mirrors it
and provides assert_class_map() so any pipeline stage can fail fast when the training
label order, the vector labels, or an incoming dataset order disagree.

Rationale (incident 2026-11-01): the same mapping was maintained independently in the
protocol doc and in the model code; they drifted on classes 1/2. N10 patch3 fixed the doc
and added this guard.
"""
import os
import csv

# Canonical order -- MUST match 05 spec §9 and train_bnn2.CLASSES
CLASS_MAP = ('rock', 'paper', 'scissors', 'junk')
CLASS_ID = {name: i for i, name in enumerate(CLASS_MAP)}
JUNK_ID = CLASS_ID['junk']


def assert_class_map(class_names=None, vector_labels=None, verbose=False):
    """Fail fast on any class-order mismatch.

    class_names  : optional sequence (e.g. dataset class names / RTL label table)
    vector_labels: optional path to labels.csv (file,class)
    """
    if tuple(CLASS_MAP) != ('rock', 'paper', 'scissors', 'junk'):
        raise AssertionError('CLASS_MAP drifted: %r' % (CLASS_MAP,))

    if class_names is not None:
        names = tuple(str(n).lower() for n in class_names)
        if len(names) < 3 or names[:3] != CLASS_MAP[:3]:
            raise AssertionError(
                'class_names order mismatch: got %r, expected first three %r (see 05 spec §9)'
                % (names, CLASS_MAP[:3]))

    if vector_labels is not None:
        missing = []
        with open(vector_labels, encoding='utf-8-sig') as fh:
            for row in csv.DictReader(fh):
                fn = row['file']
                ci = int(row['class'])
                prefix = fn.split('_')[0].lower()
                expected = CLASS_MAP[ci] if 0 <= ci < len(CLASS_MAP) else None
                if expected is None:
                    raise AssertionError('class id out of range in %s: %s' % (vector_labels, row))
                if prefix != expected:
                    missing.append((fn, ci, prefix, expected))
        if missing:
            raise AssertionError(
                'vector labels disagree with CLASS_MAP (file, class, prefix, expected): %r'
                % (missing[:5],))

    if verbose:
        print('[class_map] OK  %r' % (CLASS_MAP,))
    return True


if __name__ == '__main__':
    import sys
    vec = sys.argv[1] if len(sys.argv) > 1 else None
    assert_class_map(vector_labels=vec, verbose=True)
    print('  CLASS_MAP =', CLASS_MAP)
