"""S4VY-lite: a CPU stand-in for S4VY (Segment Anything in Feed-Forward 4D Visual Geometry).

S4VY itself (VGGT-Omega-1B + LoRA + a space-time query decoder) is unreleased and
sized for an A100. This approximates its pipeline with open models that run on a
laptop CPU:
  feed-forward visual geometry  -> Depth Anything 3 (depth + poses from any views)
  exhaustive 4D instance masks  -> SAM 2.1 automatic masks on keyframes, merged
                                   across keyframes by 3D overlap, tracked through
                                   every frame with SAM 2.1 (forward and backward)
  point / box promptable query  -> SAM 2.1 prompt, propagated, lifted to 4D
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "third_party", "sam2"), os.path.join(_ROOT, "third_party", "pydeps")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
