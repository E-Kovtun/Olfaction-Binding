"""Put the repo root on sys.path, the same way every script and notebook does.

`pyproject.toml` sets `package = false` -- this is an application, not an
installed distribution -- so there is no `pip install -e .` to lean on.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub_torch_geometric():
    """Let the graph modules import where torch_geometric is not installed.

    `orbind/gnn_extractor.py` imports it at module scope, so anything that touches the
    extractor -- the source-spec parser, the sweep's planning helpers, the v9 node mix --
    is unimportable on a machine without it, and six tests were failing for that reason
    alone rather than for anything they assert.

    NEVER shadow a real install: on the server the import below succeeds and this does
    nothing. And it lives here rather than at the top of one test module because the
    module that happened to be collected first would otherwise decide whether the rest
    of the suite ran at all.
    """
    try:
        import torch_geometric                          # noqa: F401
        return
    except ImportError:
        pass
    import types

    import torch
    tg = types.ModuleType("torch_geometric")
    nn = types.ModuleType("torch_geometric.nn")
    for cls in ("HeteroConv", "MessagePassing", "SAGEConv"):
        setattr(nn, cls, type(cls, (torch.nn.Module,), {}))
    # marked, so a test that needs a WORKING graph skips instead of failing on the
    # stub's no-op layers. The stub buys importability, not execution.
    tg.__orbind_stub__ = True
    tg.nn = nn
    sys.modules["torch_geometric"] = tg
    sys.modules["torch_geometric.nn"] = nn


_stub_torch_geometric()


def torch_geometric_is_stubbed():
    """True when the graph can be imported but not run. Use as a skipif condition in
    any test that actually trains or forward-passes a `_SignedSage`."""
    import torch_geometric
    return getattr(torch_geometric, "__orbind_stub__", False)
