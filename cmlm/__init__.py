"""
cmlm -- a context mapping language model: what a frontier model gets
instead of a context window.

A CMLM starts empty, is bound to the rules of the frontier model it serves,
and fills by TRAINING as the person works. The frontier model keeps no
transcript. It reaches the CMLM through two tools: `recall`, which asks the
CMLM what it holds about something and gets facts back, model to model;
and `teach`, with which the frontier model trains the CMLM on what the turn
established. What survives a turn is in the CMLM's weights and store, and
the CMLM is a file: it is carried between sessions and between frontier
models, and it depends on no model's cache.

    python3 -m cmlm --fake                       # one fake session: real training, real recall, estimated tokens
    python3 -m cmlm --fake --seeds 40            # measured over seeds, both worlds
    python3 -m cmlm --fake --file me.cmlm        # run twice: the second run loads and keeps training
    python3 -m cmlm --inspect me.cmlm            # what the file holds
    python3 -m cmlm --file me.cmlm               # live, on stdin: unmeasured here, no key

The limit on what it can hold is training, not tokens. What the frontier
model did not teach is gone; the record shows what came back and what did
not, per probe, against a truth the CMLM cannot see. See CMLM.md.

Apache-2.0. Depends on numpy. The SDK import is lazy. Live runs are priced
with chorus.sample when it is installed, and left unpriced otherwise.
"""
__version__ = "0.1.0"
from .model import CMLM, features, words
from .frontier import Session, Turn, RULES, RECALL, TEACH, PatternTeacher, est_tokens
