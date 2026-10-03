"""
ocean.py
========

Continuous Big Five (OCEAN) personality decision model for the ride-hailing
digital twin.  Every agent carries a personality profile, and every decision the
agent makes is a documented function of that profile and the current context -
computed at runtime, for any profile, with no per-type configuration to maintain.

Everything here is deterministic given a seed, every number is traceable to an
explicit formula, and every table can be dumped as data - the module exists to
produce reproducible, auditable test cases for the twin.


THREE LAYERS OF PERSONALITY
---------------------------
1. **Domains** - the five OCEAN scores, each in ``[0, 1]``:
   ``openness``, ``conscientiousness``, ``extraversion``, ``agreeableness``,
   ``neuroticism``.
2. **Facets** - the fifteen BFI-2 facets (three per domain), each in
   ``[0, 1]``.  A facet that is not supplied defaults to its parent domain, so a
   domain-only profile behaves exactly like a profile whose facets all equal
   their domain.
3. **Derived operational traits** - the behavioural quantities the simulator
   actually reacts to (``patience``, ``surge_greediness``, ``driving_style``,
   ...).  Each trait is an auditable linear form of the *facets*::

       trait = clip(0.5 + sum(coefficient * (facet - 0.5)), 0.0, 1.0)

   The coefficients and a one-line justification for each trait live in
   ``PASSENGER_TRAIT_MODEL`` / ``DRIVER_TRAIT_MODEL`` / ``SHARED_TRAIT_MODEL``.
   Weights are always built from traits, never from raw domains.


TWO EVALUATION MODES (never mixed)
----------------------------------
**Mode A - categorical action choice** (``start``, ``wait_requests``,
``request``, ``pickup``)::

    raw[a]  = base[a] + sum(weights[a][factor][value] for factor in context)
    raw[a]  = max(raw[a], 0.001)
    prob[a] = raw[a] / sum(raw.values())

``base`` sums to 1.0, the weights are signed deltas that sum to ~0 across the
actions of a phase (so context shifts *preference*, never total mass), and the
returned probabilities sum to 1.0.  Mode A draws no randomness.

**Mode B - 1-5 star rating with abstention** (``ride``)::

    p_abstain = clip(no_vote_base + sum(no_vote_weights[factor][value]), 0, 1)
    rating    = clip(rate_base    + sum(rate_weights[factor][value]), 1, 5)

Exactly **one** uniform draw is taken per rate call, whatever the outcome, so
seeded runs stay aligned across scenario variants.  The rating agent's *weights*
depend only on its own personality; the counterpart shifts only the two scalar
bases (``rate_base``, ``no_vote_base``).


QUICK START
-----------
::

    import random
    rng = random.Random(42)
    ocean = OceanDecisionModel("job_personalities_big5.csv", rng=rng)
    driver = ocean.new_driver_view()
    passenger = ocean.new_passenger_view()

    context = {
        "traffic_level": "mid", "surge_multiplier": "low", "route_length": "mid",
        "time_estimated_pickup": "low", "time_estimated_ride": "mid",
        "time_of_day": "afternoon", "passenger_rating": "4-5",
        "driver_age": "31-40", "zone_requested": "midtown",
    }

    probabilities = driver.request(context)                       # Mode A
    action = driver.choose("request", context, rng)               # sampled
    rating, abstained = passenger.rate_driver(driver, context, rng)  # Mode B

Constructing the model:  ``OceanDecisionModel(jobs_csv_path,
drivers_job_code="0", passengers_job_code="0", students_csv_path=None,
rng=None)``.  Job code ``"0"`` is the general population distribution,
``"students"`` the optional students distribution, any other code the specific
job with that ``Code`` in the CSV; the three are distinct concepts.
``sample_profile(actor, job_code)`` returns a plain dict (20 personality values plus
``job``/``code``), ``get_job_name(job_code)`` the human-readable job name.

The constructor codes are the *baseline*.  Personalities can be injected at
runtime like any other scenario: ``set_job_codes(drivers_job_code,
passengers_job_code)`` switches what ``new_driver_view`` / ``new_passenger_view``
sample from, ``reset_job_codes()`` restores the baseline.  Only agents created
after the switch are affected. A profile is fixed for an agent's lifetime.


CONTEXT DICTS
-------------
Keys are optional; a missing factor simply contributes nothing.  Unknown keys
are ignored.  A *known* factor carrying an unrecognised value raises
``ValueError`` - silent fallback would hide bugs in generated test cases.  The
full vocabulary (also available as ``CONTEXT_VALUES``):

===========================  =================================================
``traffic_level``            low / mid / high
``surge_multiplier``         low / mid / high
``route_length``             low / mid / high
``time_waiting_pass``        low / mid / high
``time_waiting_driv``        low / mid / high
``time_waiting_pickup``      low / mid / high
``time_estimated_pickup``    low / mid / high
``time_estimated_ride``      low / mid / high
``provider``                 uber / lyft
``time_of_day``              morning / afternoon / night
``passenger_rating``         1-2 / 2-3 / 3-4 / 4-5
``driver_rating``            1-2 / 2-3 / 3-4 / 4-5
``shift_time``               low / mid / high
``passenger_age``            16-24 / 25-34 / 35-44 / 45-54 / 55-64
``driver_age``               18-30 / 31-40 / 41-50 / 51-60 / 61-70 / 71+
``zone_requested``           downtown / midtown / suburbs
===========================  =================================================

The two age vocabularies are deliberately different (5 passenger buckets, 6
driver buckets including ``71+``) and are never interchanged.  Which factors are
live in which phase is declared in ``PHASE_FACTORS``; the actions of each Mode A
phase in ``PHASE_ACTIONS``.


CLASS-BASED API
---------------
``DriverView(profile)`` / ``PassengerView(profile)`` accept an ``OceanProfile``
or a plain dict, and also build from JSON records (``from_json``).  Each view
derives its traits and precomputes every
personality-only weight table **once**; only context-dependent arithmetic happens
per call.

============================================  ====================================
``driver.start(ctx)``                         Mode A: change_zone / start_work
``driver.wait_requests(ctx)``                 Mode A: stay_and_wait / change_zone /
                                              change_provider / stop_work
``driver.request(ctx)``                       Mode A: accept / reject
``driver.pickup(ctx)``                        Mode A: accept / reject
``driver.rate_passenger(pax, ctx, rng)``      Mode B -> (rating|None, abstained)
``passenger.request(ctx)``                    Mode A: accept /
                                              reject_and_change_provider / reject
``passenger.pickup(ctx)``                     Mode A: accept / reject
``passenger.rate_driver(drv, ctx, rng)``      Mode B -> (rating|None, abstained)
``view.choose(phase, ctx, rng)``              one sampled action from Mode A
``view.tables(phase)``                        deep copy of {"base", "weights"}
``view.traits``                               read-only trait mapping (also dotted)
``view.notes``                                one-line behavioural rationale
``view.profile``                              the ``OceanProfile``
============================================  ====================================

No view mutates a caller-supplied context or profile dict.


INSPECTING THE TABLES
---------------------
``view.tables(phase)`` returns a deep copy of that phase's
``{"base": ..., "weights": ...}`` (Mode A) or
``{"rate_weights": ..., "no_vote_weights": ...}`` (Mode B), so every number a
decision rests on is inspectable as data - which is what makes a generated test
case auditable.  ``view.traits`` exposes the derived operational traits and
``trait_rationale(role, trait)`` the one-line justification for each.
``resolve_ride`` resolves a Mode B pairing deterministically, with no draw.

An earlier revision also carried a JSON-shaped compatibility layer
(``DotDict``, ``PersonalityView``, ``build_personality_config``) and a 16-type
MBTI bridge, whose only purpose was field-by-field diffing against the legacy
MBTI JSON configuration.  Both were removed once that comparison was no longer
needed: nothing in the simulator imported them, and the discrimination probe
they served is now the 33 ``ARCHETYPE_PROFILES`` corners of the personality cube.


REPRODUCIBILITY
---------------
Seed one ``random.Random`` and pass it to ``OceanDecisionModel(..., rng=rng)``
(and to ``choose`` / ``rate_*``): the same sequence of calls then produces
identical results.  Trait derivation, weight tables, ``notes`` and
``provider_affinity`` are pure functions of the profile; nothing that depends on
context or on a draw is cached; there is no global mutable personality state.
Run the self-test with ``python ocean.py``.


ASSUMPTIONS AND LIMITATIONS
---------------------------
* **Source scale.**  Values in the jobs CSV are read as T-scores
  (``TSCORE_SCALE``, population mean 50 / SD 10) and the population and students
  distributions as 1-5 Likert (``LIKERT_SCALE``); each draw is clipped to its
  source scale and then mapped **linearly** to ``[0, 1]``.  Both scales are
  module constants, so changing them is a one-line edit.  Set
  ``TSCORE_PERCENTILE_MAPPING = True`` to recover the older normal-CDF
  (percentile) mapping for T-scores instead.
* The CSVs carry no facet structure, so facets are sampled around their parent
  domain with a jitter SD derived from that domain's own SD to hit
  ``FACET_DOMAIN_CORRELATION`` (0.55, inside the 0.43-0.67 BFI-2 reports for real
  within-domain facet intercorrelations).
* Trait coefficient magnitudes, weight magnitudes and rating bands are *design
  choices* informed by the Big Five literature and by ride-hailing rating
  distributions.  They are not fitted values; only the **signs** are treated as
  part of the specification (and asserted by the self-test).
* ``provider_affinity`` is a deterministic per-profile brand lean derived from
  the profile hash; it encodes "this personality consistently prefers one
  brand", not any real-world claim about Uber or Lyft.
* Factors act additively: there are no explicit factor x factor interactions.
  Age/zone signals are modulated by the agent's traits, not by other factors.
* **Where the variance lives.**  Real people differ a lot from one another and
  are fairly consistent with themselves.  ``PERSON_GAIN`` amplifies each agent's
  base shares away from the population-neutral agent so that variance sits in
  stable person-level differences rather than in the per-decision draw, and
  ``MACRO_FACTOR_DAMP`` damps the factors that move the whole fleet at once
  (``FLEET_SHARED_FACTORS``).  Both were introduced after measuring the earlier
  model: between-agent SD of ``P(accept)`` was 0.05-0.10, the swing across every
  context combination was 0.23, and the simulated aggregate count series had an
  index of dispersion of 1.19.  After the change those are 0.09-0.20 and 0.65 -
  more person-level signal *and* a smoother aggregate, because ``sum(p*(1-p))``
  falls as probabilities move away from 0.5.  The two gains differ by role
  because driver-side and passenger-side leverage are not symmetric; see the
  comment on ``PERSON_GAIN``.
* **Ratings are whole stars.**  ``INTEGER_STARS`` rounds the latent on the way
  out, because platforms only accept whole stars, and the star base follows a
  skewed curve (``_skewed_rate_base``) fitted to the published Uber/Lyft shape.
  The rating itself is deterministic given rater, counterpart and context - the
  single rng draw decides only whether the agent votes at all.  That is the
  faithful reading: real rating variation comes from real variation in rides,
  which the context already carries, not from a coin flip at the keypad.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import os
import random
import statistics
import tempfile
import warnings
from collections.abc import Mapping as _ABCMapping
from dataclasses import dataclass
from statistics import NormalDist
from typing import (Any, Callable, Dict, Iterator, List, Optional, Sequence,
                    Tuple, Union)

# Seed the global random generator for reproducibility of the self-test and any
# random.seed(os.getenv("RANDOM_SEED", 42))


# ==========================================================================
# 1. CONTEXT VALUE SETS AND PHASE STRUCTURE
# ==========================================================================

def clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp ``value`` into ``[lower, upper]``."""
    return max(lower, min(upper, value))


def _spread(values: Sequence[str]) -> Dict[str, float]:
    """Map an ordered list of bucket labels to evenly spaced numbers in
    [-1, 1], e.g. age brackets from youngest to oldest -> -1.0 .. 1.0."""
    n = len(values)
    if n == 1:
        return {values[0]: 0.0}
    step = 2.0 / (n - 1)
    return {v: -1.0 + i * step for i, v in enumerate(values)}


LEVELS = ["low", "mid", "high"]
PROVIDERS = ["uber", "lyft"]
TIMES_OF_DAY = ["morning", "afternoon", "night"]
RATING_BUCKETS = ["1-2", "2-3", "3-4", "4-5"]
ZONES = ["downtown", "midtown", "suburbs"]
# The two age vocabularies differ by role and are never interchanged.
PASSENGER_AGE_BRACKETS = ["16-24", "25-34", "35-44", "45-54", "55-64"]
DRIVER_AGE_BRACKETS = ["18-30", "31-40", "41-50", "51-60", "61-70", "71+"]

#: The complete context vocabulary.  Every weight table is built by iterating
#: this mapping, so coverage is structural rather than hand-written.
CONTEXT_VALUES: Dict[str, List[str]] = {
    "traffic_level": list(LEVELS),
    "surge_multiplier": list(LEVELS),
    "route_length": list(LEVELS),
    "time_waiting_pass": list(LEVELS),
    "time_waiting_driv": list(LEVELS),
    "time_waiting_pickup": list(LEVELS),
    "time_estimated_pickup": list(LEVELS),
    "time_estimated_ride": list(LEVELS),
    "provider": list(PROVIDERS),
    "time_of_day": list(TIMES_OF_DAY),
    "passenger_rating": list(RATING_BUCKETS),
    "driver_rating": list(RATING_BUCKETS),
    "shift_time": list(LEVELS),
    "passenger_age": list(PASSENGER_AGE_BRACKETS),
    "driver_age": list(DRIVER_AGE_BRACKETS),
    "zone_requested": list(ZONES),
}

PASSENGER_AGE_SPREAD = _spread(PASSENGER_AGE_BRACKETS)
DRIVER_AGE_SPREAD = _spread(DRIVER_AGE_BRACKETS)
RATING_BUCKET_SPREAD = {"1-2": -1.0, "2-3": -0.4, "3-4": 0.3, "4-5": 1.0}

#: Evaluation mode of every (role, phase) pair: "A" = categorical choice,
#: "B" = rating with abstention.
PHASE_MODE: Dict[Tuple[str, str], str] = {
    ("driver", "start"): "A",
    ("driver", "wait_requests"): "A",
    ("driver", "request"): "A",
    ("driver", "pickup"): "A",
    ("driver", "ride"): "B",
    ("passenger", "request"): "A",
    ("passenger", "pickup"): "A",
    ("passenger", "ride"): "B",
}

#: Actions of every Mode A phase (Mode B phases have none).
PHASE_ACTIONS: Dict[Tuple[str, str], List[str]] = {
    ("driver", "start"): ["change_zone", "start_work"],
    ("driver", "wait_requests"): ["stay_and_wait", "change_zone", "change_provider", "stop_work"],
    ("driver", "request"): ["accept", "reject"],
    ("driver", "pickup"): ["accept", "reject"],
    ("driver", "ride"): [],
    ("passenger", "request"): ["accept", "reject_and_change_provider", "reject"],
    ("passenger", "pickup"): ["accept", "reject"],
    ("passenger", "ride"): [],
}

#: The factors that carry signal in each phase.  Weights are non-zero for
#: exactly these factors, and cover every value of each.
PHASE_FACTORS: Dict[Tuple[str, str], List[str]] = {
    ("driver", "start"): [
        "traffic_level", "surge_multiplier", "provider", "time_of_day",
        "driver_age", "zone_requested",
    ],
    ("driver", "wait_requests"): [
        "traffic_level", "surge_multiplier", "time_waiting_driv", "provider",
        "time_of_day", "shift_time", "driver_age", "zone_requested",
    ],
    ("driver", "request"): [
        "traffic_level", "surge_multiplier", "route_length",
        "time_estimated_pickup", "time_estimated_ride", "time_of_day",
        "passenger_rating", "driver_age", "zone_requested",
    ],
    ("driver", "pickup"): [
        "traffic_level", "surge_multiplier", "route_length",
        "time_estimated_pickup", "time_estimated_ride", "time_of_day",
        "passenger_rating", "driver_age", "zone_requested",
    ],
    ("driver", "ride"): [
        "traffic_level", "provider", "time_of_day", "shift_time",
        "driver_age", "zone_requested",
    ],
    ("passenger", "request"): [
        "surge_multiplier", "route_length", "time_waiting_pass",
        "time_estimated_ride", "provider", "time_of_day", "passenger_age",
        "zone_requested",
    ],
    ("passenger", "pickup"): [
        "surge_multiplier", "route_length", "time_waiting_pickup",
        "time_estimated_pickup", "time_estimated_ride", "time_of_day",
        "driver_rating", "passenger_age", "zone_requested",
    ],
    ("passenger", "ride"): [
        "provider", "time_of_day", "passenger_age", "zone_requested",
    ],
}

MODE_A_PHASES: Dict[str, List[str]] = {
    "driver": ["start", "wait_requests", "request", "pickup"],
    "passenger": ["request", "pickup"],
}
RIDE_TABLE_KEYS = ["rate_weights", "no_vote_weights"]

# --- magnitude floors -----------------------------------------------------
# No weight is ever exactly 0.0.  A near-neutral Mode A effect collapses onto
# WEIGHT_FLOOR with the sign of the trait that motivates it; complementary
# shares are floored too, so every action keeps a non-zero entry.
# The floor applies to the *source* delta of a factor.  The complementary
# shares are strictly proportional to it, so an action taking a small slice of a
# floored delta can hold a weight below the floor - never zero, and never at the
# cost of complementarity (invariant 6), which is the stronger requirement.
WEIGHT_FLOOR = 0.008          # smallest signed Mode A source delta
SHARE_FLOOR = 0.10            # smallest slice of complementary mass
RATING_WEIGHT_FLOOR = 0.02    # smallest signed Mode B rate weight
ABSTAIN_WEIGHT_FLOOR = 0.004  # smallest signed Mode B no-vote weight
NEUTRAL_MID = 0.15            # direction given to a "mid" level
NEUTRAL_DOMAIN = 0.5          # fallback for a domain a record never mentions

# --- behavioural dispersion -----------------------------------------------
# Real people differ a lot from each other and are fairly consistent with
# themselves: between-person variance is large, within-person variance small.
# A decision model gets that backwards if every agent's action probability sits
# near the population mean, because then almost all of the realised variation is
# the per-decision coin flip.  Measured on the pre-change model, the between-agent
# SD of P(accept) at a fixed context was 0.05-0.10 and the total swing across
# *every* context combination was 0.23 - so agents were nearly interchangeable and
# the aggregate series was pure counting noise.
#
# PERSON_GAIN amplifies each agent's base shares away from the population-neutral
# agent (see _widen_base).  Raising it moves variance from the per-decision draw
# into stable person-level differences: the population mean is preserved, the
# spread widens, and because sum(p*(1-p)) falls as probabilities move off 0.5 the
# aggregate count series gets *less* noisy, not more.
#
# The two roles are tuned separately because their leverage is not symmetric: in
# the recorded runs a driver-side personality change moved 48-50 of 56 output
# metrics (mean |rank-biserial r| 0.85) while a passenger-side change moved 4-18
# (0.24-0.44).  The passenger therefore needs the larger gain; giving the driver
# the same one would only deepen a saturation that already hides the passenger.
# Fitted by grid search under two competing constraints: the between-agent SD of
# the leading action must clear 0.075 in every phase, and no action's *median*
# agent may be driven to ~0.  A minority action acquiring a long right tail is
# realistic - plenty of riders essentially never abandon a quote while a
# price-sensitive minority often does - but the median rider must still sometimes
# reject.  Additive widening at a passenger gain of 3.4 failed the second test
# badly (62% of riders below P=0.01, median pinned on the clip floor); the
# multiplicative form at 1.6 leaves the median at 0.054 with the population mean
# unchanged.
PERSON_GAIN: Dict[str, float] = {"driver": 2.0, "passenger": 1.6}
#: Bounds any single widened base share is held inside before normalization.
BASE_SHARE_FLOOR = 0.02
BASE_SHARE_CEIL = 0.96

# --- synchronised context -------------------------------------------------
# These factors take the same value for many agents at the same instant: every
# driver in a zone reads the same traffic level and the same surge, and every
# agent everywhere reads the same time of day.  A reaction to them is therefore a
# *correlated* shock, and correlated shocks are what push an aggregate count
# series past Poisson variance - the measured index of dispersion for
# drivers_change_zone was 3.14, against 0.92-1.25 for every other decision count.
# Damping them shifts expressed variation from "the whole fleet moves together"
# to "these agents differ", which is both less noisy in aggregate and the more
# defensible claim about human behaviour.
FLEET_SHARED_FACTORS = frozenset({"traffic_level", "surge_multiplier", "time_of_day"})
MACRO_FACTOR_DAMP = 0.65

#: A "mid" level is never neutral-zero: it leans, weakly, the same way as the
#: extreme it is closer to in the trait that motivates the factor.
LEVEL_PREFER_LOW = {"low": 1.0, "mid": NEUTRAL_MID, "high": -1.0}
LEVEL_PREFER_HIGH = {"low": -1.0, "mid": -NEUTRAL_MID, "high": 1.0}


# ==========================================================================
# 2. OCEAN INPUT MODEL (domains, facets, profile identity)
# ==========================================================================

DOMAIN_FACETS: Dict[str, Tuple[str, ...]] = {
    "openness": ("intellectual_curiosity", "aesthetic_sensitivity", "creative_imagination"),
    "conscientiousness": ("organization", "productiveness", "responsibility"),
    "extraversion": ("sociability", "assertiveness", "energy_level"),
    "agreeableness": ("compassion", "respectfulness", "trust"),
    "neuroticism": ("anxiety", "depression", "emotional_volatility"),
}

_ALL_DOMAIN_NAMES: List[str] = list(DOMAIN_FACETS.keys())
_ALL_FACET_NAMES: List[str] = [f for facets in DOMAIN_FACETS.values() for f in facets]
_ALL_PROFILE_KEYS: List[str] = _ALL_DOMAIN_NAMES + _ALL_FACET_NAMES
_FACET_PARENT: Dict[str, str] = {
    facet: domain for domain, facets in DOMAIN_FACETS.items() for facet in facets
}


def _as_float(value: Any, key: str) -> float:
    """Coerce a recognized personality value, naming the key on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"personality key '{key}' must be numeric, got {value!r} "
            f"({type(value).__name__})"
        ) from None


@dataclass(frozen=True)
class OceanProfile:
    """
    One agent's personality: five OCEAN domains plus fifteen optional BFI-2
    facets, all in ``[0, 1]`` (values outside the range are clipped).

    Only the domains are required.  A facet left as ``None`` defaults to its
    parent domain's value, so a domain-only profile is well defined.  The
    dataclass is frozen: an agent's personality never changes at runtime.
    """

    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

    intellectual_curiosity: Optional[float] = None
    aesthetic_sensitivity: Optional[float] = None
    creative_imagination: Optional[float] = None
    organization: Optional[float] = None
    productiveness: Optional[float] = None
    responsibility: Optional[float] = None
    sociability: Optional[float] = None
    assertiveness: Optional[float] = None
    energy_level: Optional[float] = None
    compassion: Optional[float] = None
    respectfulness: Optional[float] = None
    trust: Optional[float] = None
    anxiety: Optional[float] = None
    depression: Optional[float] = None
    emotional_volatility: Optional[float] = None

    def __post_init__(self) -> None:
        for domain, facet_names in DOMAIN_FACETS.items():
            domain_value = clip(_as_float(getattr(self, domain), domain), 0.0, 1.0)
            object.__setattr__(self, domain, domain_value)
            for facet_name in facet_names:
                raw = getattr(self, facet_name)
                value = domain_value if raw is None else clip(_as_float(raw, facet_name), 0.0, 1.0)
                object.__setattr__(self, facet_name, value)

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "OceanProfile":
        """
        Build a profile from a plain (JSON-like) record.

        Recognized keys are the five domains and the fifteen facets, in full or
        in part; every other key (``id``, ``job_code``, ``name``, ...) is
        ignored.  A recognized key holding a non-numeric value raises
        ``ValueError`` naming that key.
        """
        if not isinstance(data, dict):
            raise TypeError(f"OceanProfile.from_json expects a dict, got {type(data).__name__}")
        kwargs = {k: v for k, v in data.items() if k in _ALL_PROFILE_KEYS and v is not None}
        for domain in _ALL_DOMAIN_NAMES:
            if domain in kwargs:
                continue
            # A domain omitted but described by its facets is recovered as their
            # mean; a domain described by nothing falls back to the neutral
            # midpoint, loudly, so a mistyped key cannot pass unnoticed.
            supplied = [kwargs[f] for f in DOMAIN_FACETS[domain] if f in kwargs]
            if supplied:
                values = [_as_float(v, domain) for v in supplied]
                kwargs[domain] = sum(values) / len(values)
            else:
                warnings.warn(
                    f"OceanProfile.from_json: no value for domain '{domain}' and none of its "
                    f"facets {DOMAIN_FACETS[domain]}; defaulting to {NEUTRAL_DOMAIN}",
                    stacklevel=2,
                )
                kwargs[domain] = NEUTRAL_DOMAIN
        return cls(**kwargs)

    @classmethod
    def from_facets(cls, facets: Dict[str, float]) -> "OceanProfile":
        """
        Build a profile from facet scores; each domain becomes the mean of its
        three facets (of those supplied).  A domain with no facet supplied
        raises ``ValueError`` naming the domain.
        """
        kwargs: Dict[str, float] = {}
        for domain, names in DOMAIN_FACETS.items():
            supplied = [(n, facets[n]) for n in names if n in facets and facets[n] is not None]
            if not supplied:
                raise ValueError(
                    f"OceanProfile.from_facets: no facet supplied for domain '{domain}' "
                    f"(expected any of {names})"
                )
            values = [_as_float(v, n) for n, v in supplied]
            kwargs[domain] = sum(values) / len(values)
            for name, value in supplied:
                kwargs[name] = value
        return cls(**kwargs)

    # -- inspection --------------------------------------------------------
    def as_dict(self) -> Dict[str, float]:
        """All 20 values (5 domains then 15 facets) as a plain dict."""
        return {key: float(getattr(self, key)) for key in _ALL_PROFILE_KEYS}

    @property
    def profile_id(self) -> str:
        """
        Short, stable, process-independent hash of the rounded 20 values, so
        generated test cases can be labelled and de-duplicated.
        """
        payload = ";".join(f"{key}={getattr(self, key):.4f}" for key in _ALL_PROFILE_KEYS)
        return hashlib.blake2s(payload.encode("utf-8"), digest_size=6).hexdigest()

    def dominant_domains(self, count: int = 2) -> List[str]:
        """Domains furthest from the neutral 0.5, most extreme first."""
        ordered = sorted(
            _ALL_DOMAIN_NAMES,
            key=lambda d: (-abs(getattr(self, d) - 0.5), d),
        )
        return ordered[:count]


ProfileLike = Union[OceanProfile, Dict[str, Any]]


def _coerce_profile(profile_like: ProfileLike) -> OceanProfile:
    """Accept an ``OceanProfile`` or a JSON-like dict; never mutate the input."""
    if isinstance(profile_like, OceanProfile):
        return profile_like
    if isinstance(profile_like, _ABCMapping):
        return OceanProfile.from_json(dict(profile_like))
    raise TypeError(f"Expected OceanProfile or dict, got {type(profile_like)!r}")


# ==========================================================================
# 3. DERIVED OPERATIONAL TRAITS
# ==========================================================================
#
# Derivation contract (section 3.1 of the specification):
#
#     trait = clip(0.5 + sum(coefficient * (facet - 0.5)), 0.0, 1.0)
#
# Each entry below is ``trait -> (justification, {facet: coefficient})``.  The
# justification names the facets used and the direction of each, so the whole
# derivation is inspectable as data.  Traits are derived FIRST; every weight in
# every phase is then built from traits, never from raw domains.

TraitModel = Dict[str, Tuple[str, Dict[str, float]]]

#: Traits derived for both roles (identical formulas, so a profile's
#: reactivity and rating style do not depend on which seat it is in).
SHARED_TRAIT_MODEL: TraitModel = {
    "volatility": (
        "+emotional_volatility +energy_level -organization -responsibility: labile, "
        "high-energy agents swing with context, orderly dutiful ones barely move",
        {"emotional_volatility": 0.55, "energy_level": 0.35,
         "organization": -0.35, "responsibility": -0.30},
    ),
    "rating_discrimination": (
        "+aesthetic_sensitivity +intellectual_curiosity +organization "
        "-respectfulness -compassion: aesthetically sensitive, curious, orderly raters "
        "notice ride quality (cleanliness, smoothness) and spread their stars; "
        "deferential, compassionate ones compress everything toward 5",
        {"aesthetic_sensitivity": 0.55, "intellectual_curiosity": 0.25,
         "organization": 0.20, "respectfulness": -0.20, "compassion": -0.15},
    ),
    "rating_engagement": (
        "+sociability +responsibility +respectfulness -depression -anxiety: sociable, "
        "dutiful agents bother to vote at all; withdrawn or anxious ones stay silent",
        {"sociability": 0.40, "responsibility": 0.35, "respectfulness": 0.25,
         "depression": -0.45, "anxiety": -0.25},
    ),
}

_PASSENGER_ONLY_TRAIT_MODEL: TraitModel = {
    "patience": (
        "+compassion +responsibility -anxiety -emotional_volatility: other-regarding, "
        "dutiful riders absorb waiting; anxious or labile ones read delay as threat",
        {"compassion": 0.55, "responsibility": 0.50,
         "anxiety": -0.45, "emotional_volatility": -0.55},
    ),
    "provider_fidelity": (
        "+organization +trust -intellectual_curiosity -creative_imagination: orderly, "
        "trusting riders keep one app; curious, imaginative ones shop around",
        {"organization": 0.55, "trust": 0.35,
         "intellectual_curiosity": -0.40, "creative_imagination": -0.45},
    ),
    "surge_sensitivity": (
        "+anxiety +emotional_volatility -intellectual_curiosity -assertiveness: anxious "
        "riders feel price spikes acutely; curious, assertive ones shrug and re-plan",
        {"anxiety": 0.60, "emotional_volatility": 0.25,
         "intellectual_curiosity": -0.35, "assertiveness": -0.30},
    ),
    "trust_in_driver": (
        "+trust +compassion -anxiety: trusting, warm riders extend the benefit of the "
        "doubt to a stranger at the wheel; anxious ones do not",
        {"trust": 0.60, "compassion": 0.40, "anxiety": -0.35},
    ),
    "communication_skills": (
        "+sociability +respectfulness +energy_level -anxiety -depression: sociable, "
        "courteous, energetic riders talk easily; anxious or withdrawn ones do not",
        {"sociability": 0.55, "respectfulness": 0.35, "energy_level": 0.20,
         "anxiety": -0.30, "depression": -0.35},
    ),
    "time_budget_flexibility": (
        "+creative_imagination +compassion -organization -anxiety -productiveness: "
        "imaginative, easy-going riders have slack; scheduled, driven, anxious ones do not",
        {"creative_imagination": 0.40, "compassion": 0.30,
         "organization": -0.45, "anxiety": -0.30, "productiveness": -0.25},
    ),
}

_DRIVER_ONLY_TRAIT_MODEL: TraitModel = {
    "patience": (
        "+compassion +responsibility -anxiety -emotional_volatility: warm, dutiful "
        "drivers wait out traffic and slow pickups; labile, anxious ones bail",
        {"compassion": 0.55, "responsibility": 0.50,
         "anxiety": -0.45, "emotional_volatility": -0.55},
    ),
    "provider_fidelity": (
        "+organization +trust -intellectual_curiosity -creative_imagination: orderly, "
        "trusting drivers stay on one platform; curious, inventive ones switch",
        {"organization": 0.55, "trust": 0.35,
         "intellectual_curiosity": -0.40, "creative_imagination": -0.45},
    ),
    "surge_greediness": (
        "+assertiveness +productiveness -compassion -respectfulness: assertive, "
        "output-driven drivers chase multipliers; compassionate, deferential ones will "
        "not exploit a spike",
        {"assertiveness": 0.55, "productiveness": 0.40,
         "compassion": -0.40, "respectfulness": -0.15},
    ),
    "work_urgency": (
        "+productiveness +responsibility -depression: driven, dutiful drivers want to be "
        "working; depressive ones disengage",
        {"productiveness": 0.55, "responsibility": 0.45, "depression": -0.35},
    ),
    "trust_in_passenger": (
        "+trust +compassion -anxiety: trusting, warm drivers accept a stranger at face "
        "value (and a mediocre rating); anxious ones screen hard",
        {"trust": 0.60, "compassion": 0.40, "anxiety": -0.35},
    ),
    "communication_skills": (
        "+sociability +respectfulness +energy_level -anxiety -depression: sociable, "
        "courteous, energetic drivers manage the cabin; anxious or flat ones do not",
        {"sociability": 0.55, "respectfulness": 0.35, "energy_level": 0.20,
         "anxiety": -0.30, "depression": -0.35},
    ),
    "zone_flexibility": (
        "+creative_imagination +energy_level +intellectual_curiosity -organization: "
        "imaginative, energetic, curious drivers reposition; routine-bound ones stay",
        {"creative_imagination": 0.45, "energy_level": 0.40,
         "intellectual_curiosity": 0.20, "organization": -0.45},
    ),
    "driving_style": (
        "aggressiveness: +assertiveness +emotional_volatility -respectfulness "
        "-responsibility: pushy, labile drivers cut it fine; courteous, dutiful ones drive "
        "smoothly",
        {"assertiveness": 0.45, "emotional_volatility": 0.40,
         "respectfulness": -0.40, "responsibility": -0.35},
    ),
    "time_budget_flexibility": (
        "+creative_imagination -organization -anxiety -productiveness: improvising drivers "
        "stretch the shift; scheduled, driven, anxious ones hold the plan",
        {"creative_imagination": 0.40, "organization": -0.45,
         "anxiety": -0.30, "productiveness": -0.25},
    ),
    "money_budget_ambition": (
        "+productiveness +assertiveness +responsibility -compassion: output-driven, "
        "assertive drivers push for the earnings target; compassionate ones settle",
        {"productiveness": 0.50, "assertiveness": 0.45,
         "responsibility": 0.20, "compassion": -0.30},
    ),
}

PASSENGER_TRAIT_MODEL: TraitModel = {**_PASSENGER_ONLY_TRAIT_MODEL, **SHARED_TRAIT_MODEL}
DRIVER_TRAIT_MODEL: TraitModel = {**_DRIVER_ONLY_TRAIT_MODEL, **SHARED_TRAIT_MODEL}
TRAIT_MODELS: Dict[str, TraitModel] = {
    "passenger": PASSENGER_TRAIT_MODEL,
    "driver": DRIVER_TRAIT_MODEL,
}

#: ``provider_affinity`` is a lean in [-1, 1], not a [0, 1] trait; every other
#: derived trait is clipped to [0, 1].
LEAN_TRAITS = ("provider_affinity",)

PASSENGER_TRAIT_NAMES: Tuple[str, ...] = tuple(PASSENGER_TRAIT_MODEL) + LEAN_TRAITS
DRIVER_TRAIT_NAMES: Tuple[str, ...] = tuple(DRIVER_TRAIT_MODEL) + LEAN_TRAITS
TRAIT_NAMES: Dict[str, Tuple[str, ...]] = {
    "passenger": PASSENGER_TRAIT_NAMES,
    "driver": DRIVER_TRAIT_NAMES,
}

#: The specification's fixed signs (section 3.2), kept separate from the
#: coefficients above so the self-test checks the model against the spec rather
#: than against itself.
SPEC_TRAIT_SIGNS: Dict[str, Dict[str, Dict[str, int]]] = {
    "passenger": {
        "patience": {"compassion": +1, "responsibility": +1,
                     "anxiety": -1, "emotional_volatility": -1},
        "provider_fidelity": {"organization": +1, "trust": +1,
                              "intellectual_curiosity": -1, "creative_imagination": -1},
        "surge_sensitivity": {"anxiety": +1, "intellectual_curiosity": -1},
        "trust_in_driver": {"trust": +1, "compassion": +1, "anxiety": -1},
        "communication_skills": {"sociability": +1, "respectfulness": +1,
                                 "anxiety": -1, "depression": -1},
        "time_budget_flexibility": {"creative_imagination": +1, "compassion": +1,
                                    "organization": -1, "anxiety": -1},
    },
    "driver": {
        "patience": {"compassion": +1, "responsibility": +1,
                     "anxiety": -1, "emotional_volatility": -1},
        "provider_fidelity": {"organization": +1, "trust": +1,
                              "intellectual_curiosity": -1, "creative_imagination": -1},
        "surge_greediness": {"assertiveness": +1, "productiveness": +1, "compassion": -1},
        "work_urgency": {"productiveness": +1, "responsibility": +1, "depression": -1},
        "trust_in_passenger": {"trust": +1, "compassion": +1, "anxiety": -1},
        "communication_skills": {"sociability": +1, "respectfulness": +1,
                                 "anxiety": -1, "depression": -1},
        "zone_flexibility": {"creative_imagination": +1, "energy_level": +1,
                             "organization": -1},
        "driving_style": {"assertiveness": +1, "emotional_volatility": +1,
                          "respectfulness": -1, "responsibility": -1},
        "time_budget_flexibility": {"creative_imagination": +1,
                                    "organization": -1, "anxiety": -1},
        "money_budget_ambition": {"assertiveness": +1, "productiveness": +1,
                                  "compassion": -1},
    },
}


class TraitView(_ABCMapping):
    """
    Immutable derived-trait table.  Behaves as a read-only ``Mapping``
    (``traits["patience"]``, ``dict(traits)``, ``in``, ``len``) *and* supports
    attribute access (``traits.patience``) for readability at call sites.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Optional[Dict[str, float]] = None, **kwargs: float) -> None:
        merged: Dict[str, float] = dict(data or {})
        merged.update(kwargs)
        object.__setattr__(self, "_data", {k: float(v) for k, v in merged.items()})

    def __getitem__(self, key: str) -> float:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> float:
        if name == "_data":  # pragma: no cover - guards unpickling recursion
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"no derived trait named {name!r}") from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("TraitView is read-only")

    def __delattr__(self, name: str) -> None:
        raise TypeError("TraitView is read-only")

    def as_dict(self) -> Dict[str, float]:
        return dict(self._data)

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v:.3f}" for k, v in self._data.items())
        return f"TraitView({inner})"


# Backwards-compatible aliases: earlier revisions exposed one dataclass per
# role.  Both names now build the same read-only mapping, and keyword
# construction (``PassengerTraits(patience=..., ...)``) still works.
PassengerTraits = TraitView
DriverTraits = TraitView


def _provider_affinity(profile: OceanProfile, provider_fidelity: float) -> float:
    """
    Deterministic brand lean in [-1, 1]: positive = uber-leaning, negative =
    lyft-leaning.  Direction and depth come from the profile hash (so the lean
    is stable for a personality but unrelated to any real brand claim), scaled
    by ``provider_fidelity`` - a loyal personality leans hard, a promiscuous one
    barely leans.  The magnitude never reaches 0, so no provider weight can.
    """
    digest = hashlib.blake2s(profile.profile_id.encode("utf-8"), digest_size=8).digest()
    direction = 1.0 if digest[0] % 2 == 0 else -1.0
    depth = 0.20 + 0.80 * (int.from_bytes(digest[1:4], "big") / float(1 << 24))
    return clip(direction * depth * (0.30 + 0.70 * provider_fidelity), -1.0, 1.0)


def _derive_traits(role: str, profile: OceanProfile) -> TraitView:
    """Derive every operational trait for ``role`` from ``profile``'s facets."""
    model = TRAIT_MODELS[role]
    values: Dict[str, float] = {}
    for trait, (_justification, coefficients) in model.items():
        total = 0.5
        for facet, coefficient in coefficients.items():
            total += coefficient * (getattr(profile, facet) - 0.5)
        values[trait] = clip(total, 0.0, 1.0)
    values["provider_affinity"] = _provider_affinity(profile, values["provider_fidelity"])
    return TraitView(values)


def derive_passenger_traits(profile: ProfileLike) -> TraitView:
    """Operational traits for a passenger (see ``PASSENGER_TRAIT_MODEL``)."""
    return _derive_traits("passenger", _coerce_profile(profile))


def derive_driver_traits(profile: ProfileLike) -> TraitView:
    """Operational traits for a driver (see ``DRIVER_TRAIT_MODEL``)."""
    return _derive_traits("driver", _coerce_profile(profile))


def trait_rationale(role: str, trait: str) -> str:
    """The one-line justification recorded for a derived trait."""
    if trait in LEAN_TRAITS:
        return _provider_affinity.__doc__.strip().splitlines()[0]
    return TRAIT_MODELS[role][trait][0]


#: The all-0.5 profile: the population-neutral reference.  Used as the stand-in
#: counterpart when a ride phase is inspected without a pairing
#: (``tables("ride")``, ``resolve_ride``) and, below, as the fixed point that
#: ``_widen_base`` amplifies each agent's base shares away from.
NEUTRAL_PROFILE = OceanProfile(0.5, 0.5, 0.5, 0.5, 0.5)

#: Traits of that neutral profile, per role, derived once at import.
NEUTRAL_TRAITS: Dict[str, TraitView] = {
    role: _derive_traits(role, NEUTRAL_PROFILE) for role in TRAIT_MODELS
}


# ==========================================================================
# 4. NOTES (one-line behavioural rationale per agent)
# ==========================================================================

_DOMAIN_LETTER = {
    "openness": "O",
    "conscientiousness": "C",
    "extraversion": "E",
    "agreeableness": "A",
    "neuroticism": "N",
}
NOTES_MAX_CHARS = 200


def _band(value: float, low_word: str, mid_word: str, high_word: str,
          low: float = 0.42, high: float = 0.58) -> str:
    if value <= low:
        return low_word
    if value >= high:
        return high_word
    return mid_word


def _dominant_domain_phrase(profile: OceanProfile, count: int = 2) -> str:
    """e.g. "High N, low C" - the profile's most extreme domains, in order."""
    parts = []
    for index, domain in enumerate(profile.dominant_domains(count)):
        word = "High" if getattr(profile, domain) >= 0.5 else "Low"
        parts.append(f"{word if index == 0 else word.lower()} {_DOMAIN_LETTER[domain]}")
    return ", ".join(parts)


def _build_notes(role: str, profile: OceanProfile, traits: TraitView) -> str:
    """
    Deterministic one-line behavioural rationale (<= 200 chars), grounded in the
    profile's most extreme domains and the operational tendencies they produce.
    """
    head = _dominant_domain_phrase(profile)

    tendencies = [_band(traits.patience, "impatient", "even-tempered", "patient")]
    if role == "driver":
        tendencies.append(_band(traits.surge_greediness, "surge-indifferent",
                                "mildly surge-driven", "surge-chasing"))
        tendencies.append(_band(traits.zone_flexibility, "anchored to one zone",
                                "occasionally repositions", "roams between zones"))
        tendencies.append(_band(traits.driving_style, "smooth at the wheel",
                                "unremarkable driving", "aggressive at the wheel"))
    else:
        tendencies.append(_band(traits.surge_sensitivity, "surge-tolerant",
                                "mildly surge-averse", "surge-averse"))
        tendencies.append(_band(traits.trust_in_driver, "wary of drivers",
                                "neutral toward drivers", "trusting of drivers"))
        tendencies.append(_band(traits.time_budget_flexibility, "schedule-bound",
                                "some slack", "flexible schedule"))
    tendencies.append(_band(traits.volatility, "steady under context",
                            "moderately reactive", "volatile ratings"))
    tail = _band(traits.rating_engagement, "abstains from rating often",
                 "rates most rides", "rates almost every ride")

    text = f"{head}: {', '.join(tendencies)}; {tail}."
    if len(text) > NOTES_MAX_CHARS:
        text = text[: NOTES_MAX_CHARS - 1].rstrip(", ") + "."
    return text


def profile_notes(profile: ProfileLike) -> str:
    """Role-neutral note for a profile, for logs and inspection."""
    prof = _coerce_profile(profile)
    driver_traits = _derive_traits("driver", prof)
    head = _dominant_domain_phrase(prof)
    text = (
        f"{head}: {_band(driver_traits.patience, 'impatient', 'even-tempered', 'patient')}, "
        f"{_band(driver_traits.volatility, 'steady under context', 'moderately reactive', 'strongly context-driven')}, "
        f"{_band(driver_traits.rating_discrimination, 'rates near 5 regardless', 'mildly discriminating rater', 'discriminating rater')}; "
        f"{_band(driver_traits.rating_engagement, 'abstains often', 'rates most rides', 'rates almost always')}."
    )
    if len(text) > NOTES_MAX_CHARS:
        text = text[: NOTES_MAX_CHARS - 1].rstrip(", ") + "."
    return text


# ==========================================================================
# 5. GENERIC MODE A / MODE B EVALUATORS
# ==========================================================================

def validate_context(context: Dict[str, str]) -> None:
    """
    Reject an unrecognised value for a *known* factor.

    Unknown keys are ignored (the simulator may pass extra state), and a factor
    that is simply absent contributes nothing.  But ``traffic_level="extreme"``
    is a bug in the scenario, not a neutral input, so it raises.

    ``None`` is treated as "absent", not as a bad value: the simulator legitimately
    produces it for a factor it cannot resolve yet -- ``Drivers._driver_zone``
    returns None while a vehicle is not in the network (classes/drivers.py), and the
    ``.get()`` lookups for zone/age yield None before an agent is fully registered.
    Those are neutral inputs and must not abort a multi-hour run mid-simulation.
    """
    if not context:
        return
    for factor, value in context.items():
        allowed = CONTEXT_VALUES.get(factor)
        if allowed is None:
            continue
        if value is None:
            continue
        if value not in allowed:
            raise ValueError(
                f"unrecognised value {value!r} for context factor '{factor}'; "
                f"expected one of {allowed}"
            )


def normalize_to_probabilities(scores: Dict[str, float]) -> Dict[str, float]:
    """Clip raw scores away from zero, then normalize to sum 1.0."""
    clipped = {action: max(score, 0.001) for action, score in scores.items()}
    total = sum(clipped.values())
    return {action: score / total for action, score in clipped.items()}


def evaluate_mode_a(
        base: Dict[str, float],
        weights: Dict[str, Dict[str, Dict[str, float]]],
        context: Dict[str, str],
    ) -> Dict[str, float]:
    """
    Mode A: base + signed context deltas -> probability per action.

    Draws no randomness.  Returns every action of the phase; the probabilities
    sum to 1.0 within 1e-9.
    """
    validate_context(context)
    scores = dict(base)
    for action, weights_by_factor in weights.items():
        total_delta = 0.0
        for factor, value in (context or {}).items():
            factor_weights = weights_by_factor.get(factor)
            if factor_weights is not None and value in factor_weights:
                total_delta += factor_weights[value]
        scores[action] = scores.get(action, 0.0) + total_delta
    return normalize_to_probabilities(scores)


def evaluate_mode_b(
        rate_base: float,
        rate_weights: Dict[str, Dict[str, float]],
        no_vote_base: float,
        no_vote_weights: Dict[str, Dict[str, float]],
        context: Dict[str, str],
        rng: Optional[random.Random] = None,
    ) -> Tuple[Optional[float], bool]:
    """
    Mode B: abstain, or return a 1-5 rating.

    Exactly one uniform value is drawn from ``rng`` per call, whatever the
    outcome, so seeded runs stay aligned across scenario variants.  No
    normalization happens here: clipping alone handles the bounds.
    """
    validate_context(context)
    if rng is None:
        rng = random
    p_abstain = clip(no_vote_base + _sum_deltas(no_vote_weights, context), 0.0, 1.0)
    draw = rng.random()  # exactly one draw per call, before any branch
    if draw < p_abstain:
        return None, True
    return _finalize_rating(rate_base + _sum_deltas(rate_weights, context)), False


def resolve_mode_b(
        rate_base: float,
        rate_weights: Dict[str, Dict[str, float]],
        no_vote_base: float,
        no_vote_weights: Dict[str, Dict[str, float]],
        context: Dict[str, str],
    ) -> Tuple[float, float]:
    """Deterministic Mode B resolution: ``(p_abstain, rating)``, with no draw."""
    validate_context(context)
    p_abstain = clip(no_vote_base + _sum_deltas(no_vote_weights, context), 0.0, 1.0)
    return p_abstain, _finalize_rating(rate_base + _sum_deltas(rate_weights, context))


def _sum_deltas(table: Dict[str, Dict[str, float]], context: Dict[str, str]) -> float:
    total = 0.0
    for factor, value in (context or {}).items():
        factor_weights = table.get(factor)
        if factor_weights is not None and value in factor_weights:
            total += factor_weights[value]
    return total


def _floor_signed(value: float, fallback_sign: float = 1.0,
                  floor: float = WEIGHT_FLOOR) -> float:
    """
    Keep a weight away from zero without changing its sign.  A value that lands
    inside +/-floor is pushed out to the floor; an exact zero takes
    ``fallback_sign``, which callers set from the trait motivating the factor.
    """
    if value >= floor or value <= -floor:
        return value
    if value > 0.0:
        return floor
    if value < 0.0:
        return -floor
    return (floor if fallback_sign >= 0 else -floor)


# ==========================================================================
# 6. MODE A TABLE BUILDER
# ==========================================================================

class _ModeAPhase:
    """
    Builds one Mode A phase's ``base`` and ``weights`` for one agent.

    Weights are emitted per factor as "one source action is pushed by ``delta``,
    the complementary ``-delta`` is shared out over the other actions".  That
    construction makes the section-6 complementarity requirement structural: for
    every factor value the signed weights across a phase's actions sum to zero
    (to floating-point noise), so context shifts *preference* between actions and
    never inflates or deflates the distribution as a whole.
    """

    def __init__(self, role: str, phase: str, traits: TraitView, step: float) -> None:
        key = (role, phase)
        if PHASE_MODE[key] != "A":
            raise KeyError(f"phase {key} is not a Mode A phase")
        self.role = role
        self.phase = phase
        self.traits = traits
        self.actions: Tuple[str, ...] = tuple(PHASE_ACTIONS[key])
        self.factors: Tuple[str, ...] = tuple(PHASE_FACTORS[key])
        # Every magnitude in the phase is scaled by the agent's volatility:
        # a labile profile reacts strongly to context, a stable one barely moves.
        self.step = step * (0.45 + 1.10 * traits.volatility)
        self.weights: Dict[str, Dict[str, Dict[str, float]]] = {
            action: {factor: {} for factor in self.factors} for action in self.actions
        }

    # -- complementary mass ------------------------------------------------
    def shares(self, source: str, prefs: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        How the complement of a push on ``source`` is split over the other
        actions, in proportion to the traits that motivate each.  Two-action
        phases reduce to ``{other: 1.0}``, i.e. exactly opposite signs.
        """
        others = [a for a in self.actions if a != source]
        if not others:
            raise KeyError(f"phase ({self.role}, {self.phase}) has no action besides {source!r}")
        raw = {a: max(float((prefs or {}).get(a, 1.0)), SHARE_FLOOR) for a in others}
        total = sum(raw.values())
        return {a: v / total for a, v in raw.items()}

    # -- emission ----------------------------------------------------------
    def emit(self, factor: str, source: str, directions: Dict[str, float],
             strength: float, shares: Dict[str, float],
             fallback_sign: float = 1.0) -> None:
        """
        Write every value of ``factor``: ``directions[value] * strength * step``
        onto ``source``, and the complement onto the other actions.
        """
        if factor not in self.factors:
            raise KeyError(
                f"factor '{factor}' is not live in phase ({self.role}, {self.phase}); "
                f"live factors are {list(self.factors)}"
            )
        if self.weights[source][factor]:
            raise KeyError(f"factor '{factor}' already emitted for phase {self.phase}")
        scale = self.step * max(float(strength), 0.05)
        if factor in FLEET_SHARED_FACTORS:
            # A synchronised factor moves every agent the same way at once, so its
            # weight is damped: see FLEET_SHARED_FACTORS.
            scale *= MACRO_FACTOR_DAMP
        for value in CONTEXT_VALUES[factor]:
            if value not in directions:
                raise KeyError(
                    f"phase ({self.role}, {self.phase}) factor '{factor}': "
                    f"no direction for value {value!r}"
                )
            delta = _floor_signed(directions[value] * scale, fallback_sign)
            self.weights[source][factor][value] = delta
            for other, share in shares.items():
                self.weights[other][factor][value] = -delta * share

    # -- finalisation ------------------------------------------------------
    def finish(self, base_raw: Dict[str, float]) -> Tuple[Dict[str, float],
                                                          Dict[str, Dict[str, Dict[str, float]]]]:
        """Normalize the base, then verify coverage, non-zeroness and complementarity."""
        if set(base_raw) != set(self.actions):
            raise KeyError(
                f"phase ({self.role}, {self.phase}) base must contain exactly "
                f"{list(self.actions)}, got {sorted(base_raw)}"
            )
        floored = {a: max(float(base_raw[a]), 0.01) for a in self.actions}
        total = sum(floored.values())
        base = {a: floored[a] / total for a in self.actions}
        # Defensive re-normalization: absorb float error into the largest share.
        drift = 1.0 - sum(base.values())
        if drift:
            largest = max(base, key=lambda a: base[a])
            base[largest] += drift
        assert abs(sum(base.values()) - 1.0) < 1e-12, "Mode A base must sum to 1.0"

        for action in self.actions:
            for factor in self.factors:
                table = self.weights[action][factor]
                expected = CONTEXT_VALUES[factor]
                if set(table) != set(expected):
                    raise KeyError(
                        f"phase ({self.role}, {self.phase}) action '{action}' factor "
                        f"'{factor}': expected values {expected}, got {sorted(table)}"
                    )
                for value, weight in table.items():
                    if weight == 0.0:
                        raise ValueError(
                            f"phase ({self.role}, {self.phase}) action '{action}' factor "
                            f"'{factor}' value '{value}': weight is exactly zero"
                        )
        for factor in self.factors:
            for value in CONTEXT_VALUES[factor]:
                column = sum(self.weights[a][factor][value] for a in self.actions)
                if abs(column) > 1e-6:
                    raise ValueError(
                        f"phase ({self.role}, {self.phase}) factor '{factor}' value "
                        f"'{value}': weights sum to {column!r}, expected ~0"
                    )
        return base, self.weights


def _widen_base(shares_of: Callable[[TraitView], Dict[str, float]],
                role: str, traits: TraitView) -> Dict[str, float]:
    """
    Amplify one agent's base shares away from the population-neutral agent.

    ``shares_of`` is evaluated twice: once with this agent's traits and once with
    ``NEUTRAL_TRAITS[role]``.  Each action's share is then moved away from its
    neutral share by ``PERSON_GAIN[role]`` **multiplicatively** - the ratio to the
    neutral share is raised to the gain - so the neutral agent is an exact fixed
    point and the spread around it widens.  Evaluating the same closure for both
    keeps that reference exact for non-linear share expressions, instead of
    restating a hand-computed constant that would silently drift.

    Widening the ratio rather than the level matters for the phases with three or
    four actions.  Scaling ``value - neutral`` additively drove the small actions
    straight onto the floor: at a passenger gain of 3.4 it left 62% of riders with
    ``P(reject) < 0.01`` and 60% never cancelling at pickup, against real
    cancellation rates of 5-10%.  A ratio can shrink a minority action without
    ever reaching zero, so a rider who rarely rejects still sometimes does.

    Parameters
    ----------
    - shares_of : Callable[[TraitView], Dict[str, float]]
        Builds the phase's raw (un-normalized) base shares from traits.
    - role : str
        ``"driver"`` or ``"passenger"``; selects the gain and the neutral traits.
    - traits : TraitView
        This agent's derived traits.

    Returns
    -------
    Dict[str, float]
        Widened raw shares, each clipped into
        ``[BASE_SHARE_FLOOR, BASE_SHARE_CEIL]``.  ``_ModeAPhase.finish``
        normalizes them, so only their ratios matter.

    Raises
    ------
    KeyError
        If ``role`` is not a known role.
    """
    gain = PERSON_GAIN[role]
    agent = shares_of(traits)
    neutral = shares_of(NEUTRAL_TRAITS[role])
    widened: Dict[str, float] = {}
    for action, value in agent.items():
        reference = max(neutral[action], BASE_SHARE_FLOOR)
        ratio = max(value, BASE_SHARE_FLOOR) / reference
        widened[action] = clip(reference * ratio ** gain,
                               BASE_SHARE_FLOOR, BASE_SHARE_CEIL)
    return widened


def _provider_directions(affinity: float, flip: bool = False) -> Dict[str, float]:
    """
    Per-provider direction from the profile's brand lean.  ``flip=True`` is for
    a source action whose meaning is inverted (e.g. ``change_provider``: the
    preferred brand pushes switching *down*).
    """
    sign = -1.0 if flip else 1.0
    return {"uber": sign * affinity, "lyft": -sign * affinity}


def _age_directions(spread: Dict[str, float], coefficient: float = 1.0,
                    offset: float = 0.0) -> Dict[str, float]:
    """
    Two-sided directions proportional to an age bucket's position in [-1, 1]:
    the youngest and oldest buckets pull opposite ways.  Used where age reads as
    a preference (roam vs stay put) rather than as a sensitivity.
    """
    return {bucket: coefficient * position + offset for bucket, position in spread.items()}


def _fading_age_directions(spread: Dict[str, float], magnitude: float = 1.0,
                           tail: float = 0.05) -> Dict[str, float]:
    """
    A one-sided push that is strongest for the *youngest* bucket and shrinks
    toward ``tail`` with age - section 5.2's "younger passengers are more
    surge-sensitive; the effect shrinks with age".  The push fades, it does not
    reverse into an equally large bonus.
    """
    return {bucket: -magnitude * (1.0 - (position + 1.0) / 2.0) + tail
            for bucket, position in spread.items()}


def _rising_age_directions(spread: Dict[str, float], magnitude: float = 1.0,
                           tail: float = 0.05) -> Dict[str, float]:
    """
    The mirror image: a one-sided push that grows with age - section 5.2's
    "older drivers are more surge-sensitive" (they screen jobs harder).
    """
    return {bucket: -magnitude * ((position + 1.0) / 2.0) - tail
            for bucket, position in spread.items()}


def _time_of_day_directions(traits: TraitView) -> Dict[str, float]:
    """
    The profile's energy pattern: scheduled, low-slack agents are morning
    creatures; flexible, improvising ones tolerate (or prefer) nights.
    """
    flexibility = traits.time_budget_flexibility
    return {
        "morning": 0.60 - 0.30 * flexibility,
        "afternoon": 0.18,
        "night": -0.85 + 1.35 * flexibility,
    }


# ==========================================================================
# 7. DRIVER MODE A PHASES
# ==========================================================================
#
# Per-phase magnitude budgets.  The effective step is this value times
# (0.45 + 1.10 * volatility), so a stable profile moves ~0.45x and a labile one
# ~1.55x as far for the same context.

STEP_DRIVER_START = 0.055
STEP_DRIVER_WAIT = 0.055
STEP_DRIVER_OFFER = 0.060
STEP_PASSENGER_REQUEST = 0.060
STEP_PASSENGER_PICKUP = 0.055


def _build_driver_start(traits: TraitView) -> Tuple[Dict[str, float], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Driver ``start``: reposition first, or start working here?

    Base leans on ``work_urgency`` (a driven profile starts working) and
    ``money_budget_ambition``; ``zone_flexibility`` pulls the other way.
    """
    t = traits
    ph = _ModeAPhase("driver", "start", t, STEP_DRIVER_START)
    to_work = ph.shares("start_work")  # -> {"change_zone": 1.0}

    # Congestion makes starting here less attractive; impatience sharpens that.
    ph.emit("traffic_level", "start_work", LEVEL_PREFER_LOW,
            0.40 + 0.60 * (1.0 - t.patience), to_work, fallback_sign=1.0)
    # A multiplier is a reason to get moving now - scaled by surge greed.
    ph.emit("surge_multiplier", "start_work", LEVEL_PREFER_HIGH,
            0.30 + 0.90 * t.surge_greediness, to_work, fallback_sign=1.0)
    # Brand loyalty: the preferred app is a reason to go online rather than roam.
    ph.emit("provider", "start_work", _provider_directions(t.provider_affinity),
            0.30 + 0.70 * t.provider_fidelity, to_work,
            fallback_sign=1.0 if t.provider_affinity >= 0 else -1.0)
    ph.emit("time_of_day", "start_work", _time_of_day_directions(t),
            0.40 + 0.60 * t.work_urgency, to_work, fallback_sign=1.0)
    # Older drivers work the zone they are in; the tilt shrinks the more
    # zone-flexible the driver is (section 5.2: traits modulate the age effect).
    ph.emit("driver_age", "start_work", _age_directions(DRIVER_AGE_SPREAD, -1.0),
            0.25 + 0.75 * (1.0 - t.zone_flexibility), to_work, fallback_sign=1.0)
    # Downtown raises effective earning ambition, suburbs lowers it.
    ph.emit("zone_requested", "start_work",
            {"downtown": 0.90, "midtown": NEUTRAL_MID, "suburbs": -0.70},
            0.30 + 0.70 * t.money_budget_ambition, to_work, fallback_sign=1.0)

    def shares(tr: TraitView) -> Dict[str, float]:
        start_work = clip(0.38 + 0.40 * tr.work_urgency
                          + 0.12 * tr.money_budget_ambition
                          - 0.15 * tr.zone_flexibility, 0.08, 0.92)
        return {"start_work": start_work, "change_zone": 1.0 - start_work}

    return ph.finish(_widen_base(shares, "driver", t))


def _build_driver_wait_requests(traits: TraitView) -> Tuple[Dict[str, float], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Driver ``wait_requests``: hold position, reposition, switch app, or stop.

    The complement of any push on ``stay_and_wait`` is split over the three exit
    actions in proportion to the traits that motivate each: ``zone_flexibility``
    -> ``change_zone``, low ``provider_fidelity`` -> ``change_provider``,
    low ``work_urgency`` / high ``time_budget_flexibility`` -> ``stop_work``.
    """
    t = traits
    ph = _ModeAPhase("driver", "wait_requests", t, STEP_DRIVER_WAIT)
    exit_prefs = {
        "change_zone": 0.20 + 1.00 * t.zone_flexibility,
        "change_provider": 0.15 + 1.00 * (1.0 - t.provider_fidelity),
        "stop_work": 0.15 + 0.70 * (1.0 - t.work_urgency) + 0.55 * t.time_budget_flexibility,
    }
    from_stay = ph.shares("stay_and_wait", exit_prefs)
    stay_prefs = {
        "stay_and_wait": 1.00 + 0.80 * t.patience,
        "change_zone": 0.20 + 0.80 * t.zone_flexibility,
        "change_provider": 0.15 + 0.80 * (1.0 - t.provider_fidelity),
        "stop_work": 0.15 + 0.60 * (1.0 - t.work_urgency),
    }

    ph.emit("traffic_level", "stay_and_wait", LEVEL_PREFER_LOW,
            0.35 + 0.65 * t.patience, from_stay, fallback_sign=1.0)
    # A live multiplier is worth waiting through.
    ph.emit("surge_multiplier", "stay_and_wait", LEVEL_PREFER_HIGH,
            0.30 + 0.85 * t.surge_greediness, from_stay, fallback_sign=1.0)
    # Dead time is the core driver of leaving; impatience sharpens it.
    ph.emit("time_waiting_driv", "stay_and_wait", LEVEL_PREFER_LOW,
            0.40 + 0.90 * (1.0 - t.patience), from_stay, fallback_sign=1.0)
    ph.emit("time_of_day", "stay_and_wait", _time_of_day_directions(t),
            0.35 + 0.65 * t.work_urgency, from_stay, fallback_sign=1.0)
    # Fatigue: the longer the shift, the stronger the pull to stop - and the
    # complement of *that* push goes back to staying/roaming/switching.
    ph.emit("shift_time", "stop_work", LEVEL_PREFER_HIGH,
            0.40 + 0.80 * (1.0 - t.time_budget_flexibility),
            ph.shares("stop_work", {k: v for k, v in stay_prefs.items() if k != "stop_work"}),
            fallback_sign=1.0)
    ph.emit("driver_age", "stay_and_wait", _age_directions(DRIVER_AGE_SPREAD, -1.0),
            0.25 + 0.75 * (1.0 - t.zone_flexibility), from_stay, fallback_sign=1.0)
    ph.emit("zone_requested", "stay_and_wait",
            {"downtown": 0.80, "midtown": NEUTRAL_MID, "suburbs": -0.70},
            0.30 + 0.70 * t.money_budget_ambition, from_stay, fallback_sign=1.0)
    # The preferred brand pushes *switching* down (hence flip=True).
    ph.emit("provider", "change_provider", _provider_directions(t.provider_affinity, flip=True),
            0.30 + 0.70 * t.provider_fidelity,
            ph.shares("change_provider", {k: v for k, v in stay_prefs.items() if k != "change_provider"}),
            fallback_sign=-1.0 if t.provider_affinity >= 0 else 1.0)

    def shares(tr: TraitView) -> Dict[str, float]:
        return {
            "stay_and_wait": 0.20 + 0.45 * tr.patience,
            "change_zone": 0.08 + 0.30 * tr.zone_flexibility,
            "change_provider": 0.05 + 0.25 * (1.0 - tr.provider_fidelity),
            "stop_work": 0.06 + 0.30 * (1.0 - tr.work_urgency)
                         * (0.5 + 0.5 * tr.time_budget_flexibility),
        }

    return ph.finish(_widen_base(shares, "driver", t))


def _build_driver_offer(traits: TraitView, phase: str) -> Tuple[Dict[str, float], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Driver ``request`` / ``pickup``: accept this job, or reject it?

    Same live factors in both phases, but ``pickup`` is a later commitment: the
    accept base is higher and context moves the decision less (``damp``), because
    the driver has already invested in this ride.
    """
    t = traits
    step = STEP_DRIVER_OFFER
    ph = _ModeAPhase("driver", phase, t, step)
    pickup = phase == "pickup"
    damp = 0.75 if pickup else 1.0
    to_accept = ph.shares("accept")  # -> {"reject": 1.0}

    ph.emit("traffic_level", "accept", LEVEL_PREFER_LOW,
            damp * (0.40 + 0.60 * (1.0 - t.patience)), to_accept, fallback_sign=1.0)
    ph.emit("surge_multiplier", "accept", LEVEL_PREFER_HIGH,
            damp * (0.30 + 0.90 * t.surge_greediness), to_accept, fallback_sign=1.0)
    # A long route is a cost to a time-poor driver and a prize to an ambitious one.
    ph.emit("route_length", "accept",
            {"low": 0.35, "mid": NEUTRAL_MID, "high": -0.75 + 1.00 * t.money_budget_ambition},
            damp * (0.40 + 0.60 * t.time_budget_flexibility), to_accept, fallback_sign=1.0)
    ph.emit("time_estimated_pickup", "accept", LEVEL_PREFER_LOW,
            damp * (0.35 + 0.85 * (1.0 - t.patience)), to_accept, fallback_sign=1.0)
    ph.emit("time_estimated_ride", "accept",
            {"low": 0.25, "mid": NEUTRAL_MID, "high": -0.55 + 0.85 * t.money_budget_ambition},
            damp * (0.40 + 0.60 * t.time_budget_flexibility), to_accept, fallback_sign=1.0)
    ph.emit("time_of_day", "accept", _time_of_day_directions(t),
            damp * (0.35 + 0.65 * t.work_urgency), to_accept, fallback_sign=1.0)
    # A trusting driver barely reads the passenger's rating; a wary one screens on it.
    ph.emit("passenger_rating", "accept", RATING_BUCKET_SPREAD,
            damp * (0.25 + 0.85 * (1.0 - t.trust_in_passenger)), to_accept, fallback_sign=1.0)
    # Section 5.2: older drivers are the more price-selective ones - the penalty
    # on accepting grows with age, and the whole tilt is carried by surge greed
    # rather than applied as a flat constant.
    ph.emit("driver_age", "accept", _rising_age_directions(DRIVER_AGE_SPREAD, 0.90),
            damp * (0.20 + 0.80 * t.surge_greediness), to_accept, fallback_sign=-1.0)
    ph.emit("zone_requested", "accept",
            {"downtown": 0.70, "midtown": NEUTRAL_MID, "suburbs": -0.60},
            damp * (0.30 + 0.70 * t.money_budget_ambition), to_accept, fallback_sign=1.0)

    def shares(tr: TraitView) -> Dict[str, float]:
        accept = clip((0.56 if pickup else 0.44)
                      + 0.16 * tr.surge_greediness + 0.12 * tr.work_urgency
                      + 0.08 * tr.trust_in_passenger - 0.12 * (1.0 - tr.patience),
                      0.12, 0.88)
        return {"accept": accept, "reject": 1.0 - accept}

    return ph.finish(_widen_base(shares, "driver", t))


# ==========================================================================
# 8. PASSENGER MODE A PHASES
# ==========================================================================

def _build_passenger_request(traits: TraitView) -> Tuple[Dict[str, float], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Passenger ``request``: accept the quote, reject and try the other app, or
    reject outright.

    The complement of any push on ``accept`` is split between the two rejections:
    low ``provider_fidelity`` sends it to ``reject_and_change_provider``, a tight
    time budget and high surge sensitivity send it to ``reject``.
    """
    t = traits
    ph = _ModeAPhase("passenger", "request", t, STEP_PASSENGER_REQUEST)
    reject_prefs = {
        "reject_and_change_provider": 0.20 + 1.00 * (1.0 - t.provider_fidelity),
        "reject": 0.20 + 0.80 * (1.0 - t.time_budget_flexibility) + 0.50 * t.surge_sensitivity,
    }
    from_accept = ph.shares("accept", reject_prefs)

    ph.emit("surge_multiplier", "accept", LEVEL_PREFER_LOW,
            0.30 + 0.90 * t.surge_sensitivity, from_accept, fallback_sign=1.0)
    ph.emit("route_length", "accept",
            {"low": 0.30, "mid": NEUTRAL_MID, "high": -0.60},
            0.30 + 0.70 * (1.0 - t.time_budget_flexibility), from_accept, fallback_sign=1.0)
    ph.emit("time_waiting_pass", "accept", LEVEL_PREFER_LOW,
            0.35 + 0.85 * (1.0 - t.patience), from_accept, fallback_sign=1.0)
    ph.emit("time_estimated_ride", "accept",
            {"low": 0.25, "mid": NEUTRAL_MID, "high": -0.50},
            0.30 + 0.70 * (1.0 - t.time_budget_flexibility), from_accept, fallback_sign=1.0)
    ph.emit("time_of_day", "accept", _time_of_day_directions(t),
            0.35 + 0.65 * (1.0 - t.trust_in_driver), from_accept, fallback_sign=1.0)
    # Section 5.2: younger riders are the surge-sensitive ones and the effect
    # fades with age; the whole tilt scales with the rider's own sensitivity, so
    # a surge-tolerant rider shows a much smaller age effect than an anxious one.
    ph.emit("passenger_age", "accept", _fading_age_directions(PASSENGER_AGE_SPREAD, 0.95),
            0.20 + 0.80 * t.surge_sensitivity, from_accept, fallback_sign=-1.0)
    # Downtown: surge is expected and alternatives are scarce, so it dampens
    # price-driven rejection; suburbs does the opposite.
    ph.emit("zone_requested", "accept",
            {"downtown": 0.55, "midtown": NEUTRAL_MID, "suburbs": -0.45},
            0.25 + 0.75 * t.surge_sensitivity, from_accept, fallback_sign=1.0)
    # The preferred brand pushes brand-switching down.
    ph.emit("provider", "reject_and_change_provider",
            _provider_directions(t.provider_affinity, flip=True),
            0.30 + 0.70 * t.provider_fidelity,
            ph.shares("reject_and_change_provider",
                      {"accept": 1.00 + 0.80 * t.patience,
                       "reject": 0.20 + 0.60 * t.surge_sensitivity}),
            fallback_sign=-1.0 if t.provider_affinity >= 0 else 1.0)

    def shares(tr: TraitView) -> Dict[str, float]:
        accept = clip(0.55 + 0.15 * tr.patience + 0.12 * tr.trust_in_driver
                      - 0.18 * tr.surge_sensitivity, 0.15, 0.88)
        switch = clip(0.10 + 0.25 * (1.0 - tr.provider_fidelity), 0.05, 0.42)
        reject = max(1.0 - accept - switch, 0.05)
        return {"accept": accept,
                "reject_and_change_provider": switch,
                "reject": reject}

    return ph.finish(_widen_base(shares, "passenger", t))


def _build_passenger_pickup(traits: TraitView) -> Tuple[Dict[str, float], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Passenger ``pickup``: get in, or cancel?  A later commitment than
    ``request``, so the accept base is higher and the deltas are damped.
    """
    t = traits
    ph = _ModeAPhase("passenger", "pickup", t, STEP_PASSENGER_PICKUP)
    to_accept = ph.shares("accept")  # -> {"reject": 1.0}
    damp = 0.85

    ph.emit("surge_multiplier", "accept", LEVEL_PREFER_LOW,
            damp * (0.25 + 0.80 * t.surge_sensitivity), to_accept, fallback_sign=1.0)
    ph.emit("route_length", "accept",
            {"low": 0.25, "mid": NEUTRAL_MID, "high": -0.45},
            damp * (0.25 + 0.65 * (1.0 - t.time_budget_flexibility)), to_accept, fallback_sign=1.0)
    # Already-elapsed waiting at the kerb is the sharpest cancel driver.
    ph.emit("time_waiting_pickup", "accept", LEVEL_PREFER_LOW,
            damp * (0.40 + 0.90 * (1.0 - t.patience)), to_accept, fallback_sign=1.0)
    ph.emit("time_estimated_pickup", "accept", LEVEL_PREFER_LOW,
            damp * (0.35 + 0.75 * (1.0 - t.patience)), to_accept, fallback_sign=1.0)
    ph.emit("time_estimated_ride", "accept",
            {"low": 0.20, "mid": NEUTRAL_MID, "high": -0.35},
            damp * (0.25 + 0.65 * (1.0 - t.time_budget_flexibility)), to_accept, fallback_sign=1.0)
    ph.emit("time_of_day", "accept", _time_of_day_directions(t),
            damp * (0.30 + 0.70 * (1.0 - t.trust_in_driver)), to_accept, fallback_sign=1.0)
    # A trusting rider hardly looks at the star count; a wary one leans on it.
    ph.emit("driver_rating", "accept", RATING_BUCKET_SPREAD,
            damp * (0.30 + 0.90 * (1.0 - t.trust_in_driver)), to_accept, fallback_sign=1.0)
    ph.emit("passenger_age", "accept", _fading_age_directions(PASSENGER_AGE_SPREAD, 0.95),
            damp * (0.20 + 0.80 * t.surge_sensitivity), to_accept, fallback_sign=-1.0)
    ph.emit("zone_requested", "accept",
            {"downtown": 0.40, "midtown": NEUTRAL_MID, "suburbs": -0.35},
            damp * (0.25 + 0.75 * t.surge_sensitivity), to_accept, fallback_sign=1.0)

    def shares(tr: TraitView) -> Dict[str, float]:
        accept = clip(0.66 + 0.14 * tr.patience + 0.10 * tr.trust_in_driver
                      - 0.15 * tr.surge_sensitivity, 0.20, 0.93)
        return {"accept": accept, "reject": 1.0 - accept}

    return ph.finish(_widen_base(shares, "passenger", t))


# ==========================================================================
# 9. MODE B RIDE PHASES (rating with abstention)
# ==========================================================================
#
# Real ride-hailing ratings are heavily right-skewed: an unremarkable ride tends
# to 5, and dissatisfaction surfaces as silence at least as often as it surfaces
# as a low vote.  That skew lives in the bases and the weights - nothing is
# post-processed.  The rating agent's weight tables depend ONLY on its own
# personality; the counterpart shifts only the two scalar bases.

#: Platforms take whole stars only, so the continuous latent is rounded on the way
#: out (see _finalize_rating).  Set False to recover the continuous value.
INTEGER_STARS = True

# A pairing score in [0, 1] maps to a star base through a skewed curve rather than
# a straight line: an unremarkable pairing sits near 5, and the base falls away
# only at the genuinely bad end.  That is the shape real ride-hailing ratings have
# - overwhelmingly 5s, a thin 4 shoulder, a small tail - and it must live in the
# base rather than in post-processing.  A linear map over a 4.0-5.0 band could not
# produce it: measured on the pre-change model, *no* rating in 4000 draws landed
# below 4, so a dissatisfied rider had no way to express dissatisfaction except by
# abstaining.
# Tuned against the published Uber/Lyft rating shape (~80-90% five-star, mean
# 4.8-4.9, a thin sub-4 tail) by grid search over the floor, the exponent and the
# pairing gain, scoring each combination on total absolute deviation from that
# shape over a realistic mix of ride contexts.  The fitted point reproduces
# mean 4.85 with 88.0% / 10.2% / 0.9% / 0.8% / 0.1% across five to one star.
RATE_BASE_CEIL = 5.0
RATE_BASE_FLOOR = 1.0
RATE_BASE_SKEW = 4.0
#: Amplifies a pairing's distance from the neutral pairing, the same way
#: PERSON_GAIN does for Mode A bases: without it the raw trait blend lands in a
#: narrow band on the flat top of the skew curve and every ride rounds to 5.
RATING_PAIRING_GAIN = 2.5
#: Ceiling on how far context can move one agent's stars, per factor.
RATING_MAGNITUDE_CEIL = 0.32
NO_VOTE_BASE_BAND = (0.12, 0.55)


def _pairing_score(blend: Callable[[TraitView, TraitView], float],
                   own: TraitView, counterpart: TraitView,
                   neutral_counterpart: TraitView) -> float:
    """
    Centre a raw trait blend on the neutral pairing, then widen it.

    ``blend`` is evaluated for this pairing and for the same rater against a
    neutral counterpart.  The difference is amplified by
    ``RATING_PAIRING_GAIN`` and added to 0.5, so a neutral pairing scores exactly
    0.5 (the middle of the skew curve) and better/worse pairings spread out from
    there.  Centring matters more than the gain: the raw blend averages well
    above 0.5, which parked every pairing on the curve's flat top.

    Parameters
    ----------
    - blend : Callable[[TraitView, TraitView], float]
        Maps ``(own_traits, counterpart_traits)`` to a raw score.
    - own : TraitView
        The rating agent's traits.
    - counterpart : TraitView
        The rated agent's traits.
    - neutral_counterpart : TraitView
        Traits of the neutral profile in the counterpart's role.

    Returns
    -------
    float
        A pairing score in ``[0, 1]``.
    """
    raw = blend(own, counterpart)
    reference = blend(own, neutral_counterpart)
    return clip(0.5 + RATING_PAIRING_GAIN * (raw - reference))


def _skewed_rate_base(pairing: float) -> float:
    """
    Star base for a pairing score, on the right-skewed curve described above.

    Parameters
    ----------
    - pairing : float
        How good this pairing is, in ``[0, 1]``: the rater's generosity blended
        with the counterpart's quality.

    Returns
    -------
    float
        A base in ``[RATE_BASE_FLOOR, RATE_BASE_CEIL]``.  A mid pairing (0.5)
        lands near 4.7; only a jointly poor pairing approaches the floor.
    """
    span = RATE_BASE_CEIL - RATE_BASE_FLOOR
    shortfall = span * (1.0 - clip(pairing)) ** RATE_BASE_SKEW
    return clip(RATE_BASE_CEIL - shortfall, RATE_BASE_FLOOR, RATE_BASE_CEIL)


def _finalize_rating(value: float) -> float:
    """Clip a latent rating into ``[1, 5]`` and, unless disabled, round to a star."""
    stars = clip(value, 1.0, 5.0)
    if not INTEGER_STARS:
        return stars
    return float(min(5, max(1, int(stars + 0.5))))


def _rating_magnitude(traits: TraitView) -> float:
    """
    How far context can move this agent's stars.  Discriminating raters (high
    ``aesthetic_sensitivity`` via ``rating_discrimination``) and volatile ones
    spread their votes; deferential, steady ones sit on 5.
    """
    return clip(0.07 + 0.16 * traits.rating_discrimination + 0.09 * traits.volatility,
                RATING_WEIGHT_FLOOR, RATING_MAGNITUDE_CEIL)


def _abstain_magnitude(traits: TraitView) -> float:
    """How far context can move this agent's willingness to vote at all."""
    return clip(0.020 + 0.025 * (1.0 - traits.rating_engagement) + 0.015 * traits.volatility,
                ABSTAIN_WEIGHT_FLOOR, 0.055)


def _scale_mode_b_table(directions: Dict[str, Dict[str, float]], magnitude: float,
                        floor: float, factors: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Turn per-value directions into a signed, never-zero, fully covered table."""
    if set(directions) != set(factors):
        raise KeyError(f"Mode B table must cover exactly {list(factors)}, got {sorted(directions)}")
    table: Dict[str, Dict[str, float]] = {}
    for factor in factors:
        values = CONTEXT_VALUES[factor]
        given = directions[factor]
        if set(given) != set(values):
            raise KeyError(
                f"Mode B factor '{factor}': expected values {values}, got {sorted(given)}"
            )
        table[factor] = {v: _floor_signed(given[v] * magnitude, 1.0, floor) for v in values}
    return table


def _fit_no_vote_table(table: Dict[str, Dict[str, float]],
                       base_band: Tuple[float, float] = NO_VOTE_BASE_BAND,
                       margin: float = 0.002) -> Dict[str, Dict[str, float]]:
    """
    Guarantee invariant 8 *by construction*: for every reachable
    ``no_vote_base``, the base plus the most negative and the most positive
    achievable sums of this table both stay inside [0, 1].

    The table is precomputed once per agent while the base depends on the
    pairing, so the fit is taken against the *whole* base band
    (``NO_VOTE_BASE_BAND``) rather than one pairing's base.  The directions and
    bands below are chosen so the worst case already fits and this returns the
    table untouched; the scaling is the construction-time safety net that keeps
    the guarantee true for every profile, rather than leaving it to the runtime
    clip.
    """
    if not table:
        return table
    reach_up = sum(max(values.values()) for values in table.values())
    reach_down = sum(min(values.values()) for values in table.values())
    factor = 1.0
    head = 1.0 - margin - base_band[1]
    if reach_up > 0.0 and reach_up > head:
        factor = min(factor, max(head, 0.0) / reach_up)
    foot = base_band[0] - margin
    if reach_down < 0.0 and -reach_down > foot:
        factor = min(factor, max(foot, 0.0) / (-reach_down))
    factor = clip(factor, 0.05, 1.0)
    if factor >= 1.0:
        return table
    return {f: {v: w * factor for v, w in values.items()} for f, values in table.items()}


def _build_driver_ride_tables(traits: TraitView) -> Tuple[Dict[str, Dict[str, float]],
                                                          Dict[str, Dict[str, float]]]:
    """
    Driver rates passenger: ``(rate_weights, no_vote_weights)`` from the
    driver's own personality only.

    Heavy traffic and a long shift both cost stars and buy silence; the
    preferred provider is worth a hair of goodwill and a hair more engagement;
    the profile's energy pattern drives the time-of-day tilt; older drivers rate
    more conservatively and the oldest buckets abstain more; downtown is the
    stressful zone.
    """
    t = traits
    factors = PHASE_FACTORS[("driver", "ride")]
    loyalty = t.provider_affinity * (0.40 + 0.60 * t.provider_fidelity)
    rate_directions = {
        "traffic_level": {"low": 0.35, "mid": -0.12,
                          "high": -(0.55 + 0.45 * (1.0 - t.patience))},
        "provider": {"uber": loyalty, "lyft": -loyalty},
        "time_of_day": _ride_time_of_day_directions(t),
        "shift_time": {"low": 0.25, "mid": -0.20,
                       "high": -(0.50 + 0.50 * (1.0 - t.time_budget_flexibility))},
        "driver_age": _age_directions(DRIVER_AGE_SPREAD,
                                      -0.70 * (0.40 + 0.60 * (1.0 - t.patience))),
        "zone_requested": {"downtown": -(0.60 + 0.40 * (1.0 - t.patience)),
                           "midtown": -0.15, "suburbs": 0.30},
    }
    no_vote_directions = {
        "traffic_level": {"low": -0.30, "mid": -0.12, "high": 1.00},
        "provider": {"uber": -0.30 * t.provider_affinity, "lyft": 0.30 * t.provider_affinity},
        "time_of_day": {"morning": -0.28, "afternoon": -0.12, "night": 0.85},
        "shift_time": {"low": -0.30, "mid": 0.15, "high": 1.00},
        "driver_age": _age_directions(DRIVER_AGE_SPREAD, 0.45, offset=0.10),
        "zone_requested": {"downtown": 0.80, "midtown": 0.10, "suburbs": -0.25},
    }
    rate = _scale_mode_b_table(rate_directions, _rating_magnitude(t),
                               RATING_WEIGHT_FLOOR, factors)
    no_vote = _fit_no_vote_table(_scale_mode_b_table(
        no_vote_directions, _abstain_magnitude(t), ABSTAIN_WEIGHT_FLOOR, factors))
    return rate, no_vote


def _build_passenger_ride_tables(traits: TraitView) -> Tuple[Dict[str, Dict[str, float]],
                                                             Dict[str, Dict[str, float]]]:
    """
    Passenger rates driver: ``(rate_weights, no_vote_weights)`` from the
    passenger's own personality only.  Provider loyalty bias, the rider's energy
    pattern over the day, more conservative stars with age, and zone stress with
    downtown strongest.
    """
    t = traits
    factors = PHASE_FACTORS[("passenger", "ride")]
    loyalty = t.provider_affinity * (0.40 + 0.60 * t.provider_fidelity)
    rate_directions = {
        "provider": {"uber": loyalty, "lyft": -loyalty},
        "time_of_day": _ride_time_of_day_directions(t),
        "passenger_age": _age_directions(PASSENGER_AGE_SPREAD,
                                         -0.75 * (0.40 + 0.60 * (1.0 - t.patience))),
        "zone_requested": {"downtown": -(0.60 + 0.40 * (1.0 - t.patience)),
                           "midtown": -0.15, "suburbs": 0.30},
    }
    no_vote_directions = {
        "provider": {"uber": -0.30 * t.provider_affinity, "lyft": 0.30 * t.provider_affinity},
        "time_of_day": {"morning": -0.28, "afternoon": -0.12, "night": 0.85},
        "passenger_age": _age_directions(PASSENGER_AGE_SPREAD, 0.45, offset=0.10),
        "zone_requested": {"downtown": 0.85, "midtown": 0.10, "suburbs": -0.25},
    }
    rate = _scale_mode_b_table(rate_directions, _rating_magnitude(t),
                               RATING_WEIGHT_FLOOR, factors)
    no_vote = _fit_no_vote_table(_scale_mode_b_table(
        no_vote_directions, _abstain_magnitude(t), ABSTAIN_WEIGHT_FLOOR, factors))
    return rate, no_vote


def _ride_time_of_day_directions(traits: TraitView) -> Dict[str, float]:
    """
    Time-of-day tilt for a *rating*: a night ride is harder work and costs
    stars, and the penalty shrinks the more flexible (night-tolerant) the agent
    is.  Distinct from ``_time_of_day_directions``, which is about when an agent
    wants to be on the road at all.
    """
    flexibility = traits.time_budget_flexibility
    return {
        "morning": 0.40 - 0.15 * flexibility,
        "afternoon": 0.12,
        "night": -1.00 + 1.00 * flexibility,
    }


def driver_ride_pair_base(driver_traits: TraitView,
                          passenger_traits: TraitView) -> Tuple[float, float]:
    """
    ``(rate_base, no_vote_base)`` for a driver rating a passenger.

    ``rate_base`` places the pairing in the 4.5-5.0 band: a communicative,
    patient, trusting driver with an easy passenger sits at the top; a reserved,
    impatient driver with a difficult passenger at the bottom.  ``no_vote_base``
    is the driver's engagement - disengaged drivers stay silent instead of
    rating low - nudged up slightly by a difficult passenger.
    """
    def blend(own: TraitView, other: TraitView) -> float:
        generosity = (0.35 * own.communication_skills
                      + 0.35 * own.patience
                      + 0.30 * own.trust_in_passenger)
        passenger_ease = (0.55 * other.communication_skills
                          + 0.45 * other.patience)
        return 0.55 * generosity + 0.45 * passenger_ease

    passenger_ease = (0.55 * passenger_traits.communication_skills
                      + 0.45 * passenger_traits.patience)
    rate_base = _skewed_rate_base(
        _pairing_score(blend, driver_traits, passenger_traits,
                       NEUTRAL_TRAITS["passenger"]))
    no_vote_base = clip(0.40 - 0.32 * driver_traits.rating_engagement
                        + 0.10 * (1.0 - passenger_ease), *NO_VOTE_BASE_BAND)
    return rate_base, no_vote_base


def passenger_ride_pair_base(passenger_traits: TraitView,
                             driver_traits: TraitView) -> Tuple[float, float]:
    """
    ``(rate_base, no_vote_base)`` for a passenger rating a driver.

    Smooth, patient, communicative driving lands near 5.0; aggressive driving
    with poor communication lands at the bottom of the band.  ``no_vote_base``
    is the rider's engagement, nudged by how hard the driver was to read.
    """
    def blend(own: TraitView, other: TraitView) -> float:
        generosity = (0.35 * own.communication_skills
                      + 0.30 * own.patience
                      + 0.35 * own.trust_in_driver)
        quality = (0.40 * other.communication_skills
                   + 0.30 * other.patience
                   + 0.30 * (1.0 - other.driving_style))
        return 0.45 * generosity + 0.55 * quality

    driver_quality = (0.40 * driver_traits.communication_skills
                      + 0.30 * driver_traits.patience
                      + 0.30 * (1.0 - driver_traits.driving_style))
    rate_base = _skewed_rate_base(
        _pairing_score(blend, passenger_traits, driver_traits,
                       NEUTRAL_TRAITS["driver"]))
    no_vote_base = clip(0.40 - 0.32 * passenger_traits.rating_engagement
                        + 0.10 * (1.0 - driver_quality), *NO_VOTE_BASE_BAND)
    return rate_base, no_vote_base


# ==========================================================================
# 10. PER-AGENT VIEWS
# ==========================================================================

#: Stand-in counterpart used when a Mode B table is inspected without a real


class _AgentView:
    """
    Shared machinery for the two role views.

    The constructor derives the operational traits **once** and precomputes
    every personality-only table **once**; only context arithmetic happens per
    call.  Nothing that depends on a context or on a draw is cached.
    """

    role: str = ""

    def __init__(self, profile: ProfileLike) -> None:
        self.profile: OceanProfile = _coerce_profile(profile)
        self._traits: TraitView = _derive_traits(self.role, self.profile)
        self._mode_a: Dict[str, Tuple[Dict[str, float], Dict[str, Dict[str, Dict[str, float]]]]] = {}
        self._build_mode_a_tables()
        self._rate_weights, self._no_vote_weights = self._build_ride_tables()
        self.notes: str = _build_notes(self.role, self.profile, self._traits)

    # -- construction hooks ------------------------------------------------
    def _build_mode_a_tables(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _build_ride_tables(self) -> Tuple[Dict[str, Dict[str, float]],
                                          Dict[str, Dict[str, float]]]:  # pragma: no cover
        raise NotImplementedError

    def _ride_bases(self, counterpart_traits: TraitView) -> Tuple[float, float]:  # pragma: no cover
        raise NotImplementedError

    # -- inspection --------------------------------------------------------
    @property
    def traits(self) -> TraitView:
        """Read-only mapping of derived operational traits (also dotted access)."""
        return self._traits

    @property
    def profile_id(self) -> str:
        return self.profile.profile_id

    def phases(self) -> List[str]:
        """Every phase this view can evaluate, Mode A first."""
        return list(MODE_A_PHASES[self.role]) + ["ride"]

    def tables(self, phase: str) -> Dict[str, Any]:
        """
        Deep copy of one phase's ``{"base": ..., "weights": ...}``, for
        inspection, diffing and thesis documentation.

        For the Mode B ``ride`` phase the bases depend on the pairing, so the
        returned base is the one this agent would use against a neutral
        counterpart (``NEUTRAL_PROFILE``).
        """
        if phase == "ride":
            rate_base, no_vote_base = self._ride_bases(_neutral_counterpart_traits(self.role))
            return copy.deepcopy({
                "base": {"rate_base": rate_base, "no_vote_base": no_vote_base},
                "weights": {"rate_weights": self._rate_weights,
                            "no_vote_weights": self._no_vote_weights},
            })
        if phase not in self._mode_a:
            raise KeyError(
                f"unknown phase '{phase}' for role '{self.role}'; "
                f"available phases are {self.phases()}"
            )
        base, weights = self._mode_a[phase]
        return copy.deepcopy({"base": base, "weights": weights})

    # -- evaluation --------------------------------------------------------
    def _evaluate(self, phase: str, context: Optional[Dict[str, str]]) -> Dict[str, float]:
        base, weights = self._mode_a[phase]
        return evaluate_mode_a(base, weights, context or {})

    def choose(self, phase: str, context: Optional[Dict[str, str]] = None,
               rng: Optional[random.Random] = None) -> str:
        """
        Sample one action from a Mode A phase's distribution (one rng draw).
        The phase methods themselves never draw randomness.
        """
        if phase not in self._mode_a:
            raise KeyError(
                f"phase '{phase}' is not a Mode A phase for role '{self.role}'; "
                f"choose() supports {list(MODE_A_PHASES[self.role])}"
            )
        probabilities = self._evaluate(phase, context)
        if rng is None:
            rng = random
        draw = rng.random()
        cumulative = 0.0
        actions = PHASE_ACTIONS[(self.role, phase)]
        for action in actions:
            cumulative += probabilities[action]
            if draw < cumulative:
                return action
        return actions[-1]

    def _rate(self, counterpart_traits: TraitView, context: Optional[Dict[str, str]],
              rng: Optional[random.Random]) -> Tuple[Optional[float], bool]:
        rate_base, no_vote_base = self._ride_bases(counterpart_traits)
        return evaluate_mode_b(rate_base, self._rate_weights,
                               no_vote_base, self._no_vote_weights,
                               context or {}, rng)

    def resolve_ride(self, counterpart: Any = None,
                     context: Optional[Dict[str, str]] = None) -> Tuple[float, float]:
        """Deterministic ``(p_abstain, rating)`` for the ride phase - no draw."""
        traits = (_neutral_counterpart_traits(self.role) if counterpart is None
                  else _counterpart_traits(counterpart, _COUNTERPART_ROLE[self.role]))
        rate_base, no_vote_base = self._ride_bases(traits)
        return resolve_mode_b(rate_base, self._rate_weights,
                              no_vote_base, self._no_vote_weights, context or {})

    def __repr__(self) -> str:
        return f"{type(self).__name__}(profile_id={self.profile_id!r})"


_COUNTERPART_ROLE = {"driver": "passenger", "passenger": "driver"}


def _counterpart_traits(counterpart: Any, role: str) -> TraitView:
    """Traits of the other party, from a view, a profile or a plain dict."""
    if isinstance(counterpart, _AgentView):
        if counterpart.role != role:
            raise TypeError(
                f"expected a {role} counterpart, got a {counterpart.role} view"
            )
        return counterpart.traits
    if isinstance(counterpart, TraitView):
        return counterpart
    return _derive_traits(role, _coerce_profile(counterpart))


def _neutral_counterpart_traits(role: str) -> TraitView:
    return _derive_traits(_COUNTERPART_ROLE[role], NEUTRAL_PROFILE)


class DriverView(_AgentView):
    """
    One driver agent's OCEAN-derived decision behaviour.

        driver = DriverView(profile_dict_or_OceanProfile)
        probs = driver.start(context)          # change_zone / start_work
        probs = driver.wait_requests(context)  # stay / zone / provider / stop
        probs = driver.request(context)        # accept / reject
        probs = driver.pickup(context)         # accept / reject
        rating, abstained = driver.rate_passenger(passenger_view, context, rng)
    """

    role = "driver"

    def _build_mode_a_tables(self) -> None:
        self._mode_a["start"] = _build_driver_start(self._traits)
        self._mode_a["wait_requests"] = _build_driver_wait_requests(self._traits)
        self._mode_a["request"] = _build_driver_offer(self._traits, "request")
        self._mode_a["pickup"] = _build_driver_offer(self._traits, "pickup")

    def _build_ride_tables(self):
        return _build_driver_ride_tables(self._traits)

    def _ride_bases(self, counterpart_traits: TraitView) -> Tuple[float, float]:
        return driver_ride_pair_base(self._traits, counterpart_traits)

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "DriverView":
        """Build from a JSON-like record; unrelated keys are ignored."""
        return cls(OceanProfile.from_json(data))

    # -- Mode A phases -----------------------------------------------------
    def start(self, context: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        return self._evaluate("start", context)

    def wait_requests(self, context: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        return self._evaluate("wait_requests", context)

    def request(self, context: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        return self._evaluate("request", context)

    def pickup(self, context: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        return self._evaluate("pickup", context)

    # -- Mode B phase ------------------------------------------------------
    def rate_passenger(self, passenger: Any, context: Optional[Dict[str, str]] = None,
                       rng: Optional[random.Random] = None) -> Tuple[Optional[float], bool]:
        """
        Rate the passenger, or abstain: ``(rating|None, abstained)``.
        Exactly one rng draw, whatever the outcome.
        """
        return self._rate(_counterpart_traits(passenger, "passenger"), context, rng)


class PassengerView(_AgentView):
    """
    One passenger agent's OCEAN-derived decision behaviour.

        passenger = PassengerView(profile_dict_or_OceanProfile)
        probs = passenger.request(context)  # accept / reject_and_change_provider / reject
        probs = passenger.pickup(context)   # accept / reject
        rating, abstained = passenger.rate_driver(driver_view, context, rng)
    """

    role = "passenger"

    def _build_mode_a_tables(self) -> None:
        self._mode_a["request"] = _build_passenger_request(self._traits)
        self._mode_a["pickup"] = _build_passenger_pickup(self._traits)

    def _build_ride_tables(self):
        return _build_passenger_ride_tables(self._traits)

    def _ride_bases(self, counterpart_traits: TraitView) -> Tuple[float, float]:
        return passenger_ride_pair_base(self._traits, counterpart_traits)

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "PassengerView":
        """Build from a JSON-like record; unrelated keys are ignored."""
        return cls(OceanProfile.from_json(data))

    # -- Mode A phases -----------------------------------------------------
    def request(self, context: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        return self._evaluate("request", context)

    def pickup(self, context: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        return self._evaluate("pickup", context)

    # -- Mode B phase ------------------------------------------------------
    def rate_driver(self, driver: Any, context: Optional[Dict[str, str]] = None,
                    rng: Optional[random.Random] = None) -> Tuple[Optional[float], bool]:
        """
        Rate the driver, or abstain: ``(rating|None, abstained)``.
        Exactly one rng draw, whatever the outcome.
        """
        return self._rate(_counterpart_traits(driver, "driver"), context, rng)


# ==========================================================================
# 11. CSV SOURCES, SAMPLING AND THE SIMULATOR ENTRY POINT
# ==========================================================================

_STD_NORMAL = NormalDist()


@dataclass(frozen=True)
class SourceScale:
    """A measurement scale a CSV's means/SDs are expressed on."""

    name: str
    low: float
    high: float


#: BFI-2 style 1-5 Likert scale (population and students distributions).
LIKERT_SCALE = SourceScale("likert-1-5", 1.0, 5.0)
#: T-score scale of the per-job tables: population mean 50, SD 10, so +/-3 SD.
TSCORE_SCALE = SourceScale("t-score-50-10", 20.0, 80.0)
#: Change these two in one place to re-scale every sampled population.
DEFAULT_SOURCE_SCALE = LIKERT_SCALE
JOBS_SOURCE_SCALE = TSCORE_SCALE
#: Set True to map T-scores through the normal CDF (population percentile)
#: instead of linearly across ``TSCORE_SCALE``.
TSCORE_PERCENTILE_MAPPING = False
# The CSVs carry domain means and SDs but no facet structure, so facets are
# sampled around their parent domain.  The jitter SD is *derived* from the target
# within-domain facet intercorrelation rather than fixed, because a fixed SD makes
# the correlation an accident of whichever SD the job happens to carry: with the
# old FACET_SPREAD_SD = 0.08 against a domain SD near 0.17, facets came out
# correlated at 0.80-0.87, well above the 0.43-0.67 that BFI-2 reports for real
# within-domain facet intercorrelations (Soto & John, 2017).  Facets that
# redundant carry no information the domain does not already carry, so the
# fifteen-facet machinery bought nothing.
#
# For facet = domain + N(0, s) the implied correlation between two facets of the
# same domain is r = var(domain) / (var(domain) + s^2), so the s that hits a
# target r is s = sd(domain) * sqrt((1 - r) / r).
FACET_DOMAIN_CORRELATION = 0.55
#: Floor on the derived jitter, so a job whose CSV SD is ~0 still gets facets that
#: differ from one another instead of fifteen copies of one number.
FACET_JITTER_FLOOR = 0.02


def _facet_jitter_sd(domain_unit_sd: float) -> float:
    """
    SD of the per-facet jitter that yields ``FACET_DOMAIN_CORRELATION``.

    Parameters
    ----------
    - domain_unit_sd : float
        SD of the parent domain on the ``[0, 1]`` scale.

    Returns
    -------
    float
        Jitter SD, never below ``FACET_JITTER_FLOOR``.
    """
    target = clip(FACET_DOMAIN_CORRELATION, 0.05, 0.95)
    return max(domain_unit_sd * ((1.0 - target) / target) ** 0.5, FACET_JITTER_FLOOR)

# Population-level Big Five distribution on the 1-5 Likert scale (mean, sd).
# Job code "0" draws every agent from this single global distribution instead of
# assigning each agent a job.
_POPULATION_BIG5 = {
    "openness": (3.6491, 0.68558),
    "conscientiousness": (3.8857, 0.72289),
    "extraversion": (3.2165, 0.72822),
    "agreeableness": (3.8175, 0.63391),
    "neuroticism": (2.7155, 0.89516),
}

# OceanProfile domain name -> CSV column stem in job_personalities_big5.csv
_DOMAIN_COLUMNS = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "neuroticism": "Neuroticism",
}


def _map_source_to_unit(value: float, scale: SourceScale) -> float:
    """Clip a draw to its source scale, then map it into [0, 1]."""
    clipped = clip(value, scale.low, scale.high)
    if TSCORE_PERCENTILE_MAPPING and scale.name == TSCORE_SCALE.name:
        return clip(_STD_NORMAL.cdf((clipped - 50.0) / 10.0), 0.0, 1.0)
    return (clipped - scale.low) / (scale.high - scale.low)


class OceanDecisionModel:
    """
    Entry point used by the simulator: loads per-job Big Five means/SDs and
    creates per-agent decision views.

        ocean = OceanDecisionModel(OCEAN_JOBS_CSV_PATH, rng=random.Random(42))
        driver_view = ocean.new_driver_view()
        passenger_view = ocean.new_passenger_view()

    Three distribution sources stay distinct:

    * ``"0"`` - the general population distribution (``_POPULATION_BIG5``,
      1-5 Likert).  Every agent is drawn from one global distribution.
    * ``"students"`` - the optional students distribution, loaded from
      ``students_csv_path`` (same column format, 1-5 Likert).
    * any other code - the specific job with that ``Code`` in the jobs CSV
      (T-scores; agents still differ through the per-domain Normal(M, SD) draw).

    Every draw goes through ``self.rng``, so a seeded model reproduces the same
    population exactly.
    """

    STUDENTS_JOB_CODE = "students"
    POPULATION_JOB_CODE = "0"
    #: Share of drivers drawn from the professional (8322) job under the population job code
    PROFESSIONAL_DRIVERS_SHARE = 0.37

    def __init__(
            self,
            jobs_csv_path: str,
            drivers_job_code: str = "0",
            passengers_job_code: str = "0",
            students_csv_path: Optional[str] = None,
            rng: Optional[random.Random] = None,
        ) -> None:
        self.jobs = self._load_jobs(jobs_csv_path)
        self._jobs_by_code = {job["code"]: job for job in self.jobs}
        # Students OCEAN distribution (Soto & John): same CSV column format as
        # the jobs file, but the values are on the BFI-2 1-5 Likert scale rather
        # than T-scores, so sampling uses LIKERT_SCALE.
        self._students: Optional[Dict[str, Any]] = None
        if students_csv_path is not None:
            try:
                self._students = self._load_jobs(students_csv_path)[0]
            except (OSError, ValueError) as exc:
                warnings.warn(
                    f"students distribution not loaded from {students_csv_path!r}: {exc}",
                    stacklevel=2,
                )
                self._students = None
        self.rng = rng if rng is not None else random
        # Baseline codes: what every agent samples from outside an injection window.
        self.baseline_drivers_job_code = self._validate_job_code(drivers_job_code)
        self.baseline_passengers_job_code = self._validate_job_code(passengers_job_code)
        # Active codes: what ``new_driver_view`` / ``new_passenger_view`` sample
        # from right now. They start at the baseline and are swapped at runtime
        # by ``set_job_codes`` / ``reset_job_codes`` (scenario-style injection).
        self.drivers_job_code = self.baseline_drivers_job_code
        self.passengers_job_code = self.baseline_passengers_job_code

    # -- job codes ---------------------------------------------------------
    def set_job_codes(
            self,
            drivers_job_code: Optional[str] = None,
            passengers_job_code: Optional[str] = None,
        ) -> None:
        """
        Switch the job codes new agents sample from.

        Only agents created after the call are affected: a profile is fixed for
        an agent's whole lifetime, so already-created views keep theirs.  A
        ``None`` argument leaves that role untouched.

        Raises ``ValueError`` if a code is unknown (see ``_validate_job_code``).
        """
        if drivers_job_code is not None:
            self.drivers_job_code = self._validate_job_code(drivers_job_code)
        if passengers_job_code is not None:
            self.passengers_job_code = self._validate_job_code(passengers_job_code)

    def reset_job_codes(self) -> None:
        """Restore the baseline job codes given at construction."""
        self.drivers_job_code = self.baseline_drivers_job_code
        self.passengers_job_code = self.baseline_passengers_job_code

    def is_injected(self) -> bool:
        """True while at least one role samples from a non-baseline job code."""
        return (self.drivers_job_code != self.baseline_drivers_job_code
                or self.passengers_job_code != self.baseline_passengers_job_code)

    def available_job_codes(self) -> List[str]:
        """Every code this model accepts, population and students included."""
        codes = [self.POPULATION_JOB_CODE]
        if self._students is not None:
            codes.append(self.STUDENTS_JOB_CODE)
        return codes + sorted(self._jobs_by_code)

    def _validate_job_code(self, job_code: Any) -> str:
        """Normalize a job code to ``str`` and check it against the CSV."""
        code = str(job_code)
        if code == self.STUDENTS_JOB_CODE:
            if self._students is None:
                raise ValueError(
                    f"job_code '{code}' requires a students CSV "
                    f"(students_csv_path not provided or file invalid); "
                    f"available codes: {self.available_job_codes()}"
                )
            return code
        if code != self.POPULATION_JOB_CODE and code not in self._jobs_by_code:
            raise ValueError(
                f"job_code '{code}' not found in the Big Five jobs CSV "
                f"(use '{self.POPULATION_JOB_CODE}' for the general population, "
                f"'{self.STUDENTS_JOB_CODE}' for the students distribution); "
                f"available codes: {self.available_job_codes()}"
            )
        return code

    # -- CSV ---------------------------------------------------------------
    @staticmethod
    def _load_jobs(csv_path: str) -> List[Dict[str, Any]]:
        """
        Load per-job Big Five means/SDs.

        Expected columns: ``Code``, ``Job``, then ``<Domain> (M)`` and
        ``<Domain> (SD)`` for each of the five domains.  A malformed row
        (missing column, non-numeric value) is skipped with a warning naming the
        row and never aborts the load; a file with no usable row raises
        ``ValueError``.

        Returns one record per job: ``{"code", "job", "<domain>": (mean, sd)}``.
        """
        jobs: List[Dict[str, Any]] = []
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for line_number, row in enumerate(reader, start=2):
                try:
                    job: Dict[str, Any] = {"code": str(row["Code"]).strip(),
                                           "job": str(row["Job"]).strip()}
                    if not job["code"]:
                        raise ValueError("empty Code")
                    for domain, column in _DOMAIN_COLUMNS.items():
                        job[domain] = (float(row[f"{column} (M)"]),
                                       float(row[f"{column} (SD)"]))
                except (KeyError, TypeError, ValueError, AttributeError) as exc:
                    warnings.warn(
                        f"{csv_path}: skipping malformed row {line_number} "
                        f"({dict(row) if row else row}): {exc}",
                        stacklevel=2,
                    )
                    continue
                jobs.append(job)
        if not jobs:
            raise ValueError(f"No valid job records found in {csv_path}")
        return jobs

    # -- sampling ----------------------------------------------------------
    def _sample_domains(self, means_sds: Dict[str, Any],
                        scale: SourceScale) -> Dict[str, float]:
        """Gauss(M, SD) per domain on the source scale, mapped into [0, 1]."""
        domains: Dict[str, float] = {}
        for domain in _DOMAIN_COLUMNS:  # fixed order: draws stay reproducible
            mean, sd = means_sds[domain]
            domains[domain] = _map_source_to_unit(self.rng.gauss(mean, sd), scale)
        return domains

    def _sample_facets(self, domains: Dict[str, float],
                       unit_sds: Dict[str, float]) -> Dict[str, float]:
        """
        Jitter each facet around its parent domain (the CSVs carry no facets).

        The jitter SD is derived per domain from that domain's own SD, so the
        within-domain facet intercorrelation lands on
        ``FACET_DOMAIN_CORRELATION`` whatever scale the source data uses.
        """
        facets: Dict[str, float] = {}
        for domain, facet_names in DOMAIN_FACETS.items():
            jitter = _facet_jitter_sd(unit_sds.get(domain, 0.0))
            for facet in facet_names:
                facets[facet] = clip(self.rng.gauss(domains[domain], jitter), 0.0, 1.0)
        return facets

    def sample_profile(self, actor: str, job_code: str = "0") -> Dict[str, Any]:
        """
        Sample one agent's personality.

        Returns a plain dict with the five domains and fifteen facets in
        ``[0, 1]``, plus ``job`` and ``code`` (which ``OceanProfile.from_json``
        ignores).
        """
        code = self._validate_job_code(job_code)
        if code == self.STUDENTS_JOB_CODE:
            source: Dict[str, Any] = self._students  # type: ignore[assignment]
            label, scale = source["job"], LIKERT_SCALE
        elif code == self.POPULATION_JOB_CODE:
            if actor == "driver":
                # 37% of drivers are professional drivers
                if random.random() < self.PROFESSIONAL_DRIVERS_SHARE:
                    # Guarded lookup: a jobs CSV without 8322 (the self-test fixture,
                    # or any reduced CSV) must fall back to the population profile
                    # rather than raise KeyError mid-run. ``code`` is reassigned only
                    # on a hit, because it is returned in the profile as
                    # {"job": label, "code": code}.
                    professional = self._jobs_by_code.get("8322")
                    if professional is not None:
                        code = "8322"
                        source = professional
                        label, scale = source["job"], JOBS_SOURCE_SCALE
                    else:
                        source = dict(_POPULATION_BIG5)
                        label, scale = "population", DEFAULT_SOURCE_SCALE
                else:
                    source = dict(_POPULATION_BIG5)
                    label, scale = "population", DEFAULT_SOURCE_SCALE
            else:
                source = dict(_POPULATION_BIG5)
                label, scale = "population", DEFAULT_SOURCE_SCALE
        else:
            source = self._jobs_by_code[code]
            label, scale = source["job"], JOBS_SOURCE_SCALE
        domains = self._sample_domains(source, scale)
        # Source SDs expressed on the [0, 1] profile scale, so the facet jitter
        # can be derived from them (see _facet_jitter_sd).
        span = scale.high - scale.low
        unit_sds = {domain: abs(float(source[domain][1])) / span
                    for domain in _DOMAIN_COLUMNS}
        profile: Dict[str, Any] = {"job": label, "code": code}
        profile.update(domains)
        profile.update(self._sample_facets(domains, unit_sds))
        return profile

    def get_job_name(self, job_code: str) -> Optional[str]:
        """Human-readable name for a job code, or ``None`` if unknown."""
        code = str(job_code)
        if code == self.STUDENTS_JOB_CODE:
            return self._students["job"] if self._students else None
        if code == self.POPULATION_JOB_CODE:
            return "general"
        job = self._jobs_by_code.get(code)
        return job["job"] if job else None

    # -- views -------------------------------------------------------------
    def new_driver_view(self) -> DriverView:
        """Sample a profile (drivers' configured job code) as a ``DriverView``."""
        return DriverView(self.sample_profile(actor="driver", job_code=self.drivers_job_code))

    def new_passenger_view(self) -> PassengerView:
        """Sample a profile (passengers' configured job code) as a ``PassengerView``."""
        return PassengerView(self.sample_profile(actor="passenger", job_code=self.passengers_job_code))


# ==========================================================================
# 12. SELF-TEST
# ==========================================================================
#
# Every check cites the invariant it enforces (section 12 of the specification):
#   1 clipping             2 facet defaulting      3 coverage
#   4 no zeros             5 Mode A sums           6 complementarity
#   7 Mode B range         8 bounds by construction
#   9 monotonicity and discrimination             10 right skew
#  11 determinism         12 purity               13 serializability
#  14 error clarity

_TEST_JOBS_CSV = """Code,Job,Openness (M),Openness (SD),Conscientiousness (M),Conscientiousness (SD),Extraversion (M),Extraversion (SD),Agreeableness (M),Agreeableness (SD),Neuroticism (M),Neuroticism (SD)
11,Software developer,54.1,9.8,52.3,9.1,46.9,10.2,48.5,9.6,49.1,10.4
12,Nurse,49.6,9.4,55.2,8.7,52.8,9.9,57.1,8.8,47.3,10.1
13,Sales manager,50.2,9.9,53.4,9.3,58.7,9.4,47.2,9.7,45.8,10.0
8322,"Car, Taxi and Van Drivers",44.85,9.42,49.65,9.64,50.61,9.07,49.35,10.2,51.54,8.7
"""

_TEST_STUDENTS_CSV = """Code,Job,Openness (M),Openness (SD),Conscientiousness (M),Conscientiousness (SD),Extraversion (M),Extraversion (SD),Agreeableness (M),Agreeableness (SD),Neuroticism (M),Neuroticism (SD)
s1,students,3.66,0.66,3.42,0.70,3.29,0.79,3.71,0.63,2.94,0.87
"""

_TEST_BAD_CSV = """Code,Job,Openness (M),Openness (SD),Conscientiousness (M),Conscientiousness (SD),Extraversion (M),Extraversion (SD),Agreeableness (M),Agreeableness (SD),Neuroticism (M),Neuroticism (SD)
21,Fine row,50.0,10.0,50.0,10.0,50.0,10.0,50.0,10.0,50.0,10.0
22,Non numeric,fifty,10.0,50.0,10.0,50.0,10.0,50.0,10.0,50.0,10.0
23,Short row,50.0,10.0
,No code,50.0,10.0,50.0,10.0,50.0,10.0,50.0,10.0,50.0,10.0
"""

_FULL_CONTEXT = {
    "traffic_level": "high",
    "surge_multiplier": "mid",
    "route_length": "high",
    "time_waiting_pass": "mid",
    "time_waiting_driv": "high",
    "time_waiting_pickup": "mid",
    "time_estimated_pickup": "high",
    "time_estimated_ride": "mid",
    "provider": "uber",
    "time_of_day": "night",
    "passenger_rating": "2-3",
    "driver_rating": "3-4",
    "shift_time": "high",
    "passenger_age": "16-24",
    "driver_age": "61-70",
    "zone_requested": "downtown",
}


def _probe_contexts() -> List[Dict[str, str]]:
    """Empty, partial and saturated contexts, plus one of every extreme value."""
    contexts: List[Dict[str, str]] = [
        {},
        {"traffic_level": "low", "unknown_key": "ignored"},
        dict(_FULL_CONTEXT),
        {factor: values[0] for factor, values in CONTEXT_VALUES.items()},
        {factor: values[-1] for factor, values in CONTEXT_VALUES.items()},
    ]
    for factor, values in CONTEXT_VALUES.items():
        for value in values:
            contexts.append({factor: value})
    return contexts


#: Every corner of the five-domain personality cube, plus the midpoint: 33
#: profiles spanning the whole input space.  This replaced the 16 MBTI presets
#: as the discrimination probe when the MBTI bridge was dropped - the corners are
#: a strictly stronger probe, since the presets all sat well inside the cube.
ARCHETYPE_PROFILES: List[OceanProfile] = [
    OceanProfile(*corner)
    for corner in itertools.product((0.0, 1.0), repeat=5)
] + [OceanProfile(0.5, 0.5, 0.5, 0.5, 0.5)]


def _archetype_views(role: str) -> List[_AgentView]:
    """One view per archetype, built once: constructing a view is not cheap."""
    factory = DriverView if role == "driver" else PassengerView
    return [factory(profile) for profile in ARCHETYPE_PROFILES]


def _probe_profiles() -> List[OceanProfile]:
    """The archetypes (every cube corner plus the midpoint) and a seeded spread."""
    profiles = list(ARCHETYPE_PROFILES)
    rng = random.Random(20240901)
    for _ in range(8):
        payload = {domain: rng.random() for domain in _ALL_DOMAIN_NAMES}
        payload.update({facet: rng.random() for facet in _ALL_FACET_NAMES})
        payload["id"] = "unrelated"
        profiles.append(OceanProfile.from_json(payload))
    return profiles


#: Corners of the personality cube: the extremes any pairing can present.
_EDGE_PROFILES: List[OceanProfile] = [
    OceanProfile(0.0, 0.0, 0.0, 0.0, 0.0),
    OceanProfile(1.0, 1.0, 1.0, 1.0, 1.0),
    OceanProfile(0.5, 0.5, 0.5, 0.5, 0.5),
    OceanProfile(0.0, 1.0, 0.0, 1.0, 0.0),
    OceanProfile(1.0, 0.0, 1.0, 0.0, 1.0),
]


def _population_profiles(count: int = 240, seed: int = 424242) -> List[OceanProfile]:
    """
    A representative population: domains drawn from the global Big Five
    distribution, facets jittered to ``FACET_DOMAIN_CORRELATION`` around them.

    ``ARCHETYPE_PROFILES`` deliberately holds the extremes, so half of it is
    worst-case by construction and its median says nothing about what a real
    population does.  Claims about the *shape* of a distribution (invariant 10's
    right skew) are therefore checked here, and claims about reachable range are
    checked on the archetypes.
    """
    rng = random.Random(seed)
    profiles: List[OceanProfile] = []
    for _ in range(count):
        payload: Dict[str, float] = {}
        for domain, (mean, sd) in _POPULATION_BIG5.items():
            unit_sd = sd / (LIKERT_SCALE.high - LIKERT_SCALE.low)
            payload[domain] = _map_source_to_unit(rng.gauss(mean, sd), LIKERT_SCALE)
            jitter = _facet_jitter_sd(unit_sd)
            for facet in DOMAIN_FACETS[domain]:
                payload[facet] = clip(rng.gauss(payload[domain], jitter))
        profiles.append(OceanProfile.from_json(payload))
    return profiles


def _views_for(profile: OceanProfile) -> Dict[str, _AgentView]:
    return {"driver": DriverView(profile), "passenger": PassengerView(profile)}


def _facet_only_profile(facet: str, value: float) -> OceanProfile:
    """All domains and facets neutral except one facet."""
    payload = {domain: 0.5 for domain in _ALL_DOMAIN_NAMES}
    payload.update({name: 0.5 for name in _ALL_FACET_NAMES})
    payload[facet] = value
    return OceanProfile.from_json(payload)


def _test_profile_basics() -> None:
    for profile in _probe_profiles():
        for key, value in profile.as_dict().items():
            assert 0.0 <= value <= 1.0, f"[INV 1] {key}={value} outside [0, 1]"
        for role in ("driver", "passenger"):
            traits = _derive_traits(role, profile)
            assert set(traits) == set(TRAIT_NAMES[role]), \
                f"[INV 1] {role} traits {sorted(traits)} != {sorted(TRAIT_NAMES[role])}"
            for name, value in traits.items():
                if name in LEAN_TRAITS:
                    assert -1.0 <= value <= 1.0, \
                        f"[INV 1] {role}.{name}={value} outside [-1, 1]"
                else:
                    assert 0.0 <= value <= 1.0, \
                        f"[INV 1] {role}.{name}={value} outside [0, 1]"

    # Clipping of out-of-range inputs.
    wild = OceanProfile(-3.0, 4.0, 0.5, 0.5, 0.5, intellectual_curiosity=9.0)
    assert wild.openness == 0.0 and wild.conscientiousness == 1.0, "[INV 1] domain clipping"
    assert wild.intellectual_curiosity == 1.0, "[INV 1] facet clipping"

    # Facet defaulting and tolerant/strict from_json.
    partial = OceanProfile(0.3, 0.4, 0.5, 0.6, 0.7, organization=0.9)
    for domain, facets in DOMAIN_FACETS.items():
        for facet in facets:
            if facet == "organization":
                continue
            assert getattr(partial, facet) == getattr(partial, domain), \
                f"[INV 2] facet {facet} should default to domain {domain}"
    assert partial.organization == 0.9, "[INV 2] explicit facet must survive"

    record = {"id": 17, "job_code": "12", "name": "agent-a", "openness": 0.7,
              "conscientiousness": 0.2, "extraversion": 0.4, "agreeableness": 0.6,
              "neuroticism": 0.8, "anxiety": 0.9}
    reference = copy.deepcopy(record)
    tolerant = OceanProfile.from_json(record)
    assert tolerant.openness == 0.7 and tolerant.anxiety == 0.9, \
        "[INV 2] from_json must read recognized keys and ignore the rest"
    assert record == reference, "[INV 12] from_json must not mutate its input"

    try:
        OceanProfile.from_json({**record, "openness": "tall"})
    except ValueError as exc:
        assert "openness" in str(exc), f"[INV 14] message must name the key: {exc}"
    else:
        raise AssertionError("[INV 2] non-numeric recognized key must raise ValueError")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sparse = OceanProfile.from_json({"openness": 0.9, "sociability": 0.8})
        assert sparse.extraversion == 0.8, \
            "[INV 2] a domain omitted but faceted is recovered from its facets"
        assert sparse.conscientiousness == NEUTRAL_DOMAIN, \
            "[INV 2] a domain described by nothing falls back to neutral"
        assert any("conscientiousness" in str(w.message) for w in caught), \
            "[INV 2] the fallback must warn and name the domain"

    facets = {name: 0.2 + 0.05 * i for i, name in enumerate(_ALL_FACET_NAMES)}
    from_facets = OceanProfile.from_facets(facets)
    for domain, names in DOMAIN_FACETS.items():
        expected = sum(facets[n] for n in names) / 3.0
        assert abs(getattr(from_facets, domain) - expected) < 1e-12, \
            f"[INV 2] from_facets: {domain} must be the mean of its facets"
    try:
        OceanProfile.from_facets({"anxiety": 0.5})
    except ValueError as exc:
        assert "openness" in str(exc), f"[INV 14] message must name the domain: {exc}"
    else:
        raise AssertionError("[INV 2] from_facets must reject a domain with no facets")

    # profile_id is stable, process-independent and discriminating.
    ids = {p.profile_id for p in _probe_profiles()}
    assert len(ids) == len(_probe_profiles()), "[INV 9] profile_id must not collide on probes"
    assert OceanProfile(0.5, 0.5, 0.5, 0.5, 0.5).profile_id == \
        OceanProfile(0.5, 0.5, 0.5, 0.5, 0.5).profile_id, "[INV 11] profile_id must be stable"


def _test_tables() -> None:
    for profile in _probe_profiles():
        for role, view in _views_for(profile).items():
            for phase in MODE_A_PHASES[role]:
                tables = view.tables(phase)
                base, weights = tables["base"], tables["weights"]
                actions = PHASE_ACTIONS[(role, phase)]
                factors = PHASE_FACTORS[(role, phase)]

                assert set(base) == set(actions), \
                    f"[INV 5] ({role}, {phase}) base actions {sorted(base)} != {sorted(actions)}"
                assert abs(sum(base.values()) - 1.0) <= 1e-9, \
                    f"[INV 5] ({role}, {phase}) base sums to {sum(base.values())}"
                assert set(weights) == set(actions), \
                    f"[INV 3] ({role}, {phase}) weights cover {sorted(weights)} != {sorted(actions)}"
                for action in actions:
                    assert set(weights[action]) == set(factors), \
                        (f"[INV 3] ({role}, {phase}) action '{action}' factors "
                         f"{sorted(weights[action])} != {sorted(factors)}")
                    for factor in factors:
                        assert set(weights[action][factor]) == set(CONTEXT_VALUES[factor]), \
                            (f"[INV 3] ({role}, {phase}) '{action}'/'{factor}' values "
                             f"{sorted(weights[action][factor])} != {CONTEXT_VALUES[factor]}")
                        for value, weight in weights[action][factor].items():
                            assert weight != 0.0, \
                                f"[INV 4] ({role}, {phase}) '{action}'/'{factor}'/'{value}' is 0.0"
                for factor in factors:
                    for value in CONTEXT_VALUES[factor]:
                        column = sum(weights[a][factor][value] for a in actions)
                        assert abs(column) <= 1e-6, \
                            (f"[INV 6] ({role}, {phase}) '{factor}'/'{value}' weights sum "
                             f"to {column}, expected ~0")

            ride = view.tables("ride")
            ride_factors = PHASE_FACTORS[(role, "ride")]
            assert set(ride["weights"]) == set(RIDE_TABLE_KEYS), \
                f"[INV 3] ({role}, ride) weights must be {RIDE_TABLE_KEYS}"
            for table_name in RIDE_TABLE_KEYS:
                table = ride["weights"][table_name]
                assert set(table) == set(ride_factors), \
                    (f"[INV 3] ({role}, ride) {table_name} factors {sorted(table)} != "
                     f"{sorted(ride_factors)}")
                for factor in ride_factors:
                    assert set(table[factor]) == set(CONTEXT_VALUES[factor]), \
                        f"[INV 3] ({role}, ride) {table_name}/'{factor}' misses values"
                    for value, weight in table[factor].items():
                        assert weight != 0.0, \
                            f"[INV 4] ({role}, ride) {table_name}/'{factor}'/'{value}' is 0.0"
            assert 1.0 <= ride["base"]["rate_base"] <= 5.0, "[INV 7] rate_base out of band"
            assert 0.0 <= ride["base"]["no_vote_base"] <= 1.0, "[INV 7] no_vote_base out of band"

            # tables() hands out copies, never the live tables.
            snapshot = view.tables("request" if role == "passenger" else "start")
            some_action = next(iter(snapshot["weights"]))
            some_factor = next(iter(snapshot["weights"][some_action]))
            some_value = next(iter(snapshot["weights"][some_action][some_factor]))
            snapshot["weights"][some_action][some_factor][some_value] = 99.0
            fresh = view.tables("request" if role == "passenger" else "start")
            assert fresh["weights"][some_action][some_factor][some_value] != 99.0, \
                "[INV 12] tables() must return a deep copy"


def _test_mode_a_runtime() -> None:
    # The worked check from section 4.1.
    probs = evaluate_mode_a(
        {"accept": 0.6, "reject": 0.4},
        {"accept": {"traffic_level": {"low": 0.10}},
         "reject": {"traffic_level": {"low": -0.10}}},
        {"traffic_level": "low"},
    )
    assert abs(probs["accept"] - 0.70) < 1e-12 and abs(probs["reject"] - 0.30) < 1e-12, \
        f"[INV 5] worked check failed: {probs}"

    for profile in _probe_profiles()[:8]:
        for role, view in _views_for(profile).items():
            for phase in MODE_A_PHASES[role]:
                actions = PHASE_ACTIONS[(role, phase)]
                for context in _probe_contexts():
                    result = getattr(view, phase)(context)
                    assert set(result) == set(actions), \
                        f"[INV 5] ({role}, {phase}) returned {sorted(result)} != {sorted(actions)}"
                    assert abs(sum(result.values()) - 1.0) <= 1e-9, \
                        f"[INV 5] ({role}, {phase}) probabilities sum to {sum(result.values())}"
                    for action, value in result.items():
                        assert 0.0 <= value <= 1.0, \
                            f"[INV 5] ({role}, {phase}) '{action}' probability {value}"
                # A missing factor contributes nothing and must not raise.
                assert view._evaluate(phase, None) == view._evaluate(phase, {}), \
                    f"[INV 5] ({role}, {phase}) empty and absent context must agree"


def _test_mode_b() -> None:
    rng = random.Random(5)
    for profile in _probe_profiles():
        driver = DriverView(profile)
        passenger = PassengerView(profile)
        for view, counterpart, method in (
                (driver, passenger, "rate_passenger"),
                (passenger, driver, "rate_driver")):
            role = view.role
            # Invariant 8: worst-case abstention sums, before any clipping.
            no_vote = view.tables("ride")["weights"]["no_vote_weights"]
            reach_up = sum(max(v.values()) for v in no_vote.values())
            reach_down = sum(min(v.values()) for v in no_vote.values())
            # Bases are clipped into NO_VOTE_BASE_BAND by construction, so the
            # band's edges are the worst case over every possible counterpart.
            candidate_bases = list(NO_VOTE_BASE_BAND)
            for other in _EDGE_PROFILES:
                other_traits = _derive_traits(_COUNTERPART_ROLE[role], other)
                candidate_bases.append(view._ride_bases(other_traits)[1])
            for no_vote_base in candidate_bases:
                assert 0.0 <= no_vote_base + reach_up <= 1.0, \
                    (f"[INV 8] ({role}) no_vote_base {no_vote_base} + max reach {reach_up} "
                     f"leaves [0, 1]")
                assert 0.0 <= no_vote_base + reach_down <= 1.0, \
                    (f"[INV 8] ({role}) no_vote_base {no_vote_base} + min reach {reach_down} "
                     f"leaves [0, 1]")
            for context in _probe_contexts():
                for _ in range(3):
                    rating, abstained = getattr(view, method)(counterpart, context, rng)
                    if abstained:
                        assert rating is None, "[INV 7] abstention must return (None, True)"
                    else:
                        assert rating is not None and 1.0 <= rating <= 5.0, \
                            f"[INV 7] rating {rating} outside [1, 5]"
                p_abstain, rating = view.resolve_ride(counterpart, context)
                assert 0.0 <= p_abstain <= 1.0 and 1.0 <= rating <= 5.0, \
                    "[INV 7] deterministic resolution out of range"

    # Exactly one rng draw per rate call, whatever the outcome.
    driver = DriverView(OceanProfile(0.35, 0.75, 0.80, 0.85, 0.30))
    passenger = PassengerView(OceanProfile(0.85, 0.60, 0.25, 0.35, 0.55))
    for context in ({}, dict(_FULL_CONTEXT)):
        probe = random.Random(99)
        reference = random.Random(99)
        driver.rate_passenger(passenger, context, probe)
        reference.random()
        assert probe.random() == reference.random(), \
            "[INV 11] rate_* must draw exactly one value per call"

    # Invariant 10 has two halves, and they need different probe sets.
    # Range: every archetype pairing, extremes included, must stay in [1, 5].
    expected_pairings = len(ARCHETYPE_PROFILES) ** 2
    for role in ("driver", "passenger"):
        bases: List[float] = []
        counterpart_traits = [_derive_traits(_COUNTERPART_ROLE[role], profile)
                              for profile in ARCHETYPE_PROFILES]
        for view in _archetype_views(role):
            for other_traits in counterpart_traits:
                rate_base, _ = view._ride_bases(other_traits)
                bases.append(rate_base)
        assert len(bases) == expected_pairings, \
            f"[INV 10] expected {expected_pairings} pairings, got {len(bases)}"
        assert all(1.0 <= b <= 5.0 for b in bases), "[INV 10] rate_base outside [1, 5]"

    # Right skew: measured on a representative population, not on the extremes.
    population = _population_profiles()
    for role in ("driver", "passenger"):
        factory = DriverView if role == "driver" else PassengerView
        counterpart_traits = [_derive_traits(_COUNTERPART_ROLE[role], profile)
                              for profile in population[:40]]
        bases = []
        for profile in population[:40]:
            view = factory(profile)
            for other_traits in counterpart_traits:
                bases.append(view._ride_bases(other_traits)[0])
        median = statistics.median(bases)
        assert median >= 4.5, \
            f"[INV 10] median population {role} rate_base {median:.3f} < 4.5"

    # A rating below 4 stays reachable through negative context.
    harsh_context = {"traffic_level": "high", "provider": "lyft", "time_of_day": "night",
                     "shift_time": "high", "driver_age": "71+", "passenger_age": "55-64",
                     "zone_requested": "downtown"}
    reachable = {"driver": False, "passenger": False}
    drivers = _archetype_views("driver")
    passengers = _archetype_views("passenger")
    for drv in drivers:
        for pax in passengers:
            if drv.resolve_ride(pax, harsh_context)[1] < 4.0:
                reachable["driver"] = True
            if pax.resolve_ride(drv, harsh_context)[1] < 4.0:
                reachable["passenger"] = True
    assert all(reachable.values()), \
        f"[INV 10] a sub-4 rating must stay reachable for both roles, got {reachable}"


_FIXED_CONTEXT = {
    "traffic_level": "mid", "surge_multiplier": "high", "route_length": "mid",
    "time_waiting_pass": "mid", "time_waiting_driv": "mid", "time_waiting_pickup": "mid",
    "time_estimated_pickup": "mid", "time_estimated_ride": "mid", "provider": "uber",
    "time_of_day": "morning", "passenger_rating": "3-4", "driver_rating": "3-4",
    "shift_time": "mid", "passenger_age": "25-34", "driver_age": "31-40",
    "zone_requested": "midtown",
}


def _test_monotonicity_and_discrimination() -> None:
    # Every sign in the specification's trait table, checked one facet at a time.
    for role, traits_spec in SPEC_TRAIT_SIGNS.items():
        for trait, facet_signs in traits_spec.items():
            for facet, sign in facet_signs.items():
                low = _derive_traits(role, _facet_only_profile(facet, 0.2))[trait]
                high = _derive_traits(role, _facet_only_profile(facet, 0.8))[trait]
                delta = (high - low) * sign
                assert delta > 1e-9, (
                    f"[INV 9] {role}.{trait} must move "
                    f"{'up' if sign > 0 else 'down'} with {facet} (low={low:.4f}, high={high:.4f})"
                )

    # Every one of the fifteen facets influences at least one trait, per role.
    for role in ("driver", "passenger"):
        for facet in _ALL_FACET_NAMES:
            low = _derive_traits(role, _facet_only_profile(facet, 0.2))
            high = _derive_traits(role, _facet_only_profile(facet, 0.8))
            moved = [t for t in TRAIT_NAMES[role] if abs(high[t] - low[t]) > 1e-9]
            assert moved, f"[INV 9] facet '{facet}' influences no {role} trait"

    # Different personalities must behave measurably differently.
    biggest = 0.0
    for role in ("driver", "passenger"):
        for phase in MODE_A_PHASES[role]:
            spreads = []
            views = _archetype_views(role)
            for action in PHASE_ACTIONS[(role, phase)]:
                values = [view._evaluate(phase, _FIXED_CONTEXT)[action] for view in views]
                spreads.append(max(values) - min(values))
            biggest = max(biggest, max(spreads))
            assert max(spreads) >= 0.05, (
                f"[INV 9] ({role}, {phase}) is too flat: widest action spread across the "
                f"{len(ARCHETYPE_PROFILES)} archetypes is {max(spreads):.4f} < 0.05"
            )
    assert biggest >= 0.05, "[INV 9] no phase discriminates between archetypes"

    # No two archetypes produce the same set of tables.  With the JSON-config
    # facade gone, the fingerprint is taken over the inspection tables themselves.
    for role in ("driver", "passenger"):
        fingerprints: Dict[str, int] = {}
        for index, view in enumerate(_archetype_views(role)):
            payload = json.dumps({phase: view.tables(phase) for phase in view.phases()},
                                 sort_keys=True, default=float)
            assert payload not in fingerprints, (
                f"[INV 9] {role} archetypes {fingerprints.get(payload)} and {index} "
                f"produce identical tables"
            )
            fingerprints[payload] = index


def _ride_context_mix(rng: random.Random) -> Dict[str, str]:
    """One plausible ride: most factors unremarkable, some at an extreme."""
    context: Dict[str, str] = {}
    for factor, values in CONTEXT_VALUES.items():
        if values == LEVELS:
            context[factor] = rng.choices(values, [0.2, 0.6, 0.2])[0]
        else:
            context[factor] = rng.choice(values)
    return context


def _test_dispersion_and_realism() -> None:
    """
    Guards the properties the model was re-tuned for, all of which had measurably
    failed before: agents were nearly interchangeable, facets were near-redundant,
    and no rating could ever land below four stars.
    """
    # --- PERSON_GAIN: neutral agent is a fixed point, deviations scale exactly ---
    for role, gain in PERSON_GAIN.items():
        assert gain >= 1.0, f"a gain below 1.0 would flatten {role}s, not widen them"

        def shares(tr: TraitView) -> Dict[str, float]:
            return {"a": 0.30 + 0.40 * tr.patience, "b": 0.70 - 0.40 * tr.patience}

        neutral = _widen_base(shares, role, NEUTRAL_TRAITS[role])
        reference = shares(NEUTRAL_TRAITS[role])
        for action, value in neutral.items():
            assert abs(value - reference[action]) < 1e-12, \
                f"_widen_base must leave the neutral {role} untouched ({action})"
        probe = _derive_traits(role, OceanProfile(0.8, 0.2, 0.7, 0.9, 0.1))
        widened = _widen_base(shares, role, probe)
        raw = shares(probe)
        for action in raw:
            anchor = max(reference[action], BASE_SHARE_FLOOR)
            ratio = max(raw[action], BASE_SHARE_FLOOR) / anchor
            expected = clip(anchor * ratio ** gain, BASE_SHARE_FLOOR, BASE_SHARE_CEIL)
            assert abs(widened[action] - expected) < 1e-12, \
                f"_widen_base must raise the {role} share ratio to exactly {gain} ({action})"
            assert widened[action] > 0.0, \
                f"_widen_base must never zero an action ({role}, {action})"

    # --- synchronised factors really are damped ---------------------------------
    assert 0.0 < MACRO_FACTOR_DAMP < 1.0, "MACRO_FACTOR_DAMP must damp, not amplify"
    assert FLEET_SHARED_FACTORS <= set(CONTEXT_VALUES), \
        "FLEET_SHARED_FACTORS must name real context factors"

    # --- between-agent dispersion: the original defect was 0.05-0.10 ------------
    population = _population_profiles(160, seed=97)
    drivers = [DriverView(p) for p in population]
    passengers = [PassengerView(p) for p in population]
    for role, views in (("driver", drivers), ("passenger", passengers)):
        for phase in MODE_A_PHASES[role]:
            for action in PHASE_ACTIONS[(role, phase)]:
                values = [v._evaluate(phase, _FIXED_CONTEXT)[action] for v in views]
                spread = statistics.pstdev(values)
                if action == PHASE_ACTIONS[(role, phase)][0]:
                    # 0.075 sits deliberately between the two measured regimes: on
                    # this probe the pre-PERSON_GAIN model spanned 0.060-0.108 and
                    # the tuned one spans 0.094-0.198, so the guard fires if the
                    # model ever drifts back toward interchangeable agents.
                    assert spread >= 0.075, (
                        f"[INV 9] ({role}, {phase}) '{action}' between-agent SD "
                        f"{spread:.4f} < 0.075: the population is too interchangeable "
                        f"for personality to show up in aggregate output"
                    )

    # --- no action may be extinguished for the median agent ---------------------
    # A long right tail in a minority action is realistic heterogeneity; a median
    # agent that never takes the action at all is a tuning failure, and is exactly
    # what additive widening produced (see the comment on PERSON_GAIN).
    for role, views in (("driver", drivers), ("passenger", passengers)):
        for phase in MODE_A_PHASES[role]:
            for action in PHASE_ACTIONS[(role, phase)]:
                probabilities = sorted(v._evaluate(phase, _FIXED_CONTEXT)[action]
                                       for v in views)
                median = statistics.median(probabilities)
                assert median >= 0.02, (
                    f"[INV 9] ({role}, {phase}) median P('{action}') is {median:.4f}: "
                    f"the typical agent never takes this action, so the metric it "
                    f"feeds goes dead"
                )

    # --- facets must not be near-copies of their domain -------------------------
    samples = _population_profiles(600, seed=31)
    correlations: List[float] = []
    for domain, facet_names in DOMAIN_FACETS.items():
        for first, second in itertools.combinations(facet_names, 2):
            xs = [getattr(p, first) for p in samples]
            ys = [getattr(p, second) for p in samples]
            correlations.append(statistics.correlation(xs, ys))
    mean_r = statistics.fmean(correlations)
    assert 0.35 <= mean_r <= 0.75, (
        f"within-domain facet correlation {mean_r:.3f} outside the 0.35-0.75 band "
        f"BFI-2 reports (0.43-0.67); near-1.0 facets carry no information the domain "
        f"does not already carry"
    )

    # --- ratings: whole stars, right-skewed, and a bad rating must be possible ---
    rng = random.Random(4242)
    contexts = [_ride_context_mix(rng) for _ in range(120)]
    for role, raters, others, method in (("passenger", passengers, drivers, "rate_driver"),
                                         ("driver", drivers, passengers, "rate_passenger")):
        stars: List[float] = []
        for index in range(2400):
            rating, abstained = getattr(raters[index % len(raters)], method)(
                others[(index * 7) % len(others)], contexts[index % len(contexts)], rng)
            if abstained:
                assert rating is None, "[INV 7] abstention must return (None, True)"
                continue
            assert rating is not None and 1.0 <= rating <= 5.0, \
                f"[INV 7] rating {rating} outside [1, 5]"
            if INTEGER_STARS:
                assert rating == float(int(rating)), \
                    f"[INV 7] INTEGER_STARS is on, so {rating} must be a whole star"
            stars.append(rating)
        assert stars, f"[INV 10] {role} produced no ratings at all"
        top = sum(1 for s in stars if s >= 5.0) / len(stars)
        low = sum(1 for s in stars if s < 4.0) / len(stars)
        mean_stars = statistics.fmean(stars)
        assert 0.70 <= top <= 0.95, \
            f"[INV 10] {role} five-star share {top:.3f} outside the published 0.70-0.95"
        assert 4.6 <= mean_stars <= 4.95, \
            f"[INV 10] {role} mean rating {mean_stars:.3f} outside the published 4.6-4.95"
        assert low > 0.0, (
            f"[INV 10] {role} never rates below four stars, so dissatisfaction has no "
            f"outlet except abstention - this is the regression the rating curve fixes"
        )


def _test_purity() -> None:
    profile_record = {"openness": 0.7, "conscientiousness": 0.3, "extraversion": 0.6,
                      "agreeableness": 0.4, "neuroticism": 0.8, "anxiety": 0.9,
                      "id": "keep-me"}
    reference_record = copy.deepcopy(profile_record)
    context = dict(_FULL_CONTEXT)
    reference_context = copy.deepcopy(context)

    driver = DriverView(profile_record)
    passenger = PassengerView(profile_record)
    rng = random.Random(3)
    for phase in MODE_A_PHASES["driver"]:
        getattr(driver, phase)(context)
        driver.choose(phase, context, rng)
    for phase in MODE_A_PHASES["passenger"]:
        getattr(passenger, phase)(context)
        passenger.choose(phase, context, rng)
    driver.rate_passenger(passenger, context, rng)
    passenger.rate_driver(driver, context, rng)
    for phase in driver.phases():
        driver.tables(phase)
    for phase in passenger.phases():
        passenger.tables(phase)

    assert context == reference_context, "[INV 12] a caller's context must not be mutated"
    assert profile_record == reference_record, "[INV 12] a caller's profile must not be mutated"

    # Traits are read-only.
    try:
        driver.traits["patience"] = 1.0  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("[INV 12] traits must be read-only")
    try:
        driver.profile.openness = 0.1  # type: ignore[misc]
    except AttributeError:
        pass  # dataclasses.FrozenInstanceError
    else:
        raise AssertionError("[INV 12] OceanProfile must be frozen")


def _test_errors() -> None:
    driver = DriverView(OceanProfile(0.30, 0.85, 0.75, 0.45, 0.35))
    passenger = PassengerView(OceanProfile(0.85, 0.35, 0.30, 0.80, 0.65))

    try:
        driver.request({"traffic_level": "extreme"})
    except ValueError as exc:
        message = str(exc)
        assert "traffic_level" in message and "extreme" in message, \
            f"[INV 14] message must name factor and value: {message}"
    else:
        raise AssertionError("[INV 14] an unrecognised factor value must raise ValueError")

    try:
        passenger.rate_driver(driver, {"zone_requested": "airport"})
    except ValueError as exc:
        assert "zone_requested" in str(exc) and "airport" in str(exc), \
            f"[INV 14] Mode B must validate too: {exc}"
    else:
        raise AssertionError("[INV 14] Mode B must reject an unrecognised value")

    # An unknown key is ignored, an absent factor contributes nothing.
    assert driver.request({"weather": "rain"}) == driver.request({}), \
        "[INV 5] unknown context keys must be ignored"

    # None means "could not resolve", not "bad value": the simulator produces it for
    # zone/age (Drivers._driver_zone returns None while a vehicle is off-network, and
    # the zone/age .get() lookups miss before an agent is registered).  Treating None
    # as a validation error aborts a live multi-hour run, so it must behave as absent.
    for none_factor in ("zone_requested", "driver_age", "traffic_level", "provider"):
        assert driver.request({none_factor: None}) == driver.request({}), \
            f"[INV 5] a None value for '{none_factor}' must behave as absent"
    try:
        # rate_driver samples, so assert only that None does not raise.
        passenger.rate_driver(driver, {"zone_requested": None, "passenger_age": None})
    except ValueError as exc:
        raise AssertionError(f"[INV 5] Mode B must treat None as absent too: {exc}")

    # Age vocabularies are never interchangeable.
    try:
        driver.request({"driver_age": "25-34"})
    except ValueError as exc:
        assert "driver_age" in str(exc), f"[INV 14] message must name the factor: {exc}"
    else:
        raise AssertionError("[INV 14] a passenger age bucket must not pass as a driver age")

    try:
        driver.tables("negotiate")
    except KeyError as exc:
        assert "negotiate" in str(exc), f"[INV 14] message must name the phase: {exc}"
    else:
        raise AssertionError("[INV 14] an unknown phase must raise KeyError")

    try:
        driver.choose("ride", {}, random.Random(1))
    except KeyError as exc:
        assert "ride" in str(exc), f"[INV 14] message must name the phase: {exc}"
    else:
        raise AssertionError("[INV 14] choose() must reject a Mode B phase")

    try:
        driver.rate_passenger(driver, {})
    except TypeError as exc:
        assert "passenger" in str(exc), f"[INV 14] message must name the expected role: {exc}"
    else:
        raise AssertionError("[INV 14] rating the wrong role must raise TypeError")


def _write(path: str, text: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _test_csv_and_sampling() -> None:
    global TSCORE_PERCENTILE_MAPPING
    with tempfile.TemporaryDirectory() as tmp:
        jobs_csv = _write(os.path.join(tmp, "jobs.csv"), _TEST_JOBS_CSV)
        students_csv = _write(os.path.join(tmp, "students.csv"), _TEST_STUDENTS_CSV)
        bad_csv = _write(os.path.join(tmp, "bad.csv"), _TEST_BAD_CSV)
        empty_csv = _write(os.path.join(tmp, "empty.csv"),
                           _TEST_BAD_CSV.splitlines()[0] + "\n" +
                           "\n".join(_TEST_BAD_CSV.splitlines()[2:]) + "\n")

        # Malformed rows are skipped with a warning naming the row, never fatal.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = OceanDecisionModel._load_jobs(bad_csv)
        assert [job["code"] for job in loaded] == ["21"], \
            f"[INV 14] only the well-formed row should survive, got {loaded}"
        assert len(caught) == 3, f"[INV 14] expected 3 skip warnings, got {len(caught)}"
        assert any("row 3" in str(w.message) for w in caught), \
            "[INV 14] a skip warning must name the row"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                OceanDecisionModel._load_jobs(empty_csv)
            except ValueError as exc:
                assert "No valid job records" in str(exc), f"[INV 14] {exc}"
            else:
                raise AssertionError("[INV 14] a CSV with no usable row must raise ValueError")

        # An invalid job code names itself and lists the alternatives.
        try:
            OceanDecisionModel(jobs_csv, drivers_job_code="999")
        except ValueError as exc:
            assert "999" in str(exc) and "11" in str(exc), \
                f"[INV 14] message must name the code and list the available ones: {exc}"
        else:
            raise AssertionError("[INV 14] an unknown job code must raise ValueError")
        try:
            OceanDecisionModel(jobs_csv, passengers_job_code="students")
        except ValueError as exc:
            assert "students" in str(exc), f"[INV 14] {exc}"
        else:
            raise AssertionError("[INV 14] 'students' without a students CSV must raise")

        # The three distributions stay distinct.
        model = OceanDecisionModel(jobs_csv, drivers_job_code="0",
                                   passengers_job_code="12",
                                   students_csv_path=students_csv,
                                   rng=random.Random(1234))
        assert model.available_job_codes() == ["0", "students", "11", "12", "13", "8322"], \
            f"unexpected job codes {model.available_job_codes()}"
        assert model.get_job_name("0") == "general"
        assert model.get_job_name("students") == "students"
        assert model.get_job_name("404") is None
        for code in ("0", "students"):
            sampled = model.sample_profile(actor="passenger", job_code=code)
            assert sampled["code"] == code
            for key in _ALL_PROFILE_KEYS:
                assert 0.0 <= sampled[key] <= 1.0, \
                    f"[INV 1] sampled {key}={sampled[key]} outside [0, 1] for code {code}"
            profile = OceanProfile.from_json(sampled)
            assert set(profile.as_dict()) == set(_ALL_PROFILE_KEYS), \
                "a sampled record must round-trip into a complete profile"

        # Source-scale mapping, including the documented percentile escape hatch.
        assert _map_source_to_unit(1.0, LIKERT_SCALE) == 0.0
        assert _map_source_to_unit(5.0, LIKERT_SCALE) == 1.0
        assert abs(_map_source_to_unit(3.0, LIKERT_SCALE) - 0.5) < 1e-12
        assert abs(_map_source_to_unit(50.0, TSCORE_SCALE) - 0.5) < 1e-12
        assert _map_source_to_unit(999.0, TSCORE_SCALE) == 1.0, "draws are clipped first"
        TSCORE_PERCENTILE_MAPPING = True
        try:
            assert abs(_map_source_to_unit(60.0, TSCORE_SCALE) - 0.8413) < 1e-3, \
                "percentile mapping must use the standard normal CDF"
        finally:
            TSCORE_PERCENTILE_MAPPING = False

        # Runtime injection: only agents created inside the window sample from
        # the injected codes; the baseline is restored by reset_job_codes.
        model = OceanDecisionModel(jobs_csv, students_csv_path=students_csv, rng=random.Random(5))
        assert not model.is_injected()
        before = model.new_passenger_view()
        assert before.profile_id  # created before the switch, profile is fixed
        model.set_job_codes(passengers_job_code="students")
        assert model.is_injected()
        assert model.drivers_job_code == "0", "an untouched role keeps its code"
        assert model.sample_profile("passenger", model.passengers_job_code)["code"] == "students"
        model.set_job_codes(drivers_job_code="12")
        assert model.sample_profile("driver", model.drivers_job_code)["code"] == "12"
        try:
            model.set_job_codes(drivers_job_code="999")
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown injected job code must raise ValueError")
        assert model.drivers_job_code == "12", "a rejected switch must not change the active code"
        model.reset_job_codes()
        assert not model.is_injected()
        assert (model.drivers_job_code, model.passengers_job_code) == ("0", "0")

        # Every public surface, exercised once.
        rng = random.Random(77)
        model = OceanDecisionModel(jobs_csv, students_csv_path=students_csv, rng=rng)
        driver = model.new_driver_view()
        passenger = model.new_passenger_view()
        assert driver.phases() == ["start", "wait_requests", "request", "pickup", "ride"]
        assert passenger.phases() == ["request", "pickup", "ride"]
        for view in (driver, passenger):
            assert isinstance(view.notes, str) and 0 < len(view.notes) <= NOTES_MAX_CHARS, \
                f"notes must be one line of at most 200 chars: {view.notes!r}"
            assert isinstance(view.traits, _ABCMapping) and view.traits["patience"] == view.traits.patience
            assert view.profile_id == view.profile.profile_id
            for phase in view.phases():
                tables = view.tables(phase)
                assert set(tables) == {"base", "weights"}
            for phase in MODE_A_PHASES[view.role]:
                assert view.choose(phase, _FULL_CONTEXT, rng) in PHASE_ACTIONS[(view.role, phase)]
        assert driver.start(_FULL_CONTEXT) and driver.wait_requests(_FULL_CONTEXT)
        assert driver.request(_FULL_CONTEXT) and driver.pickup(_FULL_CONTEXT)
        assert passenger.request(_FULL_CONTEXT) and passenger.pickup(_FULL_CONTEXT)
        driver.rate_passenger(passenger, _FULL_CONTEXT, rng)
        passenger.rate_driver(driver, _FULL_CONTEXT, rng)
        # Counterparts may also be given as raw profiles or dicts.
        driver.rate_passenger(passenger.profile, _FULL_CONTEXT, rng)
        passenger.rate_driver(driver.profile.as_dict(), _FULL_CONTEXT, rng)
        # The module-level generator is only a fallback, never a substitute.
        # random.seed(4)
        driver.rate_passenger(passenger, _FULL_CONTEXT)
        driver.choose("request", _FULL_CONTEXT)

        assert derive_driver_traits(driver.profile) == driver.traits
        assert derive_passenger_traits(passenger.profile) == passenger.traits
        assert PassengerTraits(patience=0.5).patience == 0.5, "legacy trait alias must build"
        assert "compassion" in trait_rationale("driver", "patience")
        assert trait_rationale("passenger", "provider_affinity")
        assert DriverView.from_json({"openness": 0.4, "conscientiousness": 0.6,
                                     "extraversion": 0.5, "agreeableness": 0.5,
                                     "neuroticism": 0.5, "id": 3}).traits.patience >= 0.0
        assert driver.profile.dominant_domains(2) and len(driver.profile.as_dict()) == 20
        probabilities = normalize_to_probabilities({"a": -5.0, "b": 1.0})
        assert abs(sum(probabilities.values()) - 1.0) < 1e-12 and probabilities["a"] > 0.0, \
            "[INV 5] raw scores are clipped away from zero, never to zero"


def _test_determinism() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        jobs_csv = _write(os.path.join(tmp, "jobs.csv"), _TEST_JOBS_CSV)
        students_csv = _write(os.path.join(tmp, "students.csv"), _TEST_STUDENTS_CSV)

        def run(seed: int) -> str:
            model = OceanDecisionModel(jobs_csv, drivers_job_code="11",
                                       passengers_job_code="0",
                                       students_csv_path=students_csv,
                                       rng=random.Random(seed))
            out: List[Any] = []
            for _ in range(4):
                driver = model.new_driver_view()
                passenger = model.new_passenger_view()
                out.append([driver.profile.as_dict(), passenger.profile.as_dict(),
                            driver.notes, passenger.notes,
                            driver.traits.as_dict(), passenger.traits.as_dict()])
                for context in (_FIXED_CONTEXT, _FULL_CONTEXT):
                    out.append(driver.request(context))
                    out.append(passenger.pickup(context))
                    out.append(driver.choose("wait_requests", context, model.rng))
                    out.append(passenger.rate_driver(driver, context, model.rng))
                    out.append(driver.rate_passenger(passenger, context, model.rng))
                out.append(model.sample_profile(actor="passenger", job_code="students"))
                out.append({phase: driver.tables(phase) for phase in driver.phases()})
            return json.dumps(out, sort_keys=True)

        assert run(2026) == run(2026), \
            "[INV 11] equally seeded runs must produce identical output"
        assert run(2026) != run(2027), \
            "[INV 11] different seeds must produce different populations"

        # Mode A phase methods draw nothing at all.
        model = OceanDecisionModel(jobs_csv, rng=random.Random(9))
        driver = model.new_driver_view()
        probe = random.Random(123)
        state = probe.getstate()
        for phase in MODE_A_PHASES["driver"]:
            getattr(driver, phase)(_FULL_CONTEXT)
        assert probe.getstate() == state, "[INV 11] Mode A must not draw randomness"

        # Tables and notes are pure functions of the profile.
        again = DriverView(driver.profile)
        assert again.tables("request") == driver.tables("request"), \
            "[INV 11] weight tables must be a pure function of the profile"
        assert again.notes == driver.notes, "[INV 11] notes must be deterministic"
        assert again.traits.provider_affinity == driver.traits.provider_affinity, \
            "[INV 11] provider_affinity must be deterministic"


def _self_test() -> None:
    """Assert every invariant of section 12 and exercise every public method."""
    _test_profile_basics()
    _test_tables()
    _test_mode_a_runtime()
    _test_mode_b()
    _test_monotonicity_and_discrimination()
    _test_dispersion_and_realism()
    _test_purity()
    _test_errors()
    _test_csv_and_sampling()
    _test_determinism()
    print(
        f"ocean.py self-test OK - every invariant holds across "
        f"{len(_probe_profiles())} probe profiles, {len(ARCHETYPE_PROFILES)} archetypes, "
        f"{len(PHASE_MODE)} phases and {len(CONTEXT_VALUES)} context factors."
    )


if __name__ == "__main__":
    _self_test()