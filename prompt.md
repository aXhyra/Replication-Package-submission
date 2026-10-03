# Prompt: Generate the OCEAN Python Personality Model (`ocean.py`)

You are generating a complete, executable Python personality decision model for a ride-hailing
digital twin simulation. Every agent carries a continuous Big Five (OCEAN) profile, and every
decision it makes is a documented function of that profile and the current context, computed at
runtime for any profile, with no per-type configuration to maintain. (The model descends from a
static 16-type MBTI JSON configuration; that lineage is retired and must not reappear — see §8.)

The generated module is used to produce **reproducible test cases** for a digital twin. Every
number it produces must therefore be *deterministic given a seed*, *traceable to an explicit
formula*, and *inspectable as data*. Auditability is a hard requirement, not a nicety.

The module must also be **realistic at the population level**: agents must differ from one
another far more than a single agent varies from one decision to the next, and simulated ratings
must have the shape real ride-hailing ratings have. These are measurable properties, and the
self-test measures them (§12.9, §12.10, §13).

Do not generate a JSON configuration file. Generate Python code.

---

## 0. Non-negotiables

If any of these fail, the output is rejected regardless of other quality:

1. One valid, complete, executable Python file. No `...`, no `pass` stubs, no TODOs, no
   "same as above", no truncation, no code fences, no prose outside comments/docstrings.
2. Standard library only. Python ≥ 3.10 (the self-test uses `statistics.correlation`).
3. Every phase's weight tables contain **exactly** the factors listed for that phase in §5 —
   no more, no fewer — and **every** permitted value of each factor.
4. No weight is ever exactly `0.0`.
5. Mode A bases sum to `1.0`; Mode A results sum to `1.0`. Mode B is never normalized.
6. Different OCEAN profiles produce measurably different behavior, and the population-level
   dispersion and rating-shape targets hold (§12.9, §12.10).
7. No MBTI bridge and no JSON compatibility layer (§8).
8. `_self_test()` runs clean in a few seconds and asserts every invariant in §12 and every
   calibration guard in §13.

---

## 1. Required output

Output the entire contents of one Python file named `ocean.py`.

The file must:

- Include module-level documentation covering construction, simulator integration, context
  formats, and every public method (see §11).
- Include type annotations on all public APIs.
- Organize into clearly labeled, numbered sections with banner comments, in this order: context
  vocabulary and phase structure; OCEAN input model; derived operational traits; notes; generic
  Mode A/B evaluators; Mode A table builder; driver Mode A phases; passenger Mode A phases; Mode B
  ride phases; per-agent views; CSV sources, sampling and the simulator entry point; self-test.
- Keep each tuning constant (§6.5) as one named module constant with a comment stating what it
  controls and why it has its value. Inline coefficients inside builders are fine when a comment
  says what they encode.
- Include a runnable self-test:

```python
if __name__ == "__main__":
    _self_test()
```

- Require no auxiliary files other than the CSVs accepted by `OceanDecisionModel`.
- Never seed or otherwise touch the module-level `random` generator at import time. Seeding is
  the caller's job.

### 1.1 On the reference implementation

A reference `ocean.py` may be supplied alongside this prompt. If it is:

- Follow its architectural style, naming, and integration contract.
- **Never remove** a public method, constructor parameter, or default it supports — except the
  retired surface listed in §8, which must not be reintroduced even if an older reference still
  carries it.
- Where the reference and this prompt conflict on *interface*, the reference wins.
- Where they conflict on *behavior specified here*, this prompt wins. Two known cases: every draw
  in profile sampling goes through the model's rng (§9), even where a reference reaches for the
  module-level `random`; and docstrings must describe the behavior actually implemented rather
  than an earlier design (for example, a rating base confined to a 4.5–5.0 band, §7).
- Do not copy a comment or docstring whose claim the code beside it contradicts; make the two
  agree, following this prompt wherever it decides the point.

**If no reference is supplied, or the reference is silent on a point, implement the fallback
specified in this prompt.** Do not treat any requirement below as optional on the grounds that
the reference does not mention it. There are no conditional requirements in this document.

---

## 2. Required architecture

### 2.1 `OceanProfile`

A frozen dataclass representing one agent's personality. A personality never changes at runtime.

Five domains, each clipped to `[0.0, 1.0]`:
`openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`.

Fifteen optional BFI-2 facets, three per domain, each clipped to `[0.0, 1.0]`:

| Domain | Facets |
|---|---|
| Openness | `intellectual_curiosity`, `aesthetic_sensitivity`, `creative_imagination` |
| Conscientiousness | `organization`, `productiveness`, `responsibility` |
| Extraversion | `sociability`, `assertiveness`, `energy_level` |
| Agreeableness | `compassion`, `respectfulness`, `trust` |
| Neuroticism | `anxiety`, `depression`, `emotional_volatility` |

An omitted facet defaults to its parent domain value, so a domain-only profile behaves exactly
like a profile whose facets all equal their domain.

Required constructors:

```python
@classmethod
def from_json(cls, data: Dict[str, Any]) -> "OceanProfile"      # ignores unrelated keys
@classmethod
def from_facets(cls, facets: Dict[str, float]) -> "OceanProfile"  # domain = mean of its supplied facets
```

`from_json` must accept full or partial domain/facet keys, ignore unrelated simulator fields
(e.g. `id`, `job_code`, `name`, `job`, `code`) without error, treat a `None` value as absent, and
raise `ValueError` naming the key if a recognized key holds a non-numeric value. A domain that is
omitted but described by one or more of its facets is recovered as the mean of those facets. A
domain described by nothing falls back to `NEUTRAL_DOMAIN = 0.5` **and emits a warning naming
the domain**, so a mistyped key cannot pass unnoticed. A non-dict input raises `TypeError`.

`from_facets` sets each domain to the mean of the facets supplied for it and raises `ValueError`
naming the domain when none of its facets is supplied.

Also provide `as_dict()` returning a plain `dict` of all 20 values (domains first, then facets);
a stable `profile_id` property — a short deterministic hash of the rounded 20 values, computed
with `hashlib` so it is identical across processes (never the built-in `hash()`) — so generated
test cases can be labeled and de-duplicated; and `dominant_domains(count=2)` returning the domains
furthest from `0.5`, most extreme first, with a deterministic tie-break.

### 2.2 `TraitView`

The derived-trait table of one agent: an immutable object that behaves as a read-only
`collections.abc.Mapping` (`traits["patience"]`, `dict(traits)`, `in`, `len`, equality) **and**
supports attribute access (`traits.patience`). Assignment and deletion raise `TypeError`; a
missing attribute raises `AttributeError` naming it; `as_dict()` returns a plain `dict`.

Keep `PassengerTraits` and `DriverTraits` as backwards-compatible aliases of `TraitView` that
still accept keyword construction (`PassengerTraits(patience=0.5)`).

### 2.3 `DriverView`

```python
DriverView(profile: Union[OceanProfile, Dict[str, Any]])
DriverView.from_json(data) -> "DriverView"

driver.start(context=None)             -> Dict[str, float]
driver.wait_requests(context=None)     -> Dict[str, float]
driver.request(context=None)           -> Dict[str, float]
driver.pickup(context=None)            -> Dict[str, float]
driver.rate_passenger(passenger, context=None, rng=None) -> Tuple[Optional[float], bool]
```

### 2.4 `PassengerView`

```python
PassengerView(profile: Union[OceanProfile, Dict[str, Any]])
PassengerView.from_json(data) -> "PassengerView"

passenger.request(context=None)        -> Dict[str, float]
passenger.pickup(context=None)         -> Dict[str, float]
passenger.rate_driver(driver, context=None, rng=None) -> Tuple[Optional[float], bool]
```

### 2.5 Shared view requirements

- The constructor derives operational traits **once** and precomputes every personality-only
  table **once**: the Mode A bases and weights, and the Mode B `rate_weights` and
  `no_vote_weights`. Context-dependent work happens only at call time.
- Both views expose `profile`, `profile_id`, `traits` (a `TraitView`), `notes` (§3.5), `phases()`
  (the role's Mode A phases in order, then `"ride"`), and `tables(phase)`, which returns a deep
  copy of `{"base": ..., "weights": ...}` for inspection and thesis documentation. For a Mode A
  phase, `base` holds the action shares and `weights` the per-action tables. For `"ride"`, `base`
  is `{"rate_base", "no_vote_base"}` against the neutral counterpart (§3.4) and `weights` is
  `{"rate_weights", "no_vote_weights"}`. An unknown phase raises `KeyError` naming the phase and
  listing the available ones.
- Both expose `choose(phase, context=None, rng=None) -> str`, which samples one action from the
  Mode A distribution with exactly one rng draw; asking it for a Mode B phase raises `KeyError`
  naming the phase. Phase methods themselves return probabilities and never draw randomness.
- Both expose `resolve_ride(counterpart=None, context=None) -> Tuple[float, float]`, returning
  `(p_abstain, rating)` deterministically, with no draw; `counterpart=None` means the neutral
  counterpart.
- A rate call's counterpart may be a view of the opposite role, an `OceanProfile`, a plain
  profile dict, or a `TraitView`. A view of the wrong role raises `TypeError` naming the expected
  role.
- No view mutates a caller-supplied context dict or profile dict.

### 2.6 `OceanDecisionModel`

```python
OceanDecisionModel(
    jobs_csv_path,
    drivers_job_code="0",
    passengers_job_code="0",
    students_csv_path=None,
    rng=None,
)
```

Must support:

- `new_driver_view()` and `new_passenger_view()`, which sample a profile from that role's
  **active** job code.
- `sample_profile(actor, job_code="0") -> Dict[str, Any]`, where `actor` is `"driver"` or
  `"passenger"` (sampling is actor-aware, §9.1). It returns a plain dict with the 20 personality
  values plus `"job"` (a human-readable label) and `"code"` (the code actually sampled), both of
  which `OceanProfile.from_json` ignores.
- `get_job_name(job_code) -> Optional[str]`: `"general"` for `"0"`, the students record's name
  for `"students"`, the job name for a known code, `None` otherwise.
- `available_job_codes() -> List[str]`: `"0"`, then `"students"` if loaded, then the CSV codes
  sorted.
- **Runtime personality injection.** The constructor codes are the *baseline*
  (`baseline_drivers_job_code`, `baseline_passengers_job_code`); the codes new views sample from
  are the *active* ones (`drivers_job_code`, `passengers_job_code`).
  `set_job_codes(drivers_job_code=None, passengers_job_code=None)` switches the active codes like
  any other scenario injection: `None` leaves that role untouched, and an unknown code raises
  `ValueError` and leaves the active code unchanged. `reset_job_codes()` restores the baseline;
  `is_injected()` reports whether any role is off-baseline. Only agents created after a switch
  are affected: a profile is fixed for an agent's whole lifetime.
- Class constants `POPULATION_JOB_CODE = "0"`, `STUDENTS_JOB_CODE = "students"` and
  `PROFESSIONAL_DRIVERS_SHARE = 0.37`.

Job-code sampling, the population distribution (`"0"`), and the optional students distribution
stay three distinct concepts (§9).

### 2.7 Module-level functions

`clip`, `validate_context`, `normalize_to_probabilities`, `evaluate_mode_a`, `evaluate_mode_b`,
`resolve_mode_b`, `derive_driver_traits`, `derive_passenger_traits`,
`trait_rationale(role, trait)`, `profile_notes(profile)`,
`driver_ride_pair_base(driver_traits, passenger_traits)` and
`passenger_ride_pair_base(passenger_traits, driver_traits)`, each with a docstring.

---

## 3. Personality model

### 3.1 Derivation contract

Every operational trait is a documented, deterministic function of the OCEAN profile, clipped to
`[0.0, 1.0]` (the one exception is `provider_affinity`, §3.3). Traits are derived as
`clip(0.5 + Σ coefficient × (facet − 0.5))`, an auditable linear form of the facets.

Store the derivation **as data**: one model per role, mapping
`trait -> (justification, {facet: coefficient})`, where the justification is a one-line string
naming the facets used and the direction of each. Expose the models as `PASSENGER_TRAIT_MODEL`,
`DRIVER_TRAIT_MODEL`, `SHARED_TRAIT_MODEL` and `TRAIT_MODELS`, the trait names per role as
`TRAIT_NAMES`, and each justification through `trait_rationale(role, trait)`.

Do not hide trait derivation inside the weight construction code — derive traits first, then
build every weight from traits, never directly from raw domains.

### 3.2 Required traits and required signs

Coefficient magnitudes are yours to choose. **Signs are fixed** and are asserted by the self-test
(§12.9): increasing the listed facet, holding all else constant, must move the trait in the
listed direction. You may give a trait further facets where the Big Five literature supports it,
but never one that contradicts a listed sign.

Keep these tables as their own constant, `SPEC_TRAIT_SIGNS`, separate from the coefficients, so
the self-test checks the model against the specification rather than against itself.

**Passenger traits**

| Trait | Increases with | Decreases with |
|---|---|---|
| `patience` | compassion, responsibility | anxiety, emotional_volatility |
| `provider_fidelity` | organization, trust | intellectual_curiosity, creative_imagination |
| `surge_sensitivity` | anxiety | intellectual_curiosity |
| `trust_in_driver` | trust, compassion | anxiety |
| `communication_skills` | sociability, respectfulness | anxiety, depression |
| `time_budget_flexibility` | creative_imagination, compassion | organization, anxiety |

**Driver traits**

| Trait | Increases with | Decreases with |
|---|---|---|
| `patience` | compassion, responsibility | anxiety, emotional_volatility |
| `provider_fidelity` | organization, trust | intellectual_curiosity, creative_imagination |
| `surge_greediness` | assertiveness, productiveness | compassion |
| `work_urgency` | productiveness, responsibility | depression |
| `trust_in_passenger` | trust, compassion | anxiety |
| `communication_skills` | sociability, respectfulness | anxiety, depression |
| `zone_flexibility` | creative_imagination, energy_level | organization |
| `driving_style` (aggressiveness) | assertiveness, emotional_volatility | respectfulness, responsibility |
| `time_budget_flexibility` | creative_imagination | organization, anxiety |
| `money_budget_ambition` | assertiveness, productiveness | compassion |

**Shared traits** — derived for both roles from one `SHARED_TRAIT_MODEL` with identical formulas,
so a profile's reactivity and rating style do not depend on which seat it is in:

| Trait | Increases with | Decreases with |
|---|---|---|
| `volatility` | emotional_volatility, energy_level | organization, responsibility |
| `rating_discrimination` | aesthetic_sensitivity, intellectual_curiosity | respectfulness, compassion |
| `rating_engagement` | sociability, responsibility | depression, anxiety |

- `volatility` multiplies weight magnitudes everywhere (§6.1). High-volatility profiles react
  strongly to context; stable profiles barely move.
- `rating_discrimination` is where `aesthetic_sensitivity` lives: raters sensitive to ride
  quality (cleanliness, driving smoothness) spread their stars; deferential, compassionate ones
  compress everything toward 5. It scales rating weight magnitudes for both roles (§7.3).
- `rating_engagement` decides whether an agent bothers to vote at all: it lowers `no_vote_base`
  and shrinks abstention magnitudes (§7.2, §7.3).

**Every one of the 15 facets must influence at least one trait in each role.**

### 3.3 Provider affinity

`provider_affinity` ∈ `[-1.0, 1.0]`: a deterministic per-profile lean toward `uber` (positive)
or `lyft` (negative). Its direction and depth come from a hash of `profile_id` — stable for a
personality and unrelated to any real-world claim about either brand — scaled by
`provider_fidelity`, so a loyal personality leans hard and a promiscuous one barely leans. Its
magnitude never reaches 0, so no provider weight can. It is part of each role's `TraitView`, and
it must be used consistently for every uber/lyft weight in every phase, so a given profile never
contradicts itself about which brand it prefers.

### 3.4 Neutral reference

`NEUTRAL_PROFILE` is the all-0.5 profile and `NEUTRAL_TRAITS[role]` its traits, derived once at
import. It serves three purposes: the fixed point of person-level widening (§6.3), the reference
against which ride pairings are centered (§7.1), and the stand-in counterpart when a ride phase is
inspected without a pairing (`tables("ride")`, `resolve_ride()`).

### 3.5 `notes`

Each view exposes a `notes` string: one line, ≤ 200 characters (`NOTES_MAX_CHARS`),
deterministic, generated from the profile's two most extreme domains and the operational
tendencies they produce in that role, phrased as a behavioral rationale grounded in Big Five
theory. Example shape: *"High N, low C: impatient, surge-averse, wary of drivers, schedule-bound,
volatile ratings; abstains from rating often."* Map trait values to words through fixed bands,
end with the agent's rating engagement, and truncate safely if a note would exceed the limit.
Also provide `profile_notes(profile)`, a role-neutral variant for logs. No randomness, no
placeholder text.

---

## 4. Runtime evaluation modes

The two modes are never mixed. Both validate the context first (`validate_context`):

- Unknown context keys are ignored (the simulator passes extra state).
- A factor missing from the context contributes `0` and must not raise.
- A factor whose value is **`None` is treated as absent**, not as a bad value. The simulator
  legitimately produces `None` for a factor it cannot resolve yet — a vehicle's zone while it is
  off the network, or an age/zone lookup before the agent is registered — and raising would abort
  a multi-hour run mid-simulation.
- A **known factor with an unrecognized value** (e.g. `traffic_level="extreme"`, or a passenger
  age bucket passed as `driver_age`) raises `ValueError` naming the factor and the offending
  value. Silent fallback would hide bugs in generated test cases.

### 4.1 Mode A — categorical action choice

```python
raw[action]  = base[action] + sum(weights[action][factor][value] for factor in present_factors)
raw[action]  = max(raw[action], 0.001)          # clip so nothing is exactly zero
prob[action] = raw[action] / sum(raw.values())  # normalize to 1.0
```

Requirements:

- `base` sums to `1.0` within `1e-9`.
- Weights are signed deltas.
- Returned probabilities sum to `1.0` within `1e-9`, every action of the phase is present in the
  returned dict, and each lies in `[0, 1]`.
- `context=None` and `{}` give identical results.
- No randomness is drawn.

Worked check the implementation must satisfy — two actions, base `{accept: 0.6, reject: 0.4}`,
one factor contributing `+0.10` to `accept` and `−0.10` to `reject`: `raw = {0.70, 0.30}`,
`prob = {0.70, 0.30}`.

### 4.2 Mode B — rating with abstention

```python
p_abstain = clip(no_vote_base + sum(no_vote_weights[factor][value]), 0.0, 1.0)
latent    = rate_base + sum(rate_weights[factor][value])
```

Draw **exactly one** uniform value from the rng. If it is below `p_abstain`, return
`(None, True)`. Otherwise return `(finalize(latent), False)`, where `finalize` clips into
`[1.0, 5.0]` and, while the module constant `INTEGER_STARS` is `True` (the default), rounds to the
nearest whole star (halves up), because platforms accept whole stars only. Setting
`INTEGER_STARS = False` must recover the continuous value.

- Exactly one rng draw per rate call, whatever the outcome — this keeps seeded runs aligned
  across scenario variants.
- **The rating is deterministic given rater, counterpart and context.** The single draw decides
  only whether the agent votes. Real rating variation comes from real variation between rides,
  which the context already carries, not from a coin flip at the keypad. Never add per-call
  rating noise.
- `rate_base` is a scalar in `[1.0, 5.0]`; `no_vote_base` is a scalar in `[0.0, 1.0]`. Neither is
  per-action.
- No normalization in Mode B; clipping and rounding alone handle the bounds.
- Weights depend **only** on the rating agent's own personality. The counterpart's personality
  shifts **only** the two scalar bases.
- Use a supplied `random.Random`; the module-level generator is a fallback, never a substitute.
- `resolve_mode_b(...)` returns `(p_abstain, rating)` with no draw.

---

## 5. Context parameters and phase coverage

Exactly these names and values:

```python
 1 traffic_level         ["low", "mid", "high"]
 2 surge_multiplier      ["low", "mid", "high"]
 3 route_length          ["low", "mid", "high"]
 4 time_waiting_pass     ["low", "mid", "high"]
 5 time_waiting_driv     ["low", "mid", "high"]
 6 time_waiting_pickup   ["low", "mid", "high"]
 7 time_estimated_pickup ["low", "mid", "high"]
 8 time_estimated_ride   ["low", "mid", "high"]
 9 provider              ["uber", "lyft"]
10 time_of_day           ["morning", "afternoon", "night"]
11 passenger_rating      ["1-2", "2-3", "3-4", "4-5"]
12 driver_rating         ["1-2", "2-3", "3-4", "4-5"]
13 shift_time            ["low", "mid", "high"]
14 passenger_age         ["16-24", "25-34", "35-44", "45-54", "55-64"]
15 driver_age            ["18-30", "31-40", "41-50", "51-60", "61-70", "71+"]
16 zone_requested        ["downtown", "midtown", "suburbs"]
```

Note the age buckets differ by role: passenger has 5 buckets, driver has 6 including `71+`.
Never reuse one bucket set for the other role.

Define these as module-level constants (`CONTEXT_VALUES`, `PHASE_MODE`, `PHASE_FACTORS`,
`PHASE_ACTIONS`, `MODE_A_PHASES`, `RIDE_TABLE_KEYS`, plus ordered numeric positions for the age
and rating buckets) and build every table by iterating those constants, so coverage is
structurally guaranteed rather than hand-written.

### 5.1 Coverage matrix

Weights must be non-zero for exactly the listed factors, and must include every value of each.

| Role | Phase | Mode | Factors | Actions |
|---|---|---|---|---|
| driver | `start` | A | 1, 2, 9, 10, 15, 16 | `change_zone`, `start_work` |
| driver | `wait_requests` | A | 1, 2, 5, 9, 10, 13, 15, 16 | `stay_and_wait`, `change_zone`, `change_provider`, `stop_work` |
| driver | `request` | A | 1, 2, 3, 7, 8, 10, 11, 15, 16 | `accept`, `reject` |
| driver | `pickup` | A | 1, 2, 3, 7, 8, 10, 11, 15, 16 | `accept`, `reject` |
| driver | `ride` | B | 1, 9, 10, 13, 15, 16 | driver rates passenger |
| passenger | `request` | A | 2, 3, 4, 8, 9, 10, 14, 16 | `accept`, `reject_and_change_provider`, `reject` |
| passenger | `pickup` | A | 2, 3, 6, 7, 8, 10, 12, 14, 16 | `accept`, `reject` |
| passenger | `ride` | B | 9, 10, 14, 16 | passenger rates driver |

### 5.2 Semantics of age and zone

`passenger_age`, `driver_age`, and `zone_requested` are not decoration — they interact with the
derived traits and must carry real signal in every phase that lists them:

- Younger passengers are more surge-sensitive; the effect shrinks with age. It *fades* toward a
  small residual for the oldest bucket rather than reversing into an equally large bonus.
- Downtown reduces surge sensitivity (fewer alternatives, higher expectation of surge) and raises
  stress in rating phases.
- Older drivers are more surge-sensitive — they screen jobs harder, so the penalty on accepting
  grows with age — and more conservative in ratings; the oldest buckets also abstain more.
- Downtown raises effective `money_budget_ambition` for drivers (higher demand), suburbs lowers it.
- In the driver's positioning phases (`start`, `wait_requests`), the age effect is carried by
  `zone_flexibility`: the less zone-flexible the driver, the stronger the age tilt. State the
  direction you choose in a comment, and make sure the comment matches the code.

These interactions must be modulated by the agent's traits, not applied as flat constants — a
low-`surge_sensitivity` passenger should show a smaller age effect than a high-sensitivity one.
Factors act additively: there are no explicit factor × factor interactions.

---

## 6. Base and weight construction

### 6.1 Mode A weights

- Build one weight dict per action by iterating `PHASE_FACTORS[phase]` and
  `CONTEXT_VALUES[factor]`, so no value can be missed. Never include an unlisted factor.
- **Source and complement.** Emit each factor as a push on one *source* action — the action the
  factor is about — plus the exact complement spread over the other actions:
  `weights[source][factor][value] = delta` and
  `weights[other][factor][value] = −delta × share[other]`, where the shares are proportional to
  the traits that motivate each other action, floored (`SHARE_FLOOR`) so every action keeps a
  non-zero entry, and normalized to 1. This makes **complementarity** structural: for every
  factor value, the signed weights across a phase's actions sum to zero (within `1e-6` after
  scaling), so context shifts preference between actions rather than inflating or deflating the
  whole distribution. For two-action phases this reduces to exactly opposite signs.
  - For `wait_requests`, a value that pushes down `stay_and_wait` distributes the complementary
    mass across the three exit actions in proportion to the traits that motivate each
    (`zone_flexibility` → `change_zone`, low `provider_fidelity` → `change_provider`, low
    `work_urgency`/high `time_budget_flexibility` → `stop_work`). `shift_time` is sourced on
    `stop_work` (fatigue) and `provider` on `change_provider` (the preferred brand pushes
    switching *down*).
  - For passenger `request`, the complement of a push on `accept` is split between the two
    rejections (low `provider_fidelity` → `reject_and_change_provider`; a tight time budget and
    high `surge_sensitivity` → `reject`), and `provider` is sourced on
    `reject_and_change_provider`.
- **Magnitude.** `delta = direction[value] × scale`, with
  `scale = STEP_<PHASE> × volatility_factor × max(strength, small_floor)`: a per-phase magnitude
  budget; a factor rising linearly with `volatility` (e.g. `0.45 + 1.10 × volatility`, so a
  stable profile moves under half as far as the budget and a labile one about one and a half
  times as far for the same context); and a trait-driven strength. Later-commitment phases
  (`pickup`) multiply every strength by a damp below 1, because the agent has already invested in
  the ride.
- **Fleet-shared factors.** `traffic_level`, `surge_multiplier` and `time_of_day` take the same
  value for many agents at the same instant: every driver in a zone reads the same traffic and the
  same surge, and every agent reads the same time of day. A reaction to them is therefore a
  *correlated* shock, and correlated shocks are what push an aggregate count series past Poisson
  variance (the motivating measurement: an index of dispersion of 3.14 for the drivers'
  change-zone count, against 0.92–1.25 for every other decision count). Declare them as
  `FLEET_SHARED_FACTORS` and multiply their scale by `MACRO_FACTOR_DAMP` ∈ `(0, 1)`. This shifts
  expressed variation from "the whole fleet moves together" to "these agents differ", which is
  both less noisy in aggregate and the more defensible claim about human behavior.
- **Never emit `0.0`.** A near-neutral source delta is pushed out to a signed floor
  (`WEIGHT_FLOOR`) with the sign of the trait that motivates it; each emission carries a fallback
  sign for an exact zero. Complementary slices stay strictly proportional to the source delta, so
  a small slice of a floored delta may sit below the floor — never at zero, and never at the cost
  of complementarity, which is the stronger requirement.
- **A "mid" level is never neutral-zero:** it leans weakly (`NEUTRAL_MID`) the same way as the
  extreme it is closer to in the motivating trait (`LEVEL_PREFER_LOW`, `LEVEL_PREFER_HIGH`).
- Verify coverage, non-zeroness and complementarity when a phase is finished, raising a clear
  error naming phase, action, factor and value on any violation.

### 6.2 Mode A bases

- Build a base dict containing exactly the phase's actions, as functions of traits (e.g. a
  high-`work_urgency` driver starts with a higher `start_work` base; a `pickup` base accepts more
  readily than the matching `request` base).
- Widen it around the neutral agent (§6.3), floor every share away from zero, normalize, absorb
  floating-point drift deterministically, then assert the sum.

### 6.3 Person-level dispersion

Real people differ a lot from each other and are fairly consistent with themselves:
between-person variance is large, within-person variance small. A decision model gets that
backwards if every agent's action probability sits near the population mean, because then almost
all realized variation is the per-decision coin flip. (Motivating measurement on an earlier
revision: the between-agent SD of `P(accept)` at a fixed context was 0.05–0.10, the swing across
*every* context combination was 0.23, and the aggregate count series had an index of dispersion
of 1.19 — agents were nearly interchangeable and the aggregate was pure counting noise.)

So every Mode A base is widened by a per-role gain, `PERSON_GAIN[role]`:

- Compute the phase's raw base shares with one function of traits and evaluate it twice — for
  this agent and for `NEUTRAL_TRAITS[role]` — so the reference stays exact for non-linear share
  expressions instead of being a hand-computed constant that silently drifts.
- Move each share away from its neutral share **multiplicatively**:
  `widened = neutral × (agent / neutral) ** PERSON_GAIN[role]`, flooring both shares at
  `BASE_SHARE_FLOOR` before taking the ratio and clipping the result into
  `[BASE_SHARE_FLOOR, BASE_SHARE_CEIL]`. The neutral agent is an exact fixed point and the spread
  around it widens. Because `Σ p(1 − p)` falls as probabilities move away from 0.5, the aggregate
  series gets *less* noisy, not more.
- Widen the **ratio, not the level**. Additive widening (`neutral + gain × (agent − neutral)`)
  drives minority actions onto the floor: in the earlier measurement it left 62% of riders with
  `P(reject) < 0.01`, against real cancellation rates of 5–10%. A ratio can shrink a minority
  action without ever reaching zero, so a rider who rarely rejects still sometimes does.
- Tune the gain separately per role, because driver-side and passenger-side personality changes
  do not have symmetric leverage on aggregate output (in recorded runs a driver-side change moved
  48–50 of 56 output metrics, a passenger-side change 4–18). Every gain must be ≥ 1.0, and the
  gains must satisfy both dispersion targets of §12.9 at once: the between-agent SD of each
  phase's leading action must clear 0.075, and no action's *median* agent may be driven toward
  zero.

### 6.4 Mode B tables

- One `rate_weights` table and one `no_vote_weights` table per agent, from the rating agent's
  personality only, each value pushed off zero by its own signed floor.
- `rate_base` and `no_vote_base` computed from the pairing (§7).
- The counterpart must never alter the rating agent's weight tables.

### 6.5 Calibrated reference values

These tuning constants were calibrated together on the reference design. Use them as starting
values. Retune only if your own trait and weight magnitudes require it; keep each as one named
module constant with its rationale in a comment; and make sure the targets in §12.9 and §12.10
still hold.

| Constant | Reference value | Controls |
|---|---|---|
| `PERSON_GAIN` | `{"driver": 2.0, "passenger": 1.6}` | person-level widening of Mode A bases (§6.3) |
| `BASE_SHARE_FLOOR`, `BASE_SHARE_CEIL` | `0.02`, `0.96` | bounds on a widened base share |
| `MACRO_FACTOR_DAMP` | `0.65` | damping of `FLEET_SHARED_FACTORS` (§6.1) |
| `WEIGHT_FLOOR`, `SHARE_FLOOR` | `0.008`, `0.10` | smallest Mode A source delta and complement slice |
| `NEUTRAL_MID` | `0.15` | weak lean of a "mid" level |
| `STEP_<PHASE>` | `0.055`–`0.060` | per-phase Mode A magnitude budget |
| `RATE_BASE_SKEW`, `RATING_PAIRING_GAIN` | `4.0`, `2.5` | rating-curve shape and pairing spread (§7.1) |
| `RATING_WEIGHT_FLOOR`, `RATING_MAGNITUDE_CEIL` | `0.02`, `0.32` | per-factor range of rating weights (§7.3) |
| `ABSTAIN_WEIGHT_FLOOR` | `0.004` | smallest signed no-vote weight |
| `NO_VOTE_BASE_BAND` | `(0.12, 0.55)` | range of `no_vote_base` (§7.2) |

---

## 7. Ride-phase semantics and realism

Real ride-hailing ratings are heavily right-skewed: roughly 80–90% five-star, a mean of 4.8–4.9,
a thin 4-star shoulder and a small tail. Encode the skew in the bases and weights, not by
post-processing. Dissatisfaction often surfaces as silence rather than a low vote, so abstention
is a meaningful alternative to a low rating — but a low vote must stay possible. (Motivating
measurement: an earlier revision mapped pairings linearly onto a 4.0–5.0 band, and *no* rating in
4,000 draws landed below 4, so a dissatisfied rider's only outlet was abstaining.)

### 7.1 Rate base: a centered pairing on a skewed curve

- Each role defines `blend(own_traits, counterpart_traits)`: the rater's generosity mixed with
  the counterpart's quality (§7.4, §7.5).
- Center the blend on the neutral pairing and widen it:
  `pairing = clip(0.5 + RATING_PAIRING_GAIN × (blend(own, counterpart) − blend(own, neutral_counterpart)))`,
  where `neutral_counterpart` is `NEUTRAL_TRAITS` of the counterpart's role. A neutral pairing
  scores exactly 0.5. Centering matters more than the gain: an uncentered blend averages well
  above 0.5 and parks every pairing on the curve's flat top, so every ride rounds to 5.
- Map the pairing to a star base through a right-skewed curve with a flat top near 5 that falls
  away only at the genuinely bad end, e.g.
  `RATE_BASE_CEIL − (RATE_BASE_CEIL − RATE_BASE_FLOOR) × (1 − pairing) ** RATE_BASE_SKEW` with
  ceiling `5.0` and floor `1.0`. A neutral pairing then lands around 4.7–4.8, and only a jointly
  poor pairing approaches the floor. A linear map onto a narrow band cannot produce the shape and
  is not acceptable.
- The reference calibration was a grid search over floor, exponent and pairing gain, scoring each
  combination on total absolute deviation from the published shape over a realistic mix of ride
  contexts; its fitted point reproduced a mean of 4.85 with 88.0% / 10.2% / 0.9% / 0.8% / 0.1%
  across five to one star. Record the calibration rationale next to the constants.

### 7.2 Abstention base

`no_vote_base` reflects the rater's engagement — low `rating_engagement` abstains more, high
abstains less — nudged up slightly by a difficult counterpart, and is clipped into
`NO_VOTE_BASE_BAND`.

### 7.3 Magnitudes and bounds

- Rating-weight magnitude grows with `rating_discrimination` (hence `aesthetic_sensitivity`) and
  `volatility`, between `RATING_WEIGHT_FLOOR` and `RATING_MAGNITUDE_CEIL` per factor:
  discriminating, volatile raters spread their votes; deferential, steady ones sit on 5.
- No-vote magnitude grows as `rating_engagement` falls and as `volatility` rises. It stays small
  (a few hundredths), with its own signed floor `ABSTAIN_WEIGHT_FLOOR`.
- `no_vote_base` plus the extreme sums of `no_vote_weights` must stay inside `[0.0, 1.0]` **by
  construction**. Because the table is precomputed once per agent while the base depends on the
  pairing, fit the table against the whole `NO_VOTE_BASE_BAND`: choose directions so the worst
  case already fits, and scale the table down at construction time (with a small margin) if it
  would not. Clipping is a safety net, not the mechanism; the self-test checks the worst-case sums
  (§12.8).

### 7.4 Driver rates passenger

`blend` mixes driver generosity (`communication_skills`, `patience`, `trust_in_passenger`) with
passenger ease (passenger `communication_skills` and `patience`): a warm, expressive driver with
an easy passenger sits at the top; a reserved, impatient driver with a difficult passenger sits
lower.

`no_vote_base` reflects driver engagement; a difficult passenger raises it slightly.

`rate_weights`: high `traffic_level` negative (more so for impatient drivers), low positive, mid
small non-zero; `provider` a small loyalty bias via `provider_affinity` scaled by
`provider_fidelity`; `time_of_day` — a night ride is harder work and costs stars, the penalty
shrinking with `time_budget_flexibility`; `shift_time` increasingly negative as it grows
(fatigue), low slightly positive; `driver_age` more conservative with age; `zone_requested`
downtown most stressful, suburbs mildly positive.

`no_vote_weights`: the same factors acting on the likelihood of voting at all — high traffic,
long shifts, nights and downtown raise abstention, the preferred provider lowers it slightly, and
the oldest driver buckets raise it.

### 7.5 Passenger rates driver

`blend` mixes passenger generosity (`communication_skills`, `patience`, `trust_in_driver`) with
driver quality (driver `communication_skills`, `patience`, and smooth — low `driving_style` —
driving). Smooth, patient, communicative driving lands near 5.0; aggressive driving with poor
communication lands lower.

`no_vote_base` reflects passenger engagement, nudged by how hard the driver was to read.

`rate_weights`: `provider` loyalty bias; `time_of_day` as for drivers; `passenger_age` more
conservative with age; `zone_requested` stress, downtown strongest.

`no_vote_weights`: the same factors acting on abstention.

---

## 8. Retired: JSON compatibility layer and MBTI bridge

Earlier revisions carried a JSON-shaped compatibility layer (`DotDict`, `PersonalityView`,
`build_personality_config`) and a 16-type MBTI bridge (`MBTI_TO_OCEAN_PRESET`,
`ocean_profile_for_mbti`, `DriverView.from_mbti`, `PassengerView.from_mbti`). Their only purpose
was field-by-field diffing against the legacy MBTI JSON configuration. That comparison is no
longer needed, and nothing in the simulator imports them.

**Do not generate any of them**, and do not add MBTI presets, MBTI-keyed ride bases, or MBTI
parameters anywhere.

What replaces them:

- **Inspection:** `view.tables(phase)`, `view.traits`, `trait_rationale(role, trait)` and
  `resolve_ride` expose every number a decision rests on as plain, JSON-serializable data
  (§12.13).
- **Discrimination probe:** `ARCHETYPE_PROFILES`, the 32 corners of the five-domain cube (each
  domain `0.0` or `1.0`) plus the all-0.5 midpoint — 33 profiles spanning the whole input space,
  a strictly stronger probe than the MBTI presets, which all sat well inside the cube.
- The module docstring records the retirement in one short paragraph, so a reader who remembers
  the old layer knows where it went.

---

## 9. CSV and sampling

Expected CSV columns (the jobs file and the optional students file share the format):

`Code`, `Job`, `Openness (M)`, `Openness (SD)`, `Conscientiousness (M)`, `Conscientiousness (SD)`,
`Extraversion (M)`, `Extraversion (SD)`, `Agreeableness (M)`, `Agreeableness (SD)`,
`Neuroticism (M)`, `Neuroticism (SD)`.

### 9.1 Three distribution sources

These stay distinct:

- **`"0"` — the general population.** A built-in table of domain means and SDs on the 1–5 Likert
  scale, not read from any CSV, so agents are drawn from one global distribution instead of being
  assigned a job:

  | Domain | Mean | SD |
  |---|---|---|
  | openness | 3.6491 | 0.68558 |
  | conscientiousness | 3.8857 | 0.72289 |
  | extraversion | 3.2165 | 0.72822 |
  | agreeableness | 3.8175 | 0.63391 |
  | neuroticism | 2.7155 | 0.89516 |

  Sampling is **actor-aware** for drivers: a share `PROFESSIONAL_DRIVERS_SHARE = 0.37` of
  population-coded drivers is drawn from the professional-driver job, code `"8322"` (car, taxi
  and van drivers), and the returned record then carries that job's code and name. The decision
  is one uniform draw **from the model's rng**, taken for every population-coded driver. If the
  jobs CSV has no `"8322"` row (a reduced CSV, a test fixture), fall back to the population
  distribution rather than raise. Passengers never take this branch.
- **`"students"` — the optional students distribution**, loaded from `students_csv_path` (its
  first valid row; 1–5 Likert). If the file cannot be loaded, warn and continue without it;
  requesting `"students"` then raises `ValueError` saying that a students CSV is required.
- **Any other code** — the job with that `Code` in the jobs CSV.

### 9.2 Source scales

- The jobs CSV holds **T-scores** (population mean 50, SD 10): read them on a 20–80 scale
  (±3 SD). The population and students distributions are on the **1–5 Likert** scale.
- Model a scale as a small frozen `SourceScale(name, low, high)` and expose `LIKERT_SCALE`,
  `TSCORE_SCALE`, `DEFAULT_SOURCE_SCALE` (population) and `JOBS_SOURCE_SCALE` (jobs), so
  re-scaling every sampled population is a one-line edit.
- Each domain is drawn from `Gauss(M, SD)` on its source scale, clipped to the scale, then mapped
  **linearly** to `[0.0, 1.0]`. Provide `TSCORE_PERCENTILE_MAPPING = False`; when it is set to
  `True`, T-scores are mapped through the standard normal CDF (population percentile) instead.

### 9.3 Facets

The CSVs carry no facet structure, so each facet is sampled around its parent domain as
`clip(Gauss(domain, s))`. **Derive** the jitter `s` from a target within-domain facet
intercorrelation instead of fixing it: for `facet = domain + N(0, s)`, the implied correlation
between two facets of one domain is `r = var(domain) / (var(domain) + s²)`, so
`s = sd_unit(domain) × sqrt((1 − r) / r)`, where `sd_unit` is the source SD divided by the
scale's span. Target `FACET_DOMAIN_CORRELATION = 0.55`, inside the 0.43–0.67 that the BFI-2
reports for real within-domain facet intercorrelations (Soto & John, 2017), and floor `s` at a
small `FACET_JITTER_FLOOR` (e.g. `0.02`) so a job whose SD is ~0 still yields distinct facets.
A fixed small spread is not acceptable: an earlier fixed SD of 0.08 produced facet correlations
of 0.80–0.87, and facets that redundant carry nothing the domain does not already carry.

### 9.4 Robustness and reproducibility

- Draw order is part of the determinism contract: the professional-driver draw (if any), then
  the five domains in a fixed order, then the fifteen facets in domain order.
- Malformed rows (missing columns, non-numeric values, an empty `Code`) are skipped with a
  warning that names the row, and never abort the load. A CSV with no usable rows raises
  `ValueError`.
- An invalid job code raises `ValueError` naming the code and listing the available codes,
  including how to request the population (`"0"`) and students (`"students"`) distributions.
- **All** sampling uses the model's rng (`self.rng`, which falls back to the module-level
  generator only when no rng was supplied), so a seeded `OceanDecisionModel` reproduces the same
  population exactly.

---

## 10. Determinism and reproducibility

This is the property the digital-twin test harness depends on. Make it explicit and testable:

- Given the same seed, the same sequence of calls produces identical results.
- Mode A phase methods draw no randomness at all and leave any rng's state untouched. Only
  `choose()`, `rate_*()`, and sampling draw.
- Exactly one rng draw per `choose()` call and per `rate_*` call (§4.2).
- No global mutable personality state, no caching of anything that depends on context or a draw,
  and no import-time seeding of the module-level generator.
- Trait derivation, weight tables, `notes`, `profile_id` and `provider_affinity` are pure
  functions of the profile.
- Job-code injection changes only which distribution *future* agents are drawn from; it never
  touches an existing view.

---

## 11. Documentation

The module docstring must explain: the three layers of personality (domains, facets, derived
operational traits, including the shared traits); Mode A vs Mode B; constructing
`OceanDecisionModel`; sampling driver and passenger views; the job-code baseline and runtime
injection (`set_job_codes` / `reset_job_codes`); the context dict format, the full value
vocabulary, and the `None`-as-absent rule; the class-based API; inspecting tables (`tables`,
`traits`, `trait_rationale`, `resolve_ride`); the retirement of the JSON/MBTI layer and the
archetype probe that replaced it (§8); seeding for repeatable experiments; running the self-test;
and the remaining assumptions and limitations:

- the source scales (T-score jobs, Likert population and students, the percentile switch);
- facet sampling and its correlation target;
- trait coefficient magnitudes, weight magnitudes and rating bands are design choices informed by
  the Big Five literature and by ride-hailing rating distributions, not fitted values, and only
  their signs are part of the specification — whereas `PERSON_GAIN`, `MACRO_FACTOR_DAMP` and the
  rating-curve constants were calibrated against measured targets;
- `provider_affinity` encodes "this personality consistently prefers one brand", not any
  real-world claim about Uber or Lyft;
- factors act additively, and age/zone signals are modulated by traits, not by other factors;
- where the variance lives: person-level widening and fleet-shared damping, with the
  measurements that motivated them;
- ratings are whole stars, deterministic given rater, counterpart and context, with the single
  draw deciding only abstention.

Include a concise runnable example:

```python
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

probabilities = driver.request(context)                          # Mode A
action = driver.choose("request", context, rng)                  # one sampled action
rating, abstained = passenger.rate_driver(driver, context, rng)  # Mode B
```

---

## 12. Invariants

Numbered so the self-test can reference them.

1. **Clipping.** All domains, facets, sampled values, and derived traits lie in `[0.0, 1.0]`;
   `provider_affinity` lies in `[-1.0, 1.0]`; out-of-range constructor inputs are clipped.
2. **Facet defaulting.** An omitted facet equals its parent domain; `from_json` tolerates
   unrelated keys, rejects non-numeric recognized keys, recovers an omitted domain from its
   facets, and falls back to `NEUTRAL_DOMAIN` with a warning naming the domain; `from_facets` sets
   each domain to the mean of its facets and rejects a domain with none.
3. **Coverage.** Every phase's weights contain exactly its listed factors and every value of each,
   for every action (Mode A) and for both `rate_weights` and `no_vote_weights` (Mode B).
4. **No zeros.** No generated weight equals `0.0`.
5. **Mode A sums.** Every base sums to `1.0` ±`1e-9`; every returned distribution sums to `1.0`
   ±`1e-9`, contains every action, and lies in `[0, 1]`; an unknown key, an absent factor, a
   `None` value, and `context=None` all behave as absent; raw scores are clipped away from zero,
   never to it.
6. **Complementarity.** For each phase and each factor value, signed weights across actions sum
   to ~`0` (±`1e-6`).
7. **Mode B range.** Every result is `(None, True)` or `(r, False)` with `1.0 ≤ r ≤ 5.0`, and `r`
   is a whole star while `INTEGER_STARS` is on; `resolve_ride` returns `p_abstain ∈ [0, 1]` and a
   rating in `[1, 5]`.
8. **Bounds by construction.** For every profile, each edge of `NO_VOTE_BASE_BAND` and the
   `no_vote_base` against extreme counterparts, plus the most negative and the most positive
   achievable sums of `no_vote_weights`, stay within `[0.0, 1.0]` before clipping.
9. **Monotonicity, discrimination and dispersion.**
   - For each trait in §3.2, raising a listed facet (e.g. from 0.2 to 0.8, all else neutral)
     moves the trait in the listed direction; every facet moves at least one trait in each role.
   - Across the 33 `ARCHETYPE_PROFILES` in a fixed context, every Mode A phase has an action
     whose probability spread across archetypes is ≥ `0.05`; no two archetypes produce identical
     tables; `profile_id` does not collide on the probe profiles.
   - On a seeded representative population (§13) in a fixed context, the between-agent SD of each
     Mode A phase's leading action (first in `PHASE_ACTIONS`) is ≥ `0.075`, and every action's
     median probability is ≥ `0.02`. A long right tail in a minority action is realistic
     heterogeneity; a median agent that never takes the action is a tuning failure.
10. **Right skew and reachability.**
    - Across all 33 × 33 archetype pairings, every `rate_base` lies in `[1.0, 5.0]`.
    - Over representative-population pairings, the median `rate_base` is ≥ `4.5`.
    - Under a harsh negative context, some archetype pairing resolves to a rating below 4, for
      both roles.
    - Over a few thousand sampled rate calls per role (representative population, realistic ride
      contexts), the five-star share is in `[0.70, 0.95]`, the mean rating is in `[4.6, 4.95]`, and
      the share below 4 stars is greater than 0.
11. **Determinism.** Two runs with equally seeded `random.Random` produce identical outputs
    (profiles, traits, notes, Mode A results, choices, ratings, tables, sampled records), and
    different seeds produce different populations; Mode A leaves an rng's state unchanged; every
    rate call draws exactly one value; tables, notes and `provider_affinity` are pure functions of
    the profile.
12. **Purity.** Caller-supplied context and profile dicts are unchanged after every call;
    `traits` is read-only; `OceanProfile` is frozen; `tables()` returns deep copies.
13. **Serializability.** For every view,
    `json.dumps({phase: view.tables(phase) for phase in view.phases()})`,
    `json.dumps(view.traits.as_dict())` and `json.dumps(view.profile.as_dict())` succeed without a
    `default=` hook, and so does `json.dumps(model.sample_profile(...))`.
14. **Error clarity.** Each of these raises (or warns) with a message naming the offending input:
    an unrecognized factor value (in Mode A and in Mode B), a passenger age bucket passed as
    `driver_age`, a non-numeric recognized profile key, a domain with no facets in `from_facets`,
    an unknown phase in `tables()`, a Mode B phase in `choose()`, a counterpart of the wrong role
    (`TypeError` naming the expected role), an unknown job code (listing the available codes),
    `"students"` without a students CSV, a malformed CSV row (a warning naming the row), and a CSV
    with no usable rows.

## 13. Self-test

`_self_test()` asserts invariants 1–14, each with an assertion message citing its number (e.g.
`"[INV 6] ..."`), plus the calibration guards below. Split it into small `_test_*` functions and
keep the whole run to a few seconds.

Probe sets:

- **Archetypes** (`ARCHETYPE_PROFILES`) for claims about *reachable range* and discrimination.
- **Probe profiles:** the archetypes plus a few seeded random full-facet profiles built through
  `from_json` with an unrelated key.
- **A representative population** — domains drawn from the population distribution, facets
  jittered as in §9.3, seeded — for claims about the *shape* of a distribution (medians, SDs,
  rating shares). Half the archetypes are worst-case by construction, so their median says
  nothing about what a real population does; never test shape on the archetypes.
- **Contexts:** empty; partial with an unknown key; a full context; all-first values; all-last
  values; every single factor value on its own; a fixed realistic context for discrimination and
  dispersion; and a ride-context mix in which level factors are mostly `"mid"`.

It also:

- Writes temporary CSVs — a valid jobs file that includes a `"8322"` row, a students file, a file
  mixing good and malformed rows, and a file with no usable row — and confirms that bad rows are
  skipped with warnings naming the row and that the all-bad file raises.
- Checks the three distribution sources, `available_job_codes()`, `get_job_name()`, the
  source-scale mappings (including the percentile switch), and runtime injection:
  `set_job_codes`, `reset_job_codes`, `is_injected`, and that a rejected switch leaves the active
  code unchanged.
- Runs the determinism check with **population-coded drivers on a jobs CSV containing `"8322"`**,
  perturbing the module-level generator between the two runs, so that any draw bypassing the
  model's rng is caught.
- Exercises every public method once, including counterparts given as profiles and plain dicts,
  and the module-level generator used as a fallback.
- Asserts the calibration guards: every `PERSON_GAIN` is ≥ 1.0; widening leaves the neutral agent
  exactly unchanged, raises the share ratio to exactly the gain, and never zeroes a share;
  `0 < MACRO_FACTOR_DAMP < 1` and `FLEET_SHARED_FACTORS ⊆ CONTEXT_VALUES`; and the mean
  within-domain facet correlation of sampled profiles lies in `[0.35, 0.75]`.
- Prints one concise success line. Requires no network and no third-party packages.

---

## 14. Anti-patterns

Do not:

- Hand-write weight tables value by value instead of generating them from the constants in §5.
- Emit `0.0` and rationalize it as "no effect".
- Build weights directly from raw OCEAN domains, bypassing the operational traits.
- Check trait signs against the model's own coefficients instead of `SPEC_TRAIT_SIGNS`.
- Let the counterpart's personality touch the rating agent's weight tables.
- Produce a model so flat that different profiles behave alike, or so narrow that agents are
  interchangeable at population level (invariant 9 exists to catch both).
- Widen bases additively, so minority actions die on the floor.
- Map ratings linearly onto a narrow band, so no rating can fall below 4, or add per-call noise
  to ratings.
- Silently ignore an unrecognized value for a known factor, or reject `None` as if it were one.
- Draw randomness inside a Mode A phase method.
- Reuse passenger age buckets for the driver, or vice versa.
- Substitute the module-level `random` when an rng was supplied — anywhere, including the
  professional-driver draw.
- Seed the module-level generator at import time.
- Reintroduce the MBTI bridge or the JSON compatibility layer.
- Leave a comment or docstring whose claim the code contradicts.
- Drop or rename any public method or constructor parameter.

---

## 15. Scale note

This is a large module: 20-value profile handling with facet recovery; sixteen role-specific
traits, three shared traits, and the provider lean, each with a documented formula; six Mode A
phases and two Mode B phases with fully generated tables; person-level widening and fleet-shared
damping; a skewed whole-star rating curve; CSV sampling with three distribution sources, two
source scales, derived facet jitter, and runtime job-code injection; and a self-test covering 14
invariants plus calibration guards. Generate all of it, completely, in one file. No truncation,
no abbreviation, no placeholders, no "same as above".

Output only the contents of `ocean.py`.
