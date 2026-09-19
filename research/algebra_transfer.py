"""Cross-algebra transfer of discovered Cortex equation forms.

The transfer layer deliberately reuses only canonical equation forms and
source-world evidence summaries. Source element labels, source table cells, and
source applicability scopes are never transferred.

Target worlds must re-ground every imported form from target evidence before it
can participate in closure.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence

from algebra_form_induction import (
    CandidateForm,
    CortexFuzzyLawInducer,
    Evidence,
    FormInference,
    LawScope,
    Table,
    _close_forms,
    _crossfit_lawset_evidence,
    _crossfit_predictive_evidence,
    _positive_scope,
    _structural_evidence,
)


@dataclass(frozen=True)
class TransferPrior:
    """Reusable equation form distilled from several source worlds."""

    form: CandidateForm
    source_worlds: int
    source_selected_worlds: int
    source_support_rate: float
    mean_structural_membership: float
    mean_predictive_membership: float

    @property
    def strength(self) -> float:
        return min(
            self.source_support_rate,
            self.mean_structural_membership,
            self.mean_predictive_membership,
        )


@dataclass(frozen=True)
class TransferLaw:
    prior: TransferPrior
    discovery: Evidence
    validation: Evidence
    scope: LawScope
    target_structural_membership: float
    target_predictive_membership: float
    structural_membership: float
    membership: float
    score: float
    active: bool
    predictive_active: bool


@dataclass(frozen=True)
class TransferInference:
    value: str | None
    resolved: bool
    provenance: tuple[str, ...]


def build_transfer_library(
    source_models: Sequence[CortexFuzzyLawInducer],
    *,
    minimum_source_support: float = 0.75,
    minimum_source_membership: float = 0.98,
    maximum_forms: int = 24,
) -> tuple[TransferPrior, ...]:
    """Distill reusable canonical forms from independently fitted source worlds."""
    if not source_models:
        raise ValueError("source_models must not be empty")
    if not 0.0 <= minimum_source_support <= 1.0:
        raise ValueError("minimum_source_support must lie in [0,1]")
    if not 0.0 <= minimum_source_membership <= 1.0:
        raise ValueError("minimum_source_membership must lie in [0,1]")
    if maximum_forms < 1:
        raise ValueError("maximum_forms must be positive")

    per_key: dict[str, list] = {}
    forms: dict[str, CandidateForm] = {}
    predictive_witnesses: dict[str, int] = {}

    for model in source_models:
        supported = {
            law.form.key: law
            for law in model.laws
            if law.active and law.predictive_active
        }
        for key, law in supported.items():
            forms[key] = law.form
            per_key.setdefault(key, []).append(law)
            predictive_witnesses[key] = (
                predictive_witnesses.get(key, 0)
                + law.validation.positive
            )

    priors: list[TransferPrior] = []
    total_worlds = len(source_models)
    for key, laws in per_key.items():
        selected_worlds = len(laws)
        support_rate = selected_worlds / total_worlds
        if support_rate < minimum_source_support:
            continue

        structural = sum(
            law.structural_membership
            for law in laws
        ) / selected_worlds
        predictive = sum(
            law.predictive_membership
            for law in laws
        ) / selected_worlds

        if structural < minimum_source_membership:
            continue
        if predictive < minimum_source_membership:
            continue

        priors.append(
            TransferPrior(
                form=forms[key],
                source_worlds=total_worlds,
                source_selected_worlds=selected_worlds,
                source_support_rate=support_rate,
                mean_structural_membership=structural,
                mean_predictive_membership=predictive,
            )
        )

    priors.sort(
        key=lambda prior: (
            -prior.strength,
            prior.form.complexity,
            -predictive_witnesses.get(prior.form.key, 0),
            prior.form.key,
        )
    )
    return tuple(priors[:maximum_forms])


def _target_minimum_witnesses(variable_count: int) -> int:
    if variable_count <= 1:
        return 2
    if variable_count == 2:
        return 2
    return 4


def _evidence_membership(
    evidence: Evidence,
    *,
    minimum_witnesses: int,
) -> float:
    if evidence.total == 0:
        return 0.0
    mass = min(1.0, evidence.total / max(1, minimum_witnesses))
    return evidence.consistency * mass


class CortexTransferredLawInducer:
    """Re-ground transferred forms in a new target algebra.

    This class never enumerates a target grammar. It scores only the forms in
    the supplied transfer library.
    """

    def __init__(
        self,
        elements: Iterable[str],
        observed: Table,
        library: Sequence[TransferPrior],
        *,
        membership_threshold: float = 0.98,
        complexity_penalty: float = 0.01,
        maximum_active_laws: int = 24,
    ) -> None:
        self.elements = tuple(sorted(elements))
        self.observed = dict(observed)
        self.library = tuple(library)

        if not self.elements:
            raise ValueError("elements must not be empty")
        if not 0.0 <= membership_threshold <= 1.0:
            raise ValueError("membership_threshold must lie in [0,1]")
        if complexity_penalty < 0.0 or not isfinite(complexity_penalty):
            raise ValueError("complexity_penalty must be finite and non-negative")
        if maximum_active_laws < 1:
            raise ValueError("maximum_active_laws must be positive")

        self.membership_threshold = membership_threshold
        self.complexity_penalty = complexity_penalty
        self.maximum_active_laws = maximum_active_laws

        self.laws = self._score_library()
        eligible = [
            law
            for law in self.laws
            if law.predictive_active
        ]
        eligible.sort(
            key=lambda law: (
                -law.score,
                law.prior.form.complexity,
                law.prior.form.key,
            )
        )

        accepted: list[TransferLaw] = []
        for law in eligible[: maximum_active_laws * 2]:
            if len(accepted) >= maximum_active_laws:
                break
            trial_forms = [
                item.prior.form
                for item in accepted
            ] + [law.prior.form]
            evidence, conflicts = _crossfit_lawset_evidence(
                trial_forms,
                self.elements,
                self.observed,
            )
            if (
                conflicts == 0
                and evidence.negative == 0
                and evidence.membership >= membership_threshold
            ):
                accepted.append(law)

        self.active_laws = tuple(accepted)
        self.completed = dict(self.observed)
        self.provenance: dict[tuple[str, str], tuple[str, ...]] = {}
        self.conflicts = 0
        self.rounds = 0
        self._closed = False

    def _score_library(self) -> tuple[TransferLaw, ...]:
        out: list[TransferLaw] = []
        for prior in self.library:
            form = prior.form
            discovery = _structural_evidence(
                form,
                self.elements,
                self.observed,
            )
            validation = _crossfit_predictive_evidence(
                form,
                self.elements,
                self.observed,
            )
            scope = _positive_scope(
                form,
                self.elements,
                self.observed,
            )

            target_structural = _evidence_membership(
                discovery,
                minimum_witnesses=_target_minimum_witnesses(
                    len(form.variables)
                ),
            )
            target_predictive = _evidence_membership(
                validation,
                minimum_witnesses=1,
            )

            structural = min(
                prior.strength,
                target_structural,
            )
            membership = min(
                structural,
                target_predictive,
            )
            score = (
                membership
                - self.complexity_penalty * form.complexity
            )

            out.append(
                TransferLaw(
                    prior=prior,
                    discovery=discovery,
                    validation=validation,
                    scope=scope,
                    target_structural_membership=target_structural,
                    target_predictive_membership=target_predictive,
                    structural_membership=structural,
                    membership=membership,
                    score=score,
                    active=structural >= self.membership_threshold,
                    predictive_active=membership >= self.membership_threshold,
                )
            )
        return tuple(out)

    def structurally_supported_keys(self) -> set[str]:
        return {
            law.prior.form.key
            for law in self.laws
            if law.active
        }

    def selected_keys(self) -> set[str]:
        return {
            law.prior.form.key
            for law in self.active_laws
        }

    def close(self) -> None:
        if self._closed:
            return

        (
            self.completed,
            self.provenance,
            self.conflicts,
            self.rounds,
        ) = _close_forms(
            self.elements,
            self.observed,
            [
                (law.prior.form, law.scope)
                for law in self.active_laws
            ],
        )
        self._closed = True

    def infer(self, left: str, right: str) -> TransferInference:
        self.close()
        pair = (left, right)
        value = self.completed.get(pair)
        if value is None:
            return TransferInference(None, False, tuple())
        if pair in self.observed:
            return TransferInference(value, True, ("observed",))
        return TransferInference(
            value,
            True,
            self.provenance.get(pair, ("derived",)),
        )


def score_transfer_heldout(
    model: object,
    table: Table,
    heldout: Iterable[tuple[str, str]],
) -> dict[str, float | int]:
    total = 0
    resolved = 0
    correct = 0

    infer = getattr(model, "infer")
    for pair in heldout:
        total += 1
        result = infer(*pair)
        if result.resolved:
            resolved += 1
            correct += int(result.value == table[pair])

    return {
        "heldout": total,
        "resolved": resolved,
        "correct": correct,
        "wrong_resolved": resolved - correct,
        "coverage": resolved / max(1, total),
        "accuracy": correct / max(1, total),
        "resolved_accuracy": correct / max(1, resolved),
    }
