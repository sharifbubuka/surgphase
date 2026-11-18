import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Prompt:
    id: str
    message: str


@dataclass(frozen=True)
class Prompts:
    descriptive: list[Prompt]
    narrative: list[Prompt]


@dataclass(frozen=True)
class Phase:
    id: str
    order: int
    name: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class PhaseGroup:
    id: str
    name: str
    description: str
    surgery_name: str
    phases: dict[str, Phase]


def load_phase_group(
    file_path: str | Path,
    exclude_phase_ids: str | list[str] | None = None,
) -> PhaseGroup:
    with Path(file_path).open("r", encoding="utf-8") as file:
        phase_group_data = json.load(file)

    if exclude_phase_ids is None:
        excluded_phase_ids = set()
    elif isinstance(exclude_phase_ids, str):
        excluded_phase_ids = {exclude_phase_ids}
    else:
        excluded_phase_ids = set(exclude_phase_ids)

    phases: dict[str, Phase] = {}
    for phase_data in phase_group_data["phases"].values():
        phase_id = phase_data["id"]
        if phase_id in excluded_phase_ids:
            continue
        if phase_id in phases:
            raise ValueError(f"Duplicate phase id: {phase_id}")

        phases[phase_id] = Phase(
            id=phase_id,
            order=phase_data["order"],
            name=phase_data["name"],
            members=(phase_id,),
        )

    return PhaseGroup(
        id=phase_group_data["id"],
        name=phase_group_data["name"],
        description=phase_group_data["description"],
        surgery_name=phase_group_data["surgery_name"],
        phases=phases,
    )


def load_prompts(file_path: str | Path) -> dict[str, Prompts]:
    with Path(file_path).open("r", encoding="utf-8") as file:
        prompts_data = json.load(file)

    return {
        phase_id: Prompts(
            descriptive=[Prompt(id=p["id"], message=p["message"]) for p in styles["descriptive"]],
            narrative=[Prompt(id=p["id"], message=p["message"]) for p in styles["narrative"]],
        )
        for phase_id, styles in prompts_data.items()
    }


def merge_phases(phase_group: PhaseGroup, merges: Iterable[Sequence[str]]) -> PhaseGroup:
    merges = [tuple(ids) for ids in merges]
    if not merges:
        return phase_group
    if any(len(ids) < 2 for ids in merges):
        raise ValueError(f"each merge needs at least two phase ids, got {merges}")
    merged_ids = [phase_id for ids in merges for phase_id in ids]
    unknown = set(merged_ids) - set(phase_group.phases)
    if unknown:
        raise ValueError(f"unknown phase ids {sorted(unknown)} in {phase_group.id!r}")
    if len(merged_ids) != len(set(merged_ids)):
        raise ValueError(f"a phase id appears in more than one merge: {merges}")

    absorbed = {phase_id for ids in merges for phase_id in ids[1:]}
    members = {ids[0]: ids for ids in merges}
    phases = {
        pid: replace(phase, members=members.get(pid, phase.members))
        for pid, phase in phase_group.phases.items()
        if pid not in absorbed
    }
    label = "_".join("+".join(ids) for ids in merges)
    return replace(phase_group, name=f"{phase_group.name}_{label}", phases=phases)


def parse_merge(text: str) -> tuple[str, ...]:
    return tuple(text.split("+"))


def ordered_phases(phase_group: PhaseGroup) -> list[Phase]:
    return sorted(phase_group.phases.values(), key=lambda phase: phase.order)
