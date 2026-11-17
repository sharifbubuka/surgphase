import json
from dataclasses import dataclass
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
    prompts: Prompts


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
            prompts=Prompts(
                descriptive=[
                    Prompt(id=p["id"], message=p["message"])
                    for p in phase_data["prompts"]["descriptive"]
                ],
                narrative=[
                    Prompt(id=p["id"], message=p["message"])
                    for p in phase_data["prompts"]["narrative"]
                ],
            ),
        )

    return PhaseGroup(
        id=phase_group_data["id"],
        name=phase_group_data["name"],
        description=phase_group_data["description"],
        surgery_name=phase_group_data["surgery_name"],
        phases=phases,
    )


def ordered_phases(phase_group: PhaseGroup) -> list[Phase]:
    return sorted(phase_group.phases.values(), key=lambda phase: phase.order)
