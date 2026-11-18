from surgphase.data.phases import PhaseGroup, ordered_phases


class PhaseMapper:
    def __init__(self, phase_group: PhaseGroup):
        self.phase_group = phase_group
        self.phases = ordered_phases(phase_group)
        self.id_to_idx = {phase.id: index for index, phase in enumerate(self.phases)}

    def map_to_index(self, phase_id: str) -> int:
        if phase_id not in self.id_to_idx:
            raise ValueError(f"Unknown phase_id: {phase_id}. Available: {list(self.id_to_idx)}")
        return self.id_to_idx[phase_id]

    def map_to_id(self, idx: int) -> str:
        if not 0 <= idx < len(self.phases):
            raise ValueError(f"Index out of range: {idx}. Valid range: 0-{len(self.phases) - 1}")
        return self.phases[idx].id

    def get_phase_name(self, idx: int) -> str:
        self.map_to_id(idx)
        return self.phases[idx].name

    def get_phase_name_from_id(self, phase_id: str) -> str:
        return self.phases[self.map_to_index(phase_id)].name

    def num_phases(self) -> int:
        return len(self.phases)

    def get_all_phase_ids(self) -> list[str]:
        return [phase.id for phase in self.phases]

    def get_all_phase_names(self) -> list[str]:
        return [phase.name for phase in self.phases]
