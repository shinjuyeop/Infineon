"""Portable frozen Terrain/Reflex four-case decision state machine."""
from __future__ import annotations
from dataclasses import dataclass

HARD={"marble", "concrete"}
def is_hard_terrain(name:str)->bool:return name in HARD
def case_for(before:str,after:str)->str|None:
    if is_hard_terrain(before) and after=="ice":return "A"
    if is_hard_terrain(before) and after=="sand":return "B"
    if before=="ice" and is_hard_terrain(after):return "C"
    if before=="sand" and is_hard_terrain(after):return "D"
    return None
@dataclass
class Decision:
    terrain_state:str|None=None; transition_case:str|None=None; slip:bool=False; sink:bool=False
    def update(self,stable_terrain:str|None,slip:bool,sink:bool)->dict[str,object]:
        if stable_terrain is not None and stable_terrain!=self.terrain_state:
            prior=self.terrain_state; self.terrain_state=stable_terrain
            found=None if prior is None else case_for(prior,stable_terrain)
            if found and self.transition_case is None:self.transition_case=found
        self.slip=slip;self.sink=sink; c=self.transition_case
        hazard=slip or sink; matching=(c=="A" and slip) or (c=="B" and sink)
        return {"terrain_state":self.terrain_state,"transition_case":c,"slip":slip,"sink":sink,"hazard_reflex_required":hazard,"case_reflex_required":matching,"recovery_required":c in {"C","D"},"unmatched_hazard":hazard and not matching,"dual_hazard":slip and sink,"hazard_context_mismatch":(c=="A" and sink) or (c=="B" and slip)}
