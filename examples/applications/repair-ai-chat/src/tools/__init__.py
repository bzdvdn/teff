"""Tool registry for the repair coordinator.

The coordinator sees only the *sub-agent* tools plus ``ask_human``; the
domain tools they orchestrate are kept off its tool list and scoped inside
each sub-agent's own ReAct loop.
"""

from src.tools.agents import (
    ExtractProjectInfo,
    PrepareEstimate,
    ProposePlan,
    RunQaCheck,
    SelectMaterials,
)
from src.tools.budget import EstimateMaterialCost, EstimateTotal
from src.tools.material import (
    CalculateLaminate,
    CalculatePaint,
    CalculatePlaster,
    CalculatePutty,
    CalculateTiles,
)
from src.tools.rag import FindSimilarMaterial, SearchMaterials
from src.tools.room import (
    CalculateCeilingArea,
    CalculateFloorArea,
    CalculatePerimeter,
    CalculateWallArea,
)
from teff.tool.builtin import AskHuman, ReplyToUser

#: Tools the coordinator may call directly.
COORDINATOR_TOOLS = [
    "extract_project_info",
    "propose_plan",
    "select_materials",
    "prepare_estimate",
    "run_qa_check",
    "ask_human",
    "reply_to_user",
]


def build_tools(
    services, catalog, *, model: str = "llama3.1:8b", provider: str = "ollama"
) -> list:
    """Instantiate the full tool set bound to *services* and *catalog*."""
    room_tools = [
        CalculateWallArea(services.room),
        CalculateFloorArea(services.room),
        CalculateCeilingArea(services.room),
        CalculatePerimeter(services.room),
    ]
    material_tools = [
        CalculateTiles(services.material),
        CalculatePaint(services.material),
        CalculateLaminate(services.material),
        CalculatePlaster(services.material),
        CalculatePutty(services.material),
    ]
    budget_tools = [
        EstimateMaterialCost(services.budget),
        EstimateTotal(services.budget),
    ]
    rag_tools = [
        SearchMaterials(catalog),
        FindSimilarMaterial(catalog),
    ]
    return [
        ExtractProjectInfo(model=model, provider=provider),
        ProposePlan(
            model=model,
            provider=provider,
            tools=room_tools,
        ),
        SelectMaterials(
            model=model,
            provider=provider,
            tools=[*rag_tools, *material_tools],
        ),
        PrepareEstimate(
            model=model,
            provider=provider,
            tools=[*room_tools, *material_tools, *budget_tools, *rag_tools],
        ),
        RunQaCheck(model=model, provider=provider),
        AskHuman(),
        ReplyToUser(),
        *room_tools,
        *material_tools,
        *budget_tools,
        *rag_tools,
    ]


__all__ = ["COORDINATOR_TOOLS", "build_tools"]
