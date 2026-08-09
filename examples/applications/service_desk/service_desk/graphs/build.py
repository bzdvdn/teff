"""Service-desk graph builder — the default supervisor chat router.

A single :class:`teff.node.Supervisor` (composed with
:meth:`teff.flow.Flow.team` — the one-call team recipe) dispatches every
request to one specialist::

    reset ─► supervisor ─ next_agent=billing ─► [ReAct billing] ─┐
       ▲         ▲                                             │
       └─────────┴────────────────────────────────────────────┘
        (next_agent=deploy → ReAct deploy → Interrupt(approval))
        (next_agent=fallback → ReAct fallback)
        (next_agent=finish → final LLM, ends the turn)

``team()`` wires the supervisor decider + one routed role per specialist in
a single step; it is the compact twin of an explicit
``supervisor()`` + ``route()`` pair.  Guards demonstrated here:
- ``done_keys`` — once *any* specialist answered, the turn finishes
  deterministically without another supervisor call;
- ``fallback_agent`` — a premature ``finish`` on an empty turn routes to
  ``fallback`` instead of ending silently;
- ``max_rounds`` — the round counter bounds even a model that never stops;
- ``Interrupt`` — the deploy gateway pauses inside the route chain; the
  operator's answer lands in ``deploy_approved`` and the final summary
  honours it (resume continues straight back at the supervisor).

Each specialist is a self-contained ReAct agent
(:class:`teff.flow.AgentRole`, built on :func:`teff.flow.agent_step`)
writing into its own state slot, scoped to a single knowledge-base tool
(``use_tools=<allowlist>``).  The function returns the assembled graph
*and* the tools pool so the caller can hand it to ``graph.run(tools=...)``
/ ``Assistant``.
"""

from __future__ import annotations

from service_desk.core.deps import build_deps
from service_desk.graphs.prompts import (
    BILLING_PROMPT,
    DEPLOY_PROMPT,
    FALLBACK_PROMPT,
    FINAL_SYSTEM,
    INCIDENT_PROMPT,
    SUPERVISOR_PROMPT,
)
from service_desk.tools import KNOWLEDGE_TOOLS, build_tools
from teff.flow import AgentRole, Flow
from teff.node import LLM, ContextBuilder, Interrupt
from teff.provider import ProviderRegistry

MODEL_DEFAULT = "llama3.1:8b"

#: Shared state keys rendered into the supervisor's context each turn.
AGENT_SECTIONS = {
    "billing": "Счета и платежи",
    "incident": "Инциденты",
    "deploy": "Деплой",
    "fallback": "Разговор",
}

#: State keys reset at the start of every fresh turn.
_RESET_KEYS = (
    "next_agent",
    "supervisor_rounds",
    "billing",
    "incident",
    "deploy",
    "fallback",
    "deploy_approved",
    "input",
    "final",
)


def build_flow(
    model: str = MODEL_DEFAULT,
    *,
    provider: str = "ollama",
    knowledge=None,
) -> tuple[Flow, list]:
    """Assemble the service-desk supervisor router.

    *provider* is threaded into every agent's harness config (per-node) so
    the graph never touches the framework's global defaults.  *knowledge* is
    the shared :class:`~service_desk.rag.knowledge.KnowledgeBase` (built via
    :func:`service_desk.core.deps.build_deps` when omitted); each specialist is scoped
    to a single knowledge-search tool from its pool.

    Returns ``(flow, tools)`` — the tools go to ``graph.run(tools=...)`` /
    :class:`~teff.Assistant`.
    """

    def role(system: str, slot: str, use_tools: str | None = None) -> AgentRole:
        return AgentRole(
            system,
            output_key=slot,
            sections=AGENT_SECTIONS,
            use_tools=use_tools if use_tools else None,
        )

    if knowledge is None:
        knowledge = build_deps(provider=provider)
    tools = build_tools(knowledge)

    flow = Flow("service_desk", providers=ProviderRegistry.from_presets(provider))

    # Entry: reset the per-turn scratch, so a new message routes afresh.
    flow.step(
        ContextBuilder(sections=AGENT_SECTIONS, reset_keys=_RESET_KEYS),
        id="reset",
    )

    # The default chat supervisor, composed in one call: the decider plus one
    # routed agent per role, the deploy gateway as a chain inside its route.
    flow.team(
        SUPERVISOR_PROMPT,
        roles={
            "billing": role(BILLING_PROMPT, "billing", KNOWLEDGE_TOOLS["billing"]),
            "incident": role(INCIDENT_PROMPT, "incident", KNOWLEDGE_TOOLS["incident"]),
            # Deploy carries a human gateway: the specialist plans the release,
            # then the run pauses and the operator's answer is read back by the
            # supervisor (which finishes) and by the final summary.
            "deploy": [
                role(DEPLOY_PROMPT, "deploy", KNOWLEDGE_TOOLS["deploy"]),
                Interrupt(
                    key="deploy_approved",
                    prompt=(
                        "План выкатки: {deploy}\n\n"
                        "Подтверждаешь выкатку в прод? Ответь: да или нет."
                    ),
                    id="approve",
                ),
            ],
            "fallback": role(FALLBACK_PROMPT, "fallback"),
        },
        model=model,
        provider=provider,
        sections=AGENT_SECTIONS,
        route_keys={
            "billing": "billing",
            "incident": "incident",
            "deploy": "deploy",
            "fallback": "fallback",
        },
        done_keys=["billing", "incident", "deploy", "fallback"],
        done_mode="any",
        fallback="fallback",
        max_rounds=8,
        finish=LLM(
            model=model,
            provider=provider,
            system=FINAL_SYSTEM,
            prompt=(
                "Счета и платежи:\n{billing}\n\n"
                "Инциденты:\n{incident}\n\n"
                "Деплой:\n{deploy}\n\n"
                "Подтверждение деплоя: {deploy_approved}\n\n"
                "Разговор:\n{fallback}"
            ),
            output_key="final",
            id="final",
        ),
        id="supervisor",
    )

    return flow, tools
