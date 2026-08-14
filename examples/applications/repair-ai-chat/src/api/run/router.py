"""Session endpoints — inspect and delete durable runs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from src.api.auth.router import require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/{chat_id}")
async def get_run(
    chat_id: str,
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> dict:
    owner = x_user_id or chat_id
    saved = await request.app.state.assistant.checkpointer.load(chat_id, owner=owner)
    if saved is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        "chat_id": chat_id,
        "owner": owner,
        "next_node_id": saved.next_node_id,
        "iteration": saved.iteration,
        "state": saved.state,
    }


@router.delete("/{chat_id}")
async def delete_run(
    chat_id: str,
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> dict:
    owner = x_user_id or chat_id
    await request.app.state.assistant.checkpointer.delete(chat_id, owner=owner)
    return {"chat_id": chat_id, "status": "deleted"}
