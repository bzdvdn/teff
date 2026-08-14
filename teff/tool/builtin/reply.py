"""ReplyToUser — write a plain-text reply and end the ReAct loop.

The coordinator sometimes needs to answer a direct message (a greeting, a
general question) without running any pipeline tool.  Instead of relying on
the model to emit bare text (which weak tool-calling models mishandle),
``reply_to_user(message=...)`` routes the answer through the tool path: the
``ToolExec`` node intercepts it, writes *message* to the agent's
``output_key`` and sets a done flag that short-circuits the ReAct loop, so
the message becomes the final reply with no extra LLM round.

Like :class:`AskHuman`, the tool's own ``arun`` is never invoked inside a
ReAct loop — the interception happens before ``execute_tool_calls``.
"""

from teff.tool.tool import Tool


class ReplyToUser(Tool):
    """Send a plain-text message to the user and end the agent loop.

    Args:
        message: The full reply text to send to the user.
    """

    name = "reply_to_user"
    description = (
        "Reply to the user directly with a plain text message.  Use this "
        "instead of producing bare text whenever the user's message is a "
        "greeting or a general question that needs no pipeline tools.  The "
        "message is delivered to the user as the final reply."
    )
    schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The full reply text to send to the user.",
            }
        },
        "required": ["message"],
    }

    async def arun(  # type: ignore[override]
        self, message: str = ""
    ) -> str:
        raise NotImplementedError(
            "reply_to_user is intercepted by the ToolExec node (it writes "
            "the message as the final reply and ends the ReAct loop); it "
            "cannot be invoked directly"
        )


__all__ = ["ReplyToUser"]
