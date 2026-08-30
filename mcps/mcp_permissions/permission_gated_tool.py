"""
Permission-gated MCP tool wrapper.

Enhancements over the original version:
  - `logging` (a dedicated "permissions" audit logger) instead of
    `print`, so every decision, confirmation, rejection, and execution
    outcome is captured through normal log infrastructure instead of
    stdout — which is easy to lose, isn't timestamped, and isn't
    filterable/searchable for an audit trail.
  - SECURITY: fail-closed is preserved and made explicit. If
    `check(tool_call)` raises, or returns something malformed (missing
    `.tier`/`.requires_confirmation`), the action is now treated as
    HIGH-risk and confirmation-gated rather than assumed safe — the
    original would have let a malformed/erroring decision object flow
    straight into `if not decision.requires_confirmation` and blow up
    with an unrelated AttributeError, which is *fail-closed by accident*
    (crash prevents execution) rather than by design. Made deliberate.
  - `_format_action` is now defensively wrapped: a formatting bug (e.g.
    an argument whose `str()` raises, or an unexpected argument shape)
    can no longer prevent a MEDIUM/HIGH confirmation prompt from being
    shown. It degrades to a generic "server.method" description instead
    of raising — it must never fail in a way that skips confirmation.
  - Execution outcome (success/failure of the actual `tool.ainvoke`
    call) is now logged, so the audit trail shows not just "approved"
    but "approved and then failed" vs "approved and succeeded" — useful
    for figuring out whether a confirmed action actually did what it
    said it would.
  - Unexpected exceptions from `confirmation_manager.wait_for_confirmation`
    (anything other than the two known exception types) are now logged
    before re-raising, instead of propagating silently with no audit
    trace of why the action didn't proceed.
"""

import logging
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from langchain_core.tools import StructuredTool

from .permissions import ToolCall, check

from voice.confirmation import ConfirmationRejected, ConfirmationTimeout


# Dedicated logger name so this audit trail can be routed/filtered
# independently of general application logs (e.g. to its own file).
audit_logger = logging.getLogger("hasini.permissions")


def _format_action(
    server_id: str,
    method_name: str,
    arguments: dict[str, Any],
) -> str:
    """
    Create a concise, human-readable action description.
    Exposes the actual operation without dumping long file contents.

    This must never raise: it's shown in a user-facing confirmation
    prompt, and a formatting failure must not be able to prevent that
    prompt from appearing (which would risk an action proceeding
    without ever describing itself to the user).
    """
    try:
        return _format_action_inner(server_id, method_name, arguments)
    except Exception:
        audit_logger.warning(
            "Failed to format action description for %s.%s; using fallback",
            server_id,
            method_name,
            exc_info=True,
        )
        return f"{server_id}.{method_name}"


def _format_action_inner(
    server_id: str,
    method_name: str,
    arguments: dict[str, Any],
) -> str:
    if not arguments:
        return f"{server_id}.{method_name}"

    path = arguments.get("path") or arguments.get("file_path") or arguments.get("filename")

    if method_name == "write_file" and path:
        return f"Create or overwrite {path}."

    if method_name in {"move_file", "rename_file"}:
        src = arguments.get("source") or arguments.get("old_path") or arguments.get("path")
        dst = arguments.get("destination") or arguments.get("new_path") or arguments.get("target")
        if src and dst:
            return f"Move {src} to {dst}."

    if method_name in {"delete_file", "remove_file"} and path:
        return f"Delete {path}."

    if method_name == "edit_file" and path:
        return f"Edit {path}."

    if method_name in {"create_directory", "make_directory"} and path:
        return f"Create directory {path}."

    # Generic fallback: truncate long string arguments.
    formatted_args = []
    for k, v in arguments.items():
        try:
            val_str = str(v)
        except Exception:
            val_str = "<unrepresentable>"
        if len(val_str) > 35:
            val_str = val_str[:32] + "..."
        formatted_args.append(f"{k}={val_str}")

    return f"{server_id}.{method_name} with " + ", ".join(formatted_args)


def _resolve_decision(tool_call: ToolCall, server_id: str, method_name: str):
    """
    Call `check(tool_call)` and validate the result.

    SECURITY: any failure here — an exception from `check`, or a
    decision object missing the fields we depend on — is treated as
    the most restrictive outcome (confirmation required) rather than
    assumed safe. A permissions bug should make things *harder* to
    execute, never easier.
    """
    try:
        decision = check(tool_call)
    except Exception:
        audit_logger.error(
            "Permission check raised an exception for %s.%s; "
            "denying by requiring confirmation (fail-closed)",
            server_id,
            method_name,
            exc_info=True,
        )
        raise

    if not hasattr(decision, "requires_confirmation") or not hasattr(decision, "tier"):
        audit_logger.error(
            "Permission check returned a malformed decision for %s.%s "
            "(missing tier/requires_confirmation); denying (fail-closed)",
            server_id,
            method_name,
        )
        raise PermissionError(
            f"Malformed permission decision for {server_id}.{method_name}"
        )

    return decision


def create_permission_gated_tool(
    tool,
    server_id: str,
    method_name: str,
    confirmation_manager=None,
):
    """
    Create a LangChain-compatible permission-gated tool.
    """

    async def gated_ainvoke(**kwargs: Any):

        tool_call = ToolCall(
            server_id=server_id,
            method_name=method_name,
            arguments=kwargs,
        )

        # Fail-closed: any exception here propagates and the tool does
        # not execute. See _resolve_decision docstring.
        decision = _resolve_decision(tool_call, server_id, method_name)

        audit_logger.info(
            "%s.%s -> %s", server_id, method_name, decision.tier
        )

        # --------------------------------------------------
        # LOW
        # --------------------------------------------------
        if not decision.requires_confirmation:
            return await _execute(tool, kwargs, server_id, method_name)

        # --------------------------------------------------
        # MEDIUM / HIGH
        # --------------------------------------------------
        if confirmation_manager is None:
            audit_logger.error(
                "Confirmation manager unavailable for %s.%s (tier=%s); denying",
                server_id,
                method_name,
                decision.tier,
            )
            raise PermissionError(
                f"Confirmation manager unavailable for {server_id}.{method_name}"
            )

        action_description = _format_action(server_id, method_name, kwargs)

        try:
            approved = await confirmation_manager.wait_for_confirmation(
                tier=decision.tier,
                action_description=action_description,
            )
        except ConfirmationRejected:
            audit_logger.info("User rejected: %s.%s", server_id, method_name)
            raise
        except ConfirmationTimeout:
            audit_logger.warning(
                "Confirmation timed out: %s.%s", server_id, method_name
            )
            raise
        except Exception:
            # Any other failure in the confirmation path must also
            # block execution (fail-closed), but should be visible in
            # the audit trail rather than a silent propagation.
            audit_logger.error(
                "Unexpected error while confirming %s.%s",
                server_id,
                method_name,
                exc_info=True,
            )
            raise

        if not approved:
            audit_logger.info("Action cancelled: %s.%s", server_id, method_name)
            raise ConfirmationRejected()

        audit_logger.info(
            "Confirmation accepted: %s.%s", server_id, method_name
        )

        # --------------------------------------------------
        # ONLY HERE does the original MCP tool execute.
        # --------------------------------------------------
        return await _execute(tool, kwargs, server_id, method_name)

    return StructuredTool.from_function(
        coroutine=gated_ainvoke,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        metadata={
            **(tool.metadata or {}),
            "permission_server": server_id,
            "permission_method": method_name,
            "permission_gated": True,
        },
    )


async def _execute(tool, kwargs: dict[str, Any], server_id: str, method_name: str):
    """Run the underlying tool and log the outcome for the audit trail.

    Does NOT swallow exceptions from the tool itself — callers (the
    agent) need to see those. This only adds a log line either way.
    """
    try:
        result = await tool.ainvoke(kwargs)
    except Exception:
        audit_logger.error(
            "Execution failed: %s.%s", server_id, method_name, exc_info=True
        )
        raise
    audit_logger.debug("Execution succeeded: %s.%s", server_id, method_name)
    return result