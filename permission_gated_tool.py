import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from langchain_core.tools import StructuredTool

from permissions import ToolCall, check

from voice.confirmation import ConfirmationRejected, ConfirmationTimeout

def _format_action(
    server_id: str,
    method_name: str,
    arguments: dict[str, Any],
) -> str:
    """
    Create a concise, human-readable action description.
    Exposes the actual operation without dumping long file contents.
    """
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

    # Generic fallback: truncate long string arguments
    formatted_args = []
    for k, v in arguments.items():
        val_str = str(v)
        if len(val_str) > 35:
            val_str = val_str[:32] + "..."
        formatted_args.append(f"{k}={val_str}")

    return f"{server_id}.{method_name} with " + ", ".join(formatted_args)


def create_permission_gated_tool(
    tool,
    server_id: str,
    method_name: str,
    confirmation_manager=None,
):
    """
    Create a LangChain-compatible permission-gated tool.
    """

    async def gated_ainvoke(
        **kwargs: Any,
    ):

        tool_call = ToolCall(
            server_id=server_id,
            method_name=method_name,
            arguments=kwargs,
        )

        decision = check(tool_call)

        print(
            f"[permissions] "
            f"{server_id}.{method_name} "
            f"-> {decision.tier}"
        )

        # --------------------------------------------------
        # LOW
        # --------------------------------------------------

        if not decision.requires_confirmation:

            return await tool.ainvoke(kwargs)

        # --------------------------------------------------
        # MEDIUM / HIGH
        # --------------------------------------------------

        if confirmation_manager is None:

            raise PermissionError(
                f"Confirmation manager unavailable for "
                f"{server_id}.{method_name}"
            )

        action_description = _format_action(
            server_id,
            method_name,
            kwargs,
        )

        try:

            approved = await confirmation_manager.wait_for_confirmation(
                tier=decision.tier,
                action_description=action_description,
            )

        except ConfirmationRejected:

            print(
                f"[permissions] User rejected: "
                f"{server_id}.{method_name}"
            )
            raise

        except ConfirmationTimeout:

            print(
                f"[permissions] Confirmation timed out: "
                f"{server_id}.{method_name}"
            )
            raise

        if not approved:

            print(
                f"[permissions] Action cancelled: "
                f"{server_id}.{method_name}"
            )
            raise ConfirmationRejected()

        print(
            f"[permissions] Confirmation accepted: "
            f"{server_id}.{method_name}"
        )

        # --------------------------------------------------
        # ONLY HERE does the original MCP tool execute.
        # --------------------------------------------------

        return await tool.ainvoke(kwargs)

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