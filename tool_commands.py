def show_tools(tools, title="Available Tools"):

    print(f"\n{title}")
    print("-" * 50)

    if not tools:
        print("No tools found.")
    else:
        for tool in tools:
            print(f"• {tool.name}")

    print("-" * 50)


def handle_tool_command(command, all_tools, web_tools):

    command = command.lower().strip()

    # /tools
    if command == "/tools":
        show_tools(all_tools)
        return True

    # /tools web
    if command == "/tools web":

        show_tools(
            web_tools,
            "Web Tools"
        )

        return True

    # /tools filesystem
    if command == "/tools filesystem":

        filesystem_tools = [
            tool for tool in all_tools
            if any(
                keyword in tool.name.lower()
                for keyword in [
                    "file",
                    "directory",
                    "folder",
                    "filesystem"
                ]
            )
            and tool not in web_tools
        ]

        show_tools(
            filesystem_tools,
            "Filesystem MCP Tools"
        )

        return True

    # /tools github
    if command == "/tools github":

        github_tools = [
            tool for tool in all_tools
            if "github" in tool.name.lower()
            or any(
                keyword in tool.name.lower()
                for keyword in [
                    "repo",
                    "repository",
                    "issue",
                    "pull_request",
                    "pullrequest"
                ]
            )
        ]

        show_tools(
            github_tools,
            "GitHub MCP Tools"
        )

        return True

    # /tool <tool_name>
    if command.startswith("/tool "):

        tool_name = command[6:].strip()

        matching_tools = [
            tool for tool in all_tools
            if tool.name.lower() == tool_name
        ]

        if matching_tools:

            tool = matching_tools[0]

            print("\nTool")
            print("-" * 50)

            print(f"Name: {tool.name}")

            if hasattr(tool, "description"):
                print(f"\nDescription:\n{tool.description}")

            print("-" * 50)

        else:
            print(f"\nTool '{tool_name}' not found.")

        return True

    return False