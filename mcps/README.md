# MCPs (Model Context Protocol Servers)

This folder manages external MCP servers that extend the agent's capabilities with specialized tools and data sources.

## 📁 Structure

```
mcps/
├── mcp_servers.json          # Configuration file for MCP servers
├── mcp_registry.py           # Registry for MCP servers and their tools
├── mcp_clients.py            # MCP server client utilities
└── __init__.py

permissions/                  # Centralized permission system (tool-agnostic)
├── __init__.py
├── permissions.json          # Permission configuration
├── permissions.py            # Permission checking logic
├── permission_registry.py   # Permission registry and discovery
└── permission_gated_tool.py  # Permission-gated tool wrapper
```

## 🔧 Adding a New MCP Server

### Step 1: Add Server Configuration

Edit `mcp_servers.json` and add your server configuration:

```json
{
    "your_server_name": {
        "transport": "stdio",
        "command": "path/to/server/executable",
        "args": ["arg1", "arg2"]
    }
}
```

**Transport Types:**
- `stdio` - Standard input/output communication
- `streamable_http` - HTTP-based communication

**Example Configurations:**

**STDIO Server:**
```json
{
    "filesystem": {
        "transport": "stdio",
        "command": "mcp-server-filesystem",
        "args": ["D:\\path\\to\\directory"]
    }
}
```

**HTTP Server:**
```json
{
    "github": {
        "transport": "streamable_http",
        "url": "https://api.example.com/mcp/",
        "headers": {
            "Authorization": "your_token_here"
        }
    }
}
```

### Step 2: Add Permissions (Optional)

If the MCP server requires permission gating, edit `permissions/permissions.json`:

```json
{
    "your_server_name": {
        "default": "MEDIUM",
        "methods": {
            "read_*": "LOW",
            "write_*": "MEDIUM",
            "delete_*": "HIGH"
        }
    }
}
```

Permission tiers:
- **LOW**: Execute immediately without confirmation
- **MEDIUM**: Require normal confirmation
- **HIGH**: Require strict confirmation

### Step 3: Restart the Application

The MCP servers are automatically discovered and initialized on startup. No code changes needed.

## 📋 Currently Configured MCP Servers

| Server | Transport | Purpose |
|--------|-----------|---------|
| `filesystem` | stdio | File system operations |
| `github` | streamable_http | GitHub API integration |
| `openalgo` | stdio | Trading/financial data |

## 🔍 How It Works

1. **Configuration Loading**: `mcp_clients.py` loads server configs from `mcp_servers.json`
2. **Server Connection**: Each server connects independently via `MultiServerMCPClient`
3. **Tool Discovery**: Tools are automatically discovered from each connected server
4. **Permission Gating**: Tools are wrapped with permission checks via the centralized `permissions/` package
5. **Registration**: Tools are registered in `MCPRegistry` under their server category

## 🛠️ Key Components

### `mcp_registry.py`
- Initializes MCP servers
- Discovers and registers MCP tools
- Applies permission gates via the centralized permission system
- Groups tools by server
- Provides access to tools and the combined MCP client

### `mcp_clients.py`
- Loads MCP server configuration
- Connects to configured servers
- Discovers available tools
- Registers MCP server discovery with the centralized permission system
- Returns combined MCP client

### `permissions/` (Centralized Permission System)
- Tool-agnostic permission registry system
- Permission-gated tool wrapper (usable by any tool system)
- User confirmation management
- Supports MCP, browser automation, email, and future tool systems

## 📝 Naming Conventions

- **Server IDs**: Use lowercase, underscore-separated names (e.g., `filesystem`, `github`)
- **Tool Names**: Automatically prefixed with server ID (e.g., `filesystem_read_file`)
- **Categories**: Server IDs serve as categories for tool grouping

## ⚠️ Security Notes

- API tokens in `mcp_servers.json` should be kept secure
- Consider using environment variables for sensitive data (not currently implemented)
- Permission gates can require user confirmation for sensitive operations
- Filesystem access is restricted to configured directories

## 🐛 Troubleshooting

**Server fails to connect:**
- Check server executable path and arguments
- Verify server is installed and accessible
- Check logs for detailed error messages

**Tools not appearing:**
- Ensure server is successfully connected
- Check `permissions/permissions.json` if permission gating is active
- Verify tool names match expected patterns

**Permission issues:**
- Review `permissions/permissions.json` configuration
- Check confirmation manager is properly initialized
- Verify tool-server mapping in permission registry

## 📚 Additional Resources

- [MCP Specification](https://modelcontextprotocol.io/)
- [LangChain MCP Adapters](https://python.langchain.com/docs/integrations/mcp/)
