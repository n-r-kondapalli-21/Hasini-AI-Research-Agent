# Tools (Built-in Application Tools)

This folder contains built-in application tools that provide core functionality to the agent. These are custom tools implemented directly in the codebase, as opposed to external MCP servers.

## 📁 Structure

```
Tools/
├── tool_registry.py          # Registry for built-in tools
├── calculator.py             # Calculator/math evaluation tool
├── weather.py                # Weather information tool
├── web_search.py             # Web search tool
└── __init__.py
```

## 🔧 Adding a New Built-in Tool

### Step 1: Create Your Tool File

Create a new Python file in the `Tools/` directory (e.g., `your_tool.py`):

```python
import logging
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Define input schema (optional but recommended)
class YourToolInput(BaseModel):
    param1: str = Field(description="Description of parameter 1")
    param2: int = Field(default=10, description="Description of parameter 2")

@tool
def your_tool_function(param1: str, param2: int = 10) -> str:
    """
    Brief description of what your tool does.
    
    More detailed description that helps the LLM understand
    when and how to use this tool effectively.
    """
    try:
        # Your tool logic here
        result = f"Processed {param1} with value {param2}"
        logger.info("Tool executed successfully: %s", result)
        return result
        
    except Exception as e:
        logger.warning("Tool failed: %s", e)
        return "Tool encountered an error. Please try again."

# Export the tool(s) - must be a list or single tool
your_tools = [your_tool_function]
```

### Step 2: Register Your Tool

Edit `tool_registry.py` and add your tool to the `initialize()` method:

```python
from Tools.your_tool import your_tools

class ToolRegistry:
    async def initialize(self):
        # ... existing registrations ...
        
        # --------------------------------------------------
        # Your tool category
        # --------------------------------------------------
        
        self.register(
            category="your_category",
            tools=your_tools,
        )
```

### Step 3: Restart the Application

The tool will be automatically discovered and available on startup.

## 📋 Currently Available Tools

| Tool | Category | Description |
|------|----------|-------------|
| `calculator` | calculator | Safe math expression evaluation |
| `get_weather` | weather | Current weather for any location |
| `web_search` | web | Web search via Tavily API |

## 🔍 How It Works

1. **Tool Definition**: Tools are defined using LangChain's `@tool` decorator
2. **Registration**: Tools are registered in `ToolRegistry` during initialization
3. **Categorization**: Tools are grouped by category for organized access
4. **Discovery**: All registered tools are available to the agent

## 🛠️ Key Components

### `tool_registry.py`
- Registers built-in application tools
- Groups tools by category
- Provides access to tools by name or category
- Manages tool lifecycle

### Tool Implementation Pattern

**Basic Tool:**
```python
@tool
def simple_tool(input: str) -> str:
    """Tool description."""
    return f"Processed: {input}"
```

**Tool with Schema:**
```python
class ToolInput(BaseModel):
    query: str = Field(description="Search query")

@tool
def advanced_tool(input: ToolInput) -> str:
    """Tool with structured input."""
    return f"Searching for: {input.query}"
```

**Multiple Tools in One File:**
```python
@tool
def tool_one(param: str) -> str:
    """First tool."""
    return f"Tool one: {param}"

@tool
def tool_two(param: int) -> str:
    """Second tool."""
    return f"Tool two: {param}"

your_tools = [tool_one, tool_two]
```

## 📝 Naming Conventions

- **File names**: Use lowercase with underscores (e.g., `web_search.py`)
- **Tool names**: Use lowercase with underscores (e.g., `web_search`)
- **Categories**: Use lowercase, simple names (e.g., `web`, `calculator`)
- **Tool exports**: Use `{tool_name}_tools` for lists, `{tool_name}_tool` for single tools

## 🎯 Best Practices

### Tool Descriptions
- Be specific about what the tool does
- Mention any limitations or constraints
- Include examples if helpful
- Describe when to use vs when not to use

### Error Handling
```python
try:
    # Tool logic
    result = perform_operation()
    return result
except SpecificException as e:
    logger.warning("Tool failed: %s", e)
    return "User-friendly error message"
except Exception as e:
    logger.error("Unexpected error: %s", e)
    return "Generic error message"
```

### Logging
```python
logger = logging.getLogger(__name__)

logger.debug("Detailed debugging info")
logger.info("Normal operation info")
logger.warning("Something unexpected but handled")
logger.error("Error that needs attention")
```

### Input Validation
```python
if not input or not input.strip():
    return "Invalid input: cannot be empty"

if len(input) > MAX_LENGTH:
    return f"Input too long (max {MAX_LENGTH} characters)"
```

## 🔐 Security Considerations

- **Never use `eval()` or `exec()`** on user input (see `calculator.py` for safe alternatives)
- Validate all inputs before processing
- Sanitize data before external API calls
- Use timeouts for network requests
- Handle sensitive data appropriately

## 📚 Dependencies

Common dependencies for tools:

```python
from langchain_core.tools import tool  # Required for all tools
from pydantic import BaseModel, Field  # For structured inputs
import logging  # For logging
import requests  # For HTTP requests
```

Add any additional dependencies to your project's `requirements.txt`.

## 🐛 Troubleshooting

**Tool not appearing:**
- Verify tool is imported in `tool_registry.py`
- Check tool is registered with `self.register()`
- Ensure tool has a `name` attribute (handled by `@tool` decorator)

**Tool not working:**
- Check logs for error messages
- Verify external dependencies (API keys, services)
- Test tool logic independently

**Import errors:**
- Ensure file is in `Tools/` directory
- Check `__init__.py` exists (can be empty)
- Verify Python path includes project root

## 📖 Examples

### Simple API Tool
```python
import requests
from langchain_core.tools import tool

@tool
def get_user_info(user_id: str) -> str:
    """Get user information by ID."""
    response = requests.get(f"https://api.example.com/users/{user_id}")
    return response.json()
```

### Tool with Configuration
```python
from config import API_KEY

@tool
def api_search(query: str) -> str:
    """Search using configured API."""
    client = APIClient(api_key=API_KEY)
    return client.search(query)
```

### Multi-output Tool
```python
@tool
def analyze_data(data: str) -> str:
    """Analyze data and return comprehensive results."""
    analysis = perform_analysis(data)
    summary = generate_summary(analysis)
    return f"Analysis: {analysis}\n\nSummary: {summary}"
```

## 🔄 MCP vs Built-in Tools

**Use Built-in Tools when:**
- Tool is simple and self-contained
- Tool doesn't require external server
- Tool needs tight integration with app logic
- Tool is project-specific

**Use MCP Servers when:**
- Tool requires external service
- Tool is complex or maintained separately
- Tool needs to be shared across projects
- Tool requires specialized infrastructure
