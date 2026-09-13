# Hasini AI Research Agent

An advanced AI-powered research agent with multiple interfaces (text, voice, and Telegram) featuring MCP server integration, permission-gated tools, and conversation memory.

## Features

- **Multiple Interfaces**: Text terminal, voice interaction, and Telegram bot
- **LangChain Integration**: Built on LangChain's agent framework
- **MCP Server Support**: Integration with Model Context Protocol servers (GitHub, OpenAlgo, Filesystem)
- **Permission System**: Tiered permission controls for tool access (LOW/MEDIUM/HIGH)
- **Voice Capabilities**: Wake word detection, speech-to-text, text-to-speech, and barge-in support
- **Conversation Memory**: Maintains context across interactions with configurable history limits
- **Extensible Tools**: Built-in tools for web search, weather, calculator, and custom MCP tools
- **Rich CLI**: Beautiful terminal interface with Rich library for text mode

## Project Structure

```
Hasini_Ai_Research_Agent/
├── main.py                    # Text/terminal interface entry point
├── voice_main.py              # Voice interface entry point
├── telegram_main.py           # Telegram bot interface entry point
├── agent_runtime.py           # Shared research agent runtime
├── config.py                  # Configuration settings
├── system_prompt.py           # System prompt configuration
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
├── Tools/                     # Built-in tools
│   ├── calculator.py
│   ├── weather.py
│   ├── web_search.py
│   └── tool_registry.py
├── services/                  # Core services
│   ├── llm.py                 # LLM provider management
│   └── conversation_memory.py # Conversation history management
├── permissions/               # Permission system
│   ├── permissions.json       # Permission tier configuration
│   ├── permission_registry.py
│   └── permission_gated_tool.py
├── mcps/                      # MCP server integration
│   ├── mcp_servers.json       # MCP server configuration
│   ├── mcp_clients.py
│   └── mcp_registry.py
├── voice/                     # Voice interface components
│   ├── audio_capture.py
│   ├── audio_player.py
│   ├── providers/             # STT, TTS, VAD, wake word providers
│   └── state_machine.py
├── telegram_interface/        # Telegram bot components
│   └── telegram_confirmation.py
└── models/                    # Model files (gitignored)
    ├── kokoro/                # TTS models
    ├── vad/                   # Voice activity detection models
    ├── wakeword/              # Wake word detection models
    └── whisper_models/        # Speech-to-text models
```

## Installation

For detailed setup instructions including model downloads and MCP server configuration, see [SETUP.md](SETUP.md).

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/n-r-kondapalli-21/Hasini-AI-Research-Agent.git
   cd Hasini_Ai_Research_Agent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   
   - **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**
   - Copy `.env.example` to `.env`
   - Add your API keys and configuration settings
   - Configure MCP server paths if needed

*For detailed setup instructions including voice interface, MCP servers, and Telegram bot setup, see [SETUP.md](SETUP.md)*

## Usage

### Text Interface
```bash
python main.py
```

### Voice Interface
```bash
python voice_main.py
```
*Note: Requires additional model downloads and audio setup*

### Telegram Bot
```bash
python telegram_main.py
```
*Note: Requires TELEGRAM_BOT_TOKEN in .env*

## Configuration

### Environment Variables (.env)

**LLM Configuration:**
- `LLM_PROVIDER`: openrouter, google, or zai
- `OPENROUTER_API_KEY`: OpenRouter API key
- `GOOGLE_AI_STUDIO_API_KEY`: Google AI Studio API key
- `ZAI_API_KEY`: Zai API key
- `MODEL_NAME`: Optional model name override

**Tool Configuration:**
- `TAVILY_API_KEY`: For web search tool
- `ENABLE_WEB_TOOL`: Enable/disable web search (true/false)
- `ENABLE_WEATHER_TOOL`: Enable/disable weather tool (true/false)
- `ENABLE_CALCULATOR_TOOL`: Enable/disable calculator (true/false)

**MCP Server Configuration:**
- `GITHUB_TOKEN`: GitHub API token for GitHub MCP
- `GITHUB_MCP_URL`: GitHub MCP server URL
- `OPENALGO_API_KEY`: OpenAlgo API key
- `OPENALGO_URL`: OpenAlgo server URL
- `AI_RESEARCH_DIR`: Path for filesystem MCP (default: ./AI_Research)
- `OPENALGO_PYTHON_PATH`: Python path for OpenAlgo MCP (default: python)
- `OPENALGO_MCP_SERVER`: Path to OpenAlgo MCP server (default: ./mcp/mcpserver.py)

**Telegram Configuration:**
- `TELEGRAM_BOT_TOKEN`: Telegram bot token

**Memory Configuration:**
- `MEMORY_ENABLED`: Enable conversation memory (true/false)
- `MEMORY_HISTORY_LIMIT`: Maximum messages to keep in memory (default: 10)

**MCP Server Flags:**
- `ENABLE_MCP_FILESYSTEM`: Enable filesystem MCP (true/false)
- `ENABLE_MCP_GITHUB`: Enable GitHub MCP (true/false)
- `ENABLE_MCP_OPENALGO`: Enable OpenAlgo MCP (true/false)

### Permission System

The agent uses a tiered permission system for tool access:
- **LOW**: Read-only operations, safe information retrieval
- **MEDIUM**: Operations that modify state but are reversible
- **HIGH**: Irreversible operations, financial transactions, destructive actions

Configure permission tiers in `permissions/permissions.json`.

## Development

### Adding New Tools
1. Create tool in `Tools/` directory
2. Register in `Tools/tool_registry.py`
3. Configure permission tier in `permissions/permissions.json`

### Adding MCP Servers
1. Add server configuration to `mcps/mcp_servers.json`
2. Configure environment variables in `.env`
3. Enable in `.env` with appropriate flag

### Voice Interface Setup
The voice interface requires additional setup:
1. Download required models (automatic on first run or manual placement in `models/`)
2. Configure audio input/output devices
3. Wake word: "Hey Jarvis" (configurable)

## Available Commands

### Text Interface
- `/tools` - List available tool categories
- `/tools <category>` - List tools in a category
- `/tool <tool_name>` - Inspect a specific tool
- `exit` / `quit` - Exit the session

### Telegram Interface
- `/start` - Start the bot
- `/help` - Show help message
- `/clear` - Clear conversation history
- `/tools` - List available tool categories
- `/tool <tool_name>` - Inspect a specific tool
- `/grant <tier>` - Pre-approve permissions (low/medium/high)
- `/revoke <tier>` - Revoke permission pre-approval
- `/permissions` - Show current permissions
- `/yes` - Approve a pending permission request
- `/no` - Deny a pending permission request

### Voice Interface
- Say "Hey Jarvis" to activate
- Say your command after wake word
- Say "exit", "quit", or "goodbye" to end session
- Barge-in support: interrupt agent responses

## License

MIT License - See LICENSE file for details
