# Setup Guide

This guide provides detailed setup instructions for the Hasini AI Research Agent, including model downloads and MCP server configuration.

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Audio hardware (for voice interface)
- Telegram Bot API token (for Telegram interface)

## Basic Installation

1. **Clone and setup:**
   ```bash
   git clone https://github.com/n-r-kondapalli-21/Hasini-AI-Research-Agent.git
   cd Hasini_Ai_Research_Agent
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## Voice Interface Setup

The voice interface requires additional models and audio setup.

### Model Setup

Voice models are automatically downloaded on first run, but you can also manually place them:

**Required Models:**
- **Kokoro TTS**: Text-to-speech model (downloads automatically)
- **Silero VAD**: Voice activity detection (downloads automatically)
- **OpenWakeWord**: Wake word detection (downloads automatically)
- **Faster Whisper**: Speech-to-text (downloads automatically)

**Manual Model Placement:**
```
models/
├── kokoro/
│   ├── kokoro-v1_0.pth
│   └── config.json
├── vad/
│   └── silero_vad.onnx
├── wakeword/
│   ├── hey_jarvis_v0.1.onnx
│   └── embedding_model.onnx
└── whisper_models/
    └── whisper-small.en/
```

### Audio Configuration

The voice interface uses default system audio devices. To configure specific devices:

1. **List available devices:**
   ```python
   import sounddevice as sd
   print(sd.query_devices())
   ```

2. **Configure in code** (if needed):
   - Edit `voice/audio_capture.py` for input device
   - Edit `voice/audio_player.py` for output device

### Testing Voice Interface

```bash
python voice_main.py
```

Test by saying "Hey Jarvis" followed by a command.

## MCP Server Setup

### GitHub MCP Server

**Purpose:** Access GitHub repositories, issues, pull requests, etc.

**Setup:**
1. Get GitHub Personal Access Token:
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate new token with required permissions
2. Add to `.env`:
   ```
   GITHUB_TOKEN=your_github_token_here
   GITHUB_MCP_URL=https://api.githubcopilot.com/mcp/
   ENABLE_MCP_GITHUB=true
   ```

### OpenAlgo MCP Server

**Purpose:** Algorithmic trading operations (requires separate OpenAlgo installation)

**Setup:**
1. Install OpenAlgo separately (see [OpenAlgo documentation](https://github.com/your-repo/openalgo))
2. Configure paths in `.env`:
   ```
   OPENALGO_API_KEY=your_openalgo_api_key_here
   OPENALGO_URL=http://127.0.0.1:5000
   OPENALGO_PYTHON_PATH=path/to/openalgo/python
   OPENALGO_MCP_SERVER=path/to/openalgo/mcp/mcpserver.py
   ENABLE_MCP_OPENALGO=true
   ```

**Note:** OpenAlgo MCP is optional and only needed for trading functionality.

### Filesystem MCP Server

**Purpose:** File system operations for research and document management

**Setup:**
1. Create research directory:
   ```bash
   mkdir AI_Research
   ```
2. Configure in `.env`:
   ```
   AI_RESEARCH_DIR=./AI_Research
   ENABLE_MCP_FILESYSTEM=true
   ```

### MCP Server Dependencies

Install MCP server packages:
```bash
pip install mcp-server-filesystem
```

## Telegram Bot Setup

**Purpose:** Telegram bot interface for mobile/remote access

**Setup:**
1. **Create Telegram Bot:**
   - Message @BotFather on Telegram
   - Use `/newbot` command
   - Follow instructions to get bot token

2. **Configure in `.env`:**
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   ```

3. **Run Telegram bot:**
   ```bash
   python telegram_main.py
   ```

4. **Test:**
   - Find your bot on Telegram
   - Send `/start` command
   - Try `/help` for available commands

## LLM Provider Setup

The agent supports multiple LLM providers. Configure one in `.env`:

### OpenRouter (Recommended)
```
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### Google AI Studio
```
LLM_PROVIDER=google
GOOGLE_AI_STUDIO_API_KEY=your_google_api_key_here
```

### Zai
```
LLM_PROVIDER=zai
ZAI_API_KEY=your_zai_api_key_here
```

## Tool Configuration

### Web Search Tool
Requires Tavily API key:
```
TAVILY_API_KEY=your_tavily_api_key_here
ENABLE_WEB_TOOL=true
```

### Weather Tool
No additional setup required:
```
ENABLE_WEATHER_TOOL=true
```

### Calculator Tool
No additional setup required:
```
ENABLE_CALCULATOR_TOOL=true
```

## Permission System Setup

The permission system is configured in `permissions/permissions.json`. 

**Default Configuration:**
- **LOW Tier**: Read operations, information retrieval
- **MEDIUM Tier**: File modifications, repository operations
- **HIGH Tier**: Destructive operations, financial transactions

**Customization:**
Edit `permissions/permissions.json` to adjust permission tiers for specific tools.

## Testing Installation

### Test Text Interface
```bash
python main.py
```
Try commands like:
- `/tools` - List available tools
- `/tool web_search` - Inspect web search tool
- "What's the weather like?" - Test weather tool

### Test Voice Interface
```bash
python voice_main.py
```
Try:
- Say "Hey Jarvis"
- Ask "What time is it?"
- Say "exit" to quit

### Test Telegram Interface
```bash
python telegram_main.py
```
Try:
- `/start` - Start the bot
- `/help` - Get help
- Send a message to test response

## Troubleshooting

### Audio Issues (Voice Interface)
- Check microphone permissions
- Verify audio device selection
- Test with `python -c "import sounddevice; print(sounddevice.query_devices())"`

### Model Download Failures
- Check internet connection
- Verify sufficient disk space (~2GB for all models)
- Manually download models and place in `models/` directory

### MCP Server Connection Issues
- Verify API tokens are correct
- Check MCP server URLs are accessible
- Test MCP server connectivity independently

### Memory Issues
- Reduce `MEMORY_HISTORY_LIMIT` in `.env`
- Disable memory with `MEMORY_ENABLED=false`
- Monitor system resources during voice mode

## Security Considerations

1. **Never commit `.env` file** - Contains sensitive API keys
2. **Use read-only tokens** where possible for MCP servers
3. **Review permission tiers** - Ensure HIGH permissions are restricted
4. **Audit MCP server access** - Regularly review which MCP servers are enabled
5. **Secure Telegram bot** - Restrict bot access if needed

## Performance Optimization

### Voice Interface
- Use faster speech recognition: configure smaller Whisper model
- Reduce audio quality if needed: adjust audio capture settings
- Disable barge-in if not needed: set in voice configuration

### Text Interface
- Disable unused tools in `.env`
- Reduce conversation memory limit
- Use faster LLM provider if available

### Telegram Interface
- Implement rate limiting for high-traffic scenarios
- Consider using webhooks instead of polling for production

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [OpenRouter Documentation](https://openrouter.ai/docs)