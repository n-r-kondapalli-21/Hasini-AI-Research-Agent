# Telegram Bot Integration for Hasini AI Research Agent

This module provides a Telegram bot interface for the Hasini AI Research Agent, allowing users to interact with the agent through Telegram messages.

## Features

- **Text-based interaction**: Send messages to the agent via Telegram
- **Conversation memory**: Maintains per-user conversation history
- **Tool integration**: Full access to all existing tools (web search, weather, calculator, etc.)
- **Command support**: `/start`, `/help`, `/clear` commands
- **Error handling**: Robust error handling and logging
- **Modular design**: Easy to extend for voice messages or other features

## Architecture

The Telegram integration follows the same architecture as the existing text and voice interfaces:

- **Reuses existing agent runtime**: Uses `create_research_agent()`, `Agent_stream`, and `ConversationMemory`
- **Per-user memory**: Each Telegram user gets their own conversation memory
- **Confirmation manager**: Telegram-specific confirmation for permission-gated tools
- **Clean separation**: Telegram-specific code is isolated in the `telegram_interface/` directory

## Setup

### 1. Create a Telegram Bot

1. Open Telegram and search for [@BotFather](https://t.me/BotFather)
2. Send `/newbot` command
3. Follow the instructions to create your bot
4. Copy the bot token (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### 2. Configure Environment Variables

Add the following to your `.env` file:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

### 3. Install Dependencies

The required dependency is already included in `requirements.txt`:

```bash
pip install python-telegram-bot==22.8
```

## Usage

### Starting the Telegram Bot

Run the Telegram bot from the project root:

```bash
python telegram_main.py
```

The bot will start and you'll see logs indicating it's running:

```
INFO:hasini.telegram_main:Starting Hasini Telegram Bot...
INFO:hasini.telegram_main:Initializing Telegram bot...
INFO:hasini.telegram_main:Creating research agent for Telegram mode...
INFO:hasini.agent_runtime:Research agent ready (mode=text).
INFO:hasini.telegram_main:Telegram bot initialized successfully.
INFO:hasini.telegram_main:Starting Telegram bot polling...
INFO:hasini.telegram_main:Hasini Telegram Bot started successfully.
INFO:hasini.telegram_main:Press Ctrl+C to stop.
```

### Interacting with the Bot

1. Open Telegram and search for your bot by name
2. Click **Start** or send `/start` to begin
3. Send messages to interact with the agent
4. Use commands:
   - `/help` - Show help message
   - `/clear` - Clear conversation history

## Implementation Details

### File Structure

```
telegram_interface/
├── __init__.py                  # Package initialization
├── telegram_confirmation.py     # Telegram confirmation manager
└── README.md                    # This file

telegram_main.py                 # Main entry point
```

### Key Components

1. **HasiniTelegramBot** (`telegram_main.py`)
   - Main bot class managing lifecycle
   - Handles Telegram updates and messages
   - Manages per-user conversation memories

2. **TelegramConfirmationManager** (`telegram_interface/telegram_confirmation.py`)
   - Handles permission confirmations for gated tools
   - Designed for future inline keyboard implementation

3. **Agent Integration**
   - Uses existing `create_research_agent()` with `mode="text"`
   - Leverages `Agent_stream` for streaming responses
   - Reuses `ConversationMemory` for conversation history

### Message Flow

1. User sends message to Telegram bot
2. Bot receives update via `python-telegram-bot`
3. Message is routed to `_handle_message()`
4. User-specific conversation memory is retrieved
5. Message is processed through `Agent_stream()`
6. Response is sent back to Telegram

## Extension Points

### Adding Voice Messages

The modular design makes it easy to add voice message support:

1. Add voice message handler in `telegram_main.py`
2. Integrate with existing STT providers from `voice/providers/stt/`
3. Process transcribed text through existing agent pipeline
4. Optionally add TTS for voice responses

### Custom Commands

Add new commands by registering handlers in `initialize()`:

```python
self.application.add_handler(CommandHandler("custom", self._handle_custom))
```

### Inline Keyboards for Confirmation

The `TelegramConfirmationManager` is designed to support inline keyboard confirmations. Implement this by:

1. Adding callback query handlers
2. Sending inline keyboards with confirmation buttons
3. Handling callback responses in `handle_confirmation_response()`

## Troubleshooting

### Bot Token Issues

If you see `TELEGRAM_BOT_TOKEN is not set`:
- Ensure the token is in your `.env` file
- Check the token format (should be `numbers:letters`)
- Verify the token is valid by testing with BotFather

### Bot Not Responding

Check the logs for:
- Connection errors (network issues)
- Agent initialization failures
- Memory configuration issues

### Permission Errors

If permission-gated tools fail:
- The confirmation system is currently in text mode
- Users can type "yes" or "confirm" to allow actions
- Future versions will support inline keyboard confirmations

## Logging

The Telegram bot uses the same logging configuration as other interfaces:

- INFO level for normal operations
- WARNING for third-party library noise
- ERROR for failures
- DEBUG for detailed troubleshooting

Logs are formatted with timestamps and logger names for easy filtering.

## Security Considerations

- **Bot token security**: Never commit the bot token to version control
- **User isolation**: Each user has separate conversation memory
- **Permission system**: Existing permission gates are enforced
- **No privileged operations**: Bot follows the same security model as text/voice interfaces

## Comparison with Other Interfaces

| Feature | Text (`main.py`) | Voice (`voice_main.py`) | Telegram (`telegram_main.py`) |
|---------|------------------|------------------------|-------------------------------|
| Input method | Terminal input | Voice audio | Telegram messages |
| Output method | Terminal output | Voice audio | Telegram messages |
| Memory | Single user | Single user | Per-user isolation |
| Confirmation | Terminal input | Voice input | Text (inline keyboard planned) |
| Tools | All | All | All |
| Agent runtime | Shared | Shared | Shared |

## Future Enhancements

Potential improvements for the Telegram interface:

1. **Inline keyboard confirmations** for permission-gated tools
2. **Voice message support** using existing STT/TTS providers
3. **File/document analysis** capabilities
4. **Image processing** integration
5. **Group chat support** with @mentions
6. **Webhook mode** for deployment (vs. current polling)
7. **Rich formatting** for responses (Markdown, HTML)
8. **Button-based tool selection** for complex tools

## Contributing

When extending the Telegram interface:

1. Follow the existing architecture patterns
2. Reuse existing agent components where possible
3. Add proper error handling and logging
4. Update this README with new features
5. Test with multiple users for memory isolation

## License

This Telegram integration follows the same license as the main Hasini AI Research Agent project.
