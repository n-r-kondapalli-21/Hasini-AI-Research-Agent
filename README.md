# Hasini AI Research Agent

An AI-powered research agent built with LangChain that enables interactive conversations with an intelligent assistant.

## Features

- **Interactive Chat Interface**: Command-line interface for real-time conversations
- **LangChain Integration**: Built on LangChain's agent framework
- **Custom Tools**: Extensible tool system for enhanced capabilities
- **Conversation History**: Maintains context across interactions

## Project Structure

```
Hasini_Ai_Research_Agent/
├── main.py              # Main entry point and CLI interface
├── config.py            # Configuration settings
├── llm.py               # LLM initialization and setup
├── system_prompt.py     # System prompt configuration
├── tools.py             # Custom tools for the agent
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not tracked)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
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
   - Create a `.env` file in the root directory
   - Add your API keys and configuration settings

## Usage

Run the agent from the command line:

```bash
python main.py
```

Once started, you can:
- Type your questions or commands
- Type `exit` or `quit` to stop the agent

## Configuration

The agent uses several configuration files:
- `config.py` - General configuration settings
- `llm.py` - LLM model configuration
- `system_prompt.py` - System prompt for the AI agent
- `tools.py` - Custom tools and functions

## Development

To extend the agent's capabilities:
1. Add new tools in `tools.py`
2. Modify the system prompt in `system_prompt.py`
3. Adjust LLM settings in `llm.py`
4. Update configuration in `config.py`

## License

This project is provided as-is for research and educational purposes.
