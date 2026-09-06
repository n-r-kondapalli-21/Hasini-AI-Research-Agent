"""
JARVIS HUD UI Server.

Serves static web UI assets and maintains real-time WebSocket communication
with the browser visualizer on localhost:8765.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
from pathlib import Path
from typing import Optional, Set

import websockets
from websockets.asyncio.server import ServerConnection
from websockets.http11 import Headers, Response

logger = logging.getLogger("hasini.voice.ui_server")

# Suppress noisy HTTP connection logging from websockets.server
logging.getLogger("websockets.server").setLevel(logging.WARNING)
logging.getLogger("websockets.protocol").setLevel(logging.WARNING)

WEB_DIR = Path(__file__).parent / "web"


class UIServer:
    """
    Lightweight non-blocking HTTP & WebSocket server for the JARVIS HUD UI.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[ServerConnection] = set()
        self.server = None

        self.current_state: str = "IDLE"
        self.user_command: str = ""
        self.agent_response: str = ""
        self.error_message: Optional[str] = None

    async def _process_request(self, connection, request):
        """Serve HTTP static files for non-websocket paths."""
        path_str = request.path.split("?")[0]
        if path_str == "/" or path_str == "":
            path_str = "/index.html"

        # Bypass websocket upgrade requests
        if path_str == "/ws" or path_str.startswith("/ws/"):
            return None

        # Clean file path
        rel_path = path_str.lstrip("/")
        target_file = WEB_DIR / rel_path

        if target_file.is_file() and (target_file == WEB_DIR or WEB_DIR in target_file.parents):
            content_type, _ = mimetypes.guess_type(target_file)
            content_type = content_type or "application/octet-stream"
            content = target_file.read_bytes()

            headers = Headers([
                ("Content-Type", content_type),
                ("Content-Length", str(len(content))),
                ("Access-Control-Allow-Origin", "*"),
            ])
            return Response(200, "OK", headers, content)

        # 404 Not Found fallback
        return Response(404, "Not Found", Headers([("Content-Type", "text/plain")]), b"404 Not Found")

    async def _handle_connection(self, websocket: ServerConnection):
        """WebSocket connection handler."""
        self.clients.add(websocket)
        logger.info("JARVIS HUD WebSocket client connected (%s). Total clients: %d", websocket.remote_address, len(self.clients))

        # Send initial snapshot
        snapshot = {
            "type": "state_update",
            "state": self.current_state,
            "user_command": self.user_command,
            "agent_response": self.agent_response,
            "error": self.error_message,
        }

        try:
            await websocket.send(json.dumps(snapshot))
            async for message in websocket:
                pass
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            logger.debug("WebSocket client error: %s", e)
        finally:
            self.clients.discard(websocket)
            logger.info("JARVIS HUD WebSocket client disconnected.")

    async def start(self) -> None:
        """Start the HTTP & WebSocket server."""
        if self.server:
            return

        last_error = None
        for try_port in range(self.port, self.port + 10):
            try:
                self.server = await websockets.serve(
                    self._handle_connection,
                    self.host,
                    try_port,
                    process_request=self._process_request,
                )
                self.port = try_port
                logger.info("JARVIS HUD UI Server running at http://%s:%d", self.host, self.port)
                return
            except OSError as e:
                last_error = e

        if last_error:
            raise last_error

    async def stop(self) -> None:
        """Stop the UI server cleanly."""
        if not self.server:
            return
        self.server.close()
        await self.server.wait_closed()
        self.server = None
        logger.info("JARVIS HUD UI Server stopped.")

    async def broadcast(self, data: dict) -> None:
        """Broadcast payload to all connected UI clients."""
        if not self.clients:
            return
        payload = json.dumps(data)
        dead_clients = []
        for client in list(self.clients):
            try:
                await client.send(payload)
            except Exception:
                dead_clients.append(client)

        for client in dead_clients:
            self.clients.discard(client)

    def update_state(
        self,
        state: Optional[str] = None,
        user_command: Optional[str] = None,
        agent_response: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Synchronously update state data and schedule async broadcast."""
        if state is not None:
            self.current_state = state
        if user_command is not None:
            self.user_command = user_command
        if agent_response is not None:
            self.agent_response = agent_response
        if error is not None:
            self.error_message = error

        data = {
            "type": "state_update",
            "state": self.current_state,
            "user_command": self.user_command,
            "agent_response": self.agent_response,
            "error": self.error_message,
        }

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcast(data))
        except RuntimeError:
            pass

    def send_token(self, token: str) -> None:
        """Stream response token to UI."""
        self.agent_response += token
        data = {
            "type": "token",
            "token": token,
        }
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcast(data))
        except RuntimeError:
            pass
