/* ==========================================================================
   JARVIS ARC REACTOR VISUALIZER ENGINE & WEBSOCKET CLIENT
   ========================================================================== */

(function () {
    // ----------------------------------------------------------------------
    // STATE & CONFIG
    // ----------------------------------------------------------------------
    let currentState = "IDLE"; // IDLE, LISTENING, PROCESSING, SPEAKING, ERROR
    let userCommandText = "";
    let agentResponseText = "";
    let errorMessage = "";
    let lastAudioLevel = null;

    // DOM Elements
    const jarvisContainer = document.getElementById("jarvisContainer");
    const canvas = document.getElementById("jarvisCanvas");
    const ctx = canvas.getContext("2d");
    const statusBadge = document.getElementById("statusBadge");
    const statusText = document.getElementById("statusText");
    const statusDot = document.getElementById("statusIndicator");
    const rawStateEl = document.getElementById("rawState");
    const fpsMeterEl = document.getElementById("fpsMeter");
    const userCommandEl = document.getElementById("userCommandText");
    const agentResponseEl = document.getElementById("agentResponseText");
    const coreLabel = document.getElementById("coreLabel");
    const coreSubLabel = document.getElementById("coreSubLabel");
    const vadStateEl = document.getElementById("vadState");
    const wsDot = document.getElementById("wsDot");
    const wsStatusText = document.getElementById("wsStatusText");
    const waveBars = document.querySelectorAll(".wave-bar");

    // Colors matching CSS states
    const STATE_COLORS = {
        IDLE: { main: "#00f3ff", sub: "#005577", glow: "rgba(0, 243, 255, 0.4)" },
        LISTENING: { main: "#00ff66", sub: "#008833", glow: "rgba(0, 255, 102, 0.5)" },
        PROCESSING: { main: "#00bfff", sub: "#0066cc", glow: "rgba(0, 191, 255, 0.6)" },
        SPEAKING: { main: "#d946ef", sub: "#8b5cf6", glow: "rgba(217, 70, 239, 0.6)" },
        ERROR: { main: "#ff3344", sub: "#990011", glow: "rgba(255, 51, 68, 0.7)" }
    };

    // ----------------------------------------------------------------------
    // CANVAS ANIMATION ENGINE
    // ----------------------------------------------------------------------
    let angle = 0;
    let particles = [];
    let lastTime = performance.now();
    let frameCount = 0;
    let fps = 60;

    // Create background floating tech particles
    for (let i = 0; i < 40; i++) {
        particles.push({
            x: (Math.random() - 0.5) * 450,
            y: (Math.random() - 0.5) * 450,
            size: Math.random() * 2 + 1,
            speedX: (Math.random() - 0.5) * 0.4,
            speedY: (Math.random() - 0.5) * 0.4,
            alpha: Math.random() * 0.6 + 0.2
        });
    }

    function renderArcReactor() {
        const now = performance.now();
        frameCount++;
        if (now - lastTime >= 1000) {
            fps = Math.round((frameCount * 1000) / (now - lastTime));
            fpsMeterEl.textContent = fps;
            frameCount = 0;
            lastTime = now;
        }

        const width = canvas.width;
        const height = canvas.height;
        const cx = width / 2;
        const cy = height / 2;

        ctx.clearRect(0, 0, width, height);

        const palette = STATE_COLORS[currentState] || STATE_COLORS.IDLE;
        const rotationSpeed = currentState === "PROCESSING" ? 0.04 : (currentState === "SPEAKING" ? 0.02 : 0.008);

        angle += rotationSpeed;

        // 1. DYNAMIC PARTICLES
        ctx.save();
        ctx.translate(cx, cy);
        particles.forEach(p => {
            p.x += p.speedX;
            p.y += p.speedY;
            const dist = Math.sqrt(p.x * p.x + p.y * p.y);
            if (dist > 230) {
                p.x = (Math.random() - 0.5) * 100;
                p.y = (Math.random() - 0.5) * 100;
            }
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = palette.main;
            ctx.globalAlpha = p.alpha * 0.5;
            ctx.fill();
        });
        ctx.restore();

        // 2. OUTER TARGET MARKS & TICKS
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(-angle * 0.5);

        ctx.strokeStyle = palette.sub;
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.4;
        ctx.beginPath();
        ctx.arc(0, 0, 230, 0, Math.PI * 2);
        ctx.stroke();

        // Outer radial ticks
        for (let i = 0; i < 36; i++) {
            const rad = (i * Math.PI) / 18;
            const r1 = 220;
            const r2 = i % 3 === 0 ? 232 : 225;
            ctx.beginPath();
            ctx.moveTo(Math.cos(rad) * r1, Math.sin(rad) * r1);
            ctx.lineTo(Math.cos(rad) * r2, Math.sin(rad) * r2);
            ctx.strokeStyle = palette.main;
            ctx.globalAlpha = i % 3 === 0 ? 0.8 : 0.3;
            ctx.lineWidth = i % 3 === 0 ? 2 : 1;
            ctx.stroke();
        }
        ctx.restore();

        // 3. SEGMENTED MIDDLE ARC RINGS
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(angle);

        ctx.shadowColor = palette.main;
        ctx.shadowBlur = 15;

        for (let i = 0; i < 4; i++) {
            const startArc = (i * Math.PI) / 2 + 0.2;
            const endArc = startArc + Math.PI / 2 - 0.4;
            ctx.beginPath();
            ctx.arc(0, 0, 180, startArc, endArc);
            ctx.strokeStyle = palette.main;
            ctx.lineWidth = 4;
            ctx.globalAlpha = 0.85;
            ctx.stroke();
        }

        // Inner counter-rotating ring
        ctx.rotate(-angle * 2.2);
        for (let i = 0; i < 6; i++) {
            const startArc = (i * Math.PI) / 3 + 0.15;
            const endArc = startArc + Math.PI / 3 - 0.3;
            ctx.beginPath();
            ctx.arc(0, 0, 140, startArc, endArc);
            ctx.strokeStyle = palette.sub;
            ctx.lineWidth = 3;
            ctx.globalAlpha = 0.6;
            ctx.stroke();
        }
        ctx.restore();

        // 4. SPEAKING / LISTENING WAVE ANIMATION
        if (currentState === "SPEAKING" || currentState === "LISTENING") {
            const pulse = Math.sin(now * 0.008) * 15 + 90;
            ctx.save();
            ctx.translate(cx, cy);
            ctx.beginPath();
            ctx.arc(0, 0, pulse, 0, Math.PI * 2);
            ctx.strokeStyle = palette.main;
            ctx.lineWidth = 3;
            ctx.globalAlpha = 0.6;
            ctx.shadowColor = palette.main;
            ctx.shadowBlur = 20;
            ctx.stroke();
            ctx.restore();
        }

        // 5. INNER ARC CORE LENS
        ctx.save();
        ctx.translate(cx, cy);
        const coreGlow = Math.sin(now * 0.004) * 8 + 65;

        const grad = ctx.createRadialGradient(0, 0, 10, 0, 0, coreGlow);
        grad.addColorStop(0, "#ffffff");
        grad.addColorStop(0.4, palette.main);
        grad.addColorStop(1, "transparent");

        ctx.beginPath();
        ctx.arc(0, 0, coreGlow, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.globalAlpha = 0.85;
        ctx.fill();

        ctx.strokeStyle = palette.main;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();

        // 6. UPDATE AUDIO WAVEFORM ON LISTENING
        updateAudioWaveform(now);

        requestAnimationFrame(renderArcReactor);
    }

    requestAnimationFrame(renderArcReactor);

    // ----------------------------------------------------------------------
    // AUDIO WAVEFORM ANIMATION (LISTENING)
    // ----------------------------------------------------------------------
    function updateAudioWaveform(now) {
        if (!waveBars || !waveBars.length) return;

        waveBars.forEach((bar, i) => {
            let h = 4;
            if (lastAudioLevel !== null && lastAudioLevel !== undefined) {
                h = Math.max(4, Math.min(22, lastAudioLevel * 22 * (0.6 + 0.4 * Math.sin(i * 1.5))));
            } else if (currentState === "LISTENING") {
                h = Math.max(4, Math.floor(Math.sin(now * 0.012 + i * 1.2) * 8 + 12));
            }
            bar.style.height = `${h}px`;
        });
    }

    // ----------------------------------------------------------------------
    // STATE UPDATE & HUD LOGIC
    // ----------------------------------------------------------------------
    function updateUIState(state, userCmd, agentResp, err, audioLevel) {
        const oldState = currentState;
        if (state) currentState = state;
        if (userCmd !== undefined) userCommandText = userCmd;
        if (agentResp !== undefined) agentResponseText = agentResp;
        if (err !== undefined) errorMessage = err;
        if (audioLevel !== undefined) lastAudioLevel = audioLevel;

        // Smooth state change pulse micro-interaction on container
        if (oldState !== currentState) {
            document.body.className = `state-${currentState.toLowerCase()}`;
            if (jarvisContainer) {
                jarvisContainer.classList.remove("state-pulse");
                void jarvisContainer.offsetWidth; // trigger reflow
                jarvisContainer.classList.add("state-pulse");
            }
        }

        // Update raw state text
        rawStateEl.textContent = currentState;

        if (errorMessage) {
            statusText.textContent = `🔴 ERROR`;
            statusDot.style.color = STATE_COLORS.ERROR.main;
            coreLabel.textContent = "ALERT";
            coreSubLabel.textContent = errorMessage;
            vadStateEl.textContent = "ERROR";
        } else {
            switch (currentState) {
                case "LISTENING":
                    statusText.textContent = "🟢 LISTENING";
                    coreLabel.textContent = "LISTENING";
                    coreSubLabel.textContent = "SPEAK YOUR REQUEST NOW";
                    vadStateEl.textContent = "ACTIVE MIC";
                    break;
                case "PROCESSING":
                    statusText.textContent = "🔵 PROCESSING";
                    coreLabel.textContent = "THINKING";
                    coreSubLabel.textContent = "ANALYZING INTENT & TOOLS";
                    vadStateEl.textContent = "PROCESSING";
                    break;
                case "SPEAKING":
                    statusText.textContent = "🟣 SPEAKING";
                    coreLabel.textContent = "SPEAKING";
                    coreSubLabel.textContent = "SYNTHESIZING AUDIO";
                    vadStateEl.textContent = "TTS ACTIVE";
                    break;
                case "ERROR":
                    statusText.textContent = "🔴 ERROR";
                    coreLabel.textContent = "SYSTEM ERROR";
                    coreSubLabel.textContent = errorMessage || "AN ERROR OCCURRED";
                    vadStateEl.textContent = "ERROR";
                    break;
                default: // IDLE
                    statusText.textContent = "⚪ IDLE";
                    coreLabel.textContent = "HASINI";
                    coreSubLabel.textContent = "AWAITING WAKE WORD (\"HEY JARVIS\")";
                    vadStateEl.textContent = "LISTENING FOR WAKE WORD";
                    break;
            }
        }

        // Render User Command
        if (userCommandText) {
            userCommandEl.innerHTML = `<span class="user-text">"${escapeHtml(userCommandText)}"</span>`;
        } else {
            userCommandEl.innerHTML = `<span class="placeholder-text">Waiting for voice command...</span>`;
        }

        // Render Agent Response with auto-scroll
        renderAgentResponse();
    }

    function renderAgentResponse() {
        if (!agentResponseText) {
            agentResponseEl.innerHTML = `<span class="placeholder-text">Hasini's response will stream live as she speaks...</span>`;
            return;
        }

        agentResponseEl.innerHTML = `<span class="response-text">${escapeHtml(agentResponseText)}</span>`;
        
        // Auto-scroll response panel smoothly to bottom
        const panelBody = agentResponseEl.parentElement;
        if (panelBody) {
            panelBody.scrollTop = panelBody.scrollHeight;
        }
    }

    function appendResponseToken(token) {
        agentResponseText += token;

        if (agentResponseEl.querySelector(".placeholder-text")) {
            agentResponseEl.innerHTML = "";
        }

        const tokenSpan = document.createElement("span");
        tokenSpan.className = "token-fade-in";
        tokenSpan.textContent = token;
        agentResponseEl.appendChild(tokenSpan);

        // Auto-scroll to bottom
        const panelBody = agentResponseEl.parentElement;
        if (panelBody) {
            panelBody.scrollTop = panelBody.scrollHeight;
        }
    }

    function setConnectionStatus(connected) {
        if (wsDot && wsStatusText) {
            if (connected) {
                wsDot.className = "ws-dot ws-connected";
                wsStatusText.textContent = "ONLINE";
            } else {
                wsDot.className = "ws-dot ws-disconnected";
                wsStatusText.textContent = "RECONNECTING...";
            }
        }
    }

    function escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
    }

    // ----------------------------------------------------------------------
    // WEBSOCKET CLIENT CONNECTION
    // ----------------------------------------------------------------------
    let ws = null;

    function connectWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        console.log(`[JARVIS HUD] Connecting WebSocket: ${wsUrl}`);
        ws = new WebSocket(wsUrl);

        ws.onopen = function () {
            console.log("[JARVIS HUD] WebSocket connected successfully.");
            setConnectionStatus(true);
        };

        ws.onmessage = function (event) {
            try {
                const data = JSON.parse(event.data);
                if (data.type === "state_update") {
                    updateUIState(data.state, data.user_command, data.agent_response, data.error, data.audio_level);
                } else if (data.type === "token") {
                    appendResponseToken(data.token);
                }
            } catch (e) {
                console.error("[JARVIS HUD] Error parsing message:", e);
            }
        };

        ws.onclose = function () {
            console.warn("[JARVIS HUD] WebSocket closed. Retrying in 2 seconds...");
            setConnectionStatus(false);
            setTimeout(connectWebSocket, 2000);
        };

        ws.onerror = function (err) {
            console.error("[JARVIS HUD] WebSocket error:", err);
            setConnectionStatus(false);
            ws.close();
        };
    }

    connectWebSocket();

    // Initial state update
    updateUIState("IDLE", "", "", null);
})();
