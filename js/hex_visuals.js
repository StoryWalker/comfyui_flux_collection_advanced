// Task-Source: T#22 / T#23 / T#24
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { hexTheme } from "./hex_theme.js";

export const HEX_NODES = [
	"WanUnifiedLoaderHex",
	"WanStorySamplerHex",
	"ImageLoaderHex",
	"LoopFetcherHex",
	"LoopStorageHex",
	"PromptSequencerHex",
	"WanVideoSaverHex",
	"FluxGGUFLoader",
	"FluxTextPromptHex",
	"FluxSamplerParametersHex",
	"GlobalSeedHex",
	"FluxControlNetLoaderHex",
	"FluxControlNetApplyHex",
	"FluxImagePreviewHex",
	"FluxControlNetApplyPreviewHex",
	"FluxImageUpscalerHex",
	"FluxLoraDetailerHex",
	"FluxVRAMLoaderBetaHex",
	"WanIndexBridgeHex",
	"WanVideoSaverDevHex",
	"NVFP4NativeLoaderHex",
	"ImageLoadEncodeHex",
	"LTXLoaderHex",
	"LTXSamplerHex",
	"LTXVideoSaverHex",
];

// Definición centralizada de settings — usada al registrar y al restaurar desde el nodo
export const HEX_SETTINGS = [
	{ id: "HEX.Theme.textColor",       name: "HEX › Section text color",       type: "text",   group: "section", key: "textColor",       def: hexTheme.section.textColor       },
	{ id: "HEX.Theme.backgroundColor", name: "HEX › Section background color", type: "text",   group: "section", key: "backgroundColor", def: hexTheme.section.backgroundColor },
	{ id: "HEX.Theme.borderColor",     name: "HEX › Section border color",     type: "text",   group: "section", key: "borderColor",     def: hexTheme.section.borderColor     },
	{ id: "HEX.Theme.font",            name: "HEX › Section font",             type: "text",   group: "section", key: "font",            def: hexTheme.section.font            },
	{ id: "HEX.Theme.lineWidth",       name: "HEX › Section line width",       type: "number", group: "section", key: "lineWidth",       def: hexTheme.section.lineWidth       },
	{ id: "HEX.Theme.minWidth",        name: "HEX › Node minimum width",       type: "number", group: "node",    key: "minWidth",        def: hexTheme.node.minWidth           },
];

app.registerExtension({
	name: "Antigravity.HexVisuals",

	setup() {
		// Tema activo en runtime — punto único de verdad para todos los nodos HEX
		app._hexTheme = {
			section: { ...hexTheme.section },
			node:    { ...hexTheme.node },
		};

		// Broadcast compartido — usado por Settings y por HexThemeHex
		app._hexBroadcast = function(theme) {
			if (!app.graph) return;
			const minW = theme.node.minWidth;
			for (const node of app.graph.nodes) {
				if (!HEX_NODES.includes(node.type)) continue;
				if (node.size[0] < minW) node.size[0] = minW;
				node.setDirtyCanvas(true, false);
			}
			app.graph.setDirtyCanvas(true, false);
		};

		// Registrar panel de Settings + aplicar valores persistidos al arranque
		for (const s of HEX_SETTINGS) {
			app.ui.settings.addSetting({
				id:           s.id,
				name:         s.name,
				type:         s.type,
				defaultValue: s.def,
				onChange(value) {
					if (!app._hexTheme) return;
					// El nodo HexThemeHex tiene prioridad; solo aplica si no hay nodo activo
					if (app._hexThemeNodeActive) return;
					app._hexTheme[s.group][s.key] = s.type === "number" ? Number(value) : value;
					app._hexBroadcast?.(app._hexTheme);
				},
			});

			// Aplica el valor persistido como base al iniciar
			const saved = app.ui.settings.getSettingValue(s.id, s.def);
			app._hexTheme[s.group][s.key] = s.type === "number" ? Number(saved) : saved;
		}

		app._hexThemeNodeActive = false;
	},

	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (!HEX_NODES.includes(nodeData.name)) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
			this.size[0] = app._hexTheme.node.minWidth;
			this.widgets.forEach(w => {
				if (!w.name?.startsWith("section_")) return;
				if (w.element) w.element.style.display = "none";

				w.draw = function(ctx, node, widget_width, y, widget_height) {
					const { section } = app._hexTheme;
					const margin = 10;

					ctx.fillStyle = section.backgroundColor;
					ctx.fillRect(0, y, widget_width, widget_height);

					ctx.fillStyle = section.textColor;
					ctx.font = section.font;
					ctx.textAlign = "center";
					const label = w.value || w.name.replace("section_", "").replace("_", " ").toUpperCase();
					ctx.fillText(label, widget_width / 2, y + widget_height / 1.5);

					ctx.strokeStyle = section.borderColor;
					ctx.lineWidth = section.lineWidth;
					ctx.beginPath();
					ctx.moveTo(margin, y + widget_height - 1);
					ctx.lineTo(widget_width - margin, y + widget_height - 1);
					ctx.stroke();
				};
			});
			return r;
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
			this.size[0] = app._hexTheme.node.minWidth;
			return r;
		};

		const computeSize = nodeType.prototype.computeSize;
		nodeType.prototype.computeSize = function () {
			const minW = app._hexTheme.node.minWidth;
			const size = computeSize ? computeSize.apply(this, arguments) : [minW, 200];
			size[0] = Math.max(size[0], minW);
			return size;
		};
	},
});

// --- TIMER DE EJECUCIÓN DEL PIPELINE PARA LA GUI ---
(function() {
	const style = document.createElement("style");
	style.textContent = `
		#hex-execution-timer {
			font-family: 'Outfit', 'Inter', 'Segoe UI', sans-serif;
			background: rgba(20, 20, 20, 0.85);
			backdrop-filter: blur(12px);
			-webkit-backdrop-filter: blur(12px);
			border: 1px solid rgba(255, 144, 0, 0.4);
			border-radius: 30px;
			padding: 10px 20px;
			color: #fff;
			box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), 0 0 15px rgba(255, 144, 0, 0.2);
			display: flex;
			align-items: center;
			gap: 12px;
			transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
			opacity: 0;
			transform: translate(-50%, -20px) scale(0.9);
			position: fixed;
			top: 20px;
			left: 50%;
			z-index: 10000;
			pointer-events: none;
			font-size: 14px;
			font-weight: 500;
			letter-spacing: 0.5px;
		}
		#hex-execution-timer.visible {
			opacity: 1;
			transform: translate(-50%, 0) scale(1);
		}
		#hex-execution-timer .timer-dot {
			width: 10px;
			height: 10px;
			border-radius: 50%;
			background-color: #ff9000;
			box-shadow: 0 0 8px #ff9000;
			transition: all 0.3s ease;
		}
		#hex-execution-timer.running .timer-dot {
			background-color: #00f0ff;
			box-shadow: 0 0 10px #00f0ff;
			animation: hex-pulse 1s infinite alternate;
		}
		#hex-execution-timer.success {
			border-color: rgba(0, 255, 150, 0.5);
			box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), 0 0 15px rgba(0, 255, 150, 0.25);
		}
		#hex-execution-timer.success .timer-dot {
			background-color: #00ff96;
			box-shadow: 0 0 10px #00ff96;
		}
		#hex-execution-timer.error {
			border-color: rgba(255, 70, 70, 0.5);
			box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), 0 0 15px rgba(255, 70, 70, 0.25);
		}
		#hex-execution-timer.error .timer-dot {
			background-color: #ff4646;
			box-shadow: 0 0 10px #ff4646;
		}
		#hex-execution-timer .timer-label {
			color: rgba(255, 255, 255, 0.7);
			font-size: 12px;
			text-transform: uppercase;
			font-weight: 600;
		}
		#hex-execution-timer .timer-value {
			font-variant-numeric: tabular-nums;
			font-weight: bold;
		}

		@keyframes hex-pulse {
			from { transform: scale(0.8); opacity: 0.6; }
			to { transform: scale(1.25); opacity: 1; }
		}
	`;
	document.head.appendChild(style);

	const badge = document.createElement("div");
	badge.id = "hex-execution-timer";
	badge.innerHTML = `
		<div class="timer-dot"></div>
		<span class="timer-label">Pipeline</span>
		<span class="timer-value">0.0s</span>
	`;
	
	if (document.body) {
		document.body.appendChild(badge);
	} else {
		window.addEventListener("DOMContentLoaded", () => {
			document.body.appendChild(badge);
		});
	}

	let startTime = null;
	let timerInterval = null;
	let hideTimeout = null;
	let executedNodes = [];
	let pipelineActive = false;
	let currentPromptId = null;
	let eventLogTrace = [];

	function logEvent(name, detail) {
		const timestamp = new Date().toISOString().replace('T', ' ').substring(0, 19);
		const detailStr = detail ? JSON.stringify(detail) : "null";
		const logLine = `[${timestamp}] Event: ${name} | Detail: ${detailStr}`;
		eventLogTrace.push(logLine);
		console.log(`[HEX Pipeline Tracer] ${logLine}`);
	}

	function resetPipelineTracking(promptId) {
		executedNodes = [];
		eventLogTrace = [];
		startTime = performance.now();
		pipelineActive = true;
		currentPromptId = promptId;
		logEvent("track_start", { promptId });
	}

	function updateUI(text, state) {
		const valEl = badge.querySelector(".timer-value");
		if (valEl) valEl.textContent = text;
		
		badge.className = ""; // clear all
		badge.classList.add("visible");
		badge.classList.add(state);
	}

	function startTimer(promptId) {
		if (hideTimeout) clearTimeout(hideTimeout);
		resetPipelineTracking(promptId);
		updateUI("0.0s", "running");
		
		if (timerInterval) clearInterval(timerInterval);
		timerInterval = setInterval(() => {
			if (startTime) {
				const elapsed = (performance.now() - startTime) / 1000;
				updateUI(`${elapsed.toFixed(1)}s`, "running");
			}
		}, 100);
	}

	function stopTimer(success = true) {
		if (timerInterval) {
			clearInterval(timerInterval);
			timerInterval = null;
		}
		if (startTime) {
			const elapsed = (performance.now() - startTime) / 1000;
			updateUI(`${elapsed.toFixed(2)}s`, success ? "success" : "error");
			startTime = null;
			
			hideTimeout = setTimeout(() => {
				badge.classList.remove("visible");
			}, 8000);
		}
	}

	function sendPipelineLog(success, endStatus = "SUCCESS") {
		const endTime = performance.now();
		const totalTime = startTime ? (endTime - startTime) / 1000 : 0;
		
		const dateStr = new Date().toISOString().replace('T', ' ').substring(0, 19);
		let log = `[${dateStr}] === PIPELINE EXECUTION STARTED (Prompt ID: ${currentPromptId}) ===\n`;
		
		log += "--- DETAILED EVENT TRACE ---\n";
		eventLogTrace.forEach(line => {
			log += `  ${line}\n`;
		});
		log += "----------------------------\n";
		
		if (executedNodes.length > 0) {
			log += "--- EXECUTED NODES ---\n";
			executedNodes.forEach(n => {
				log += `  - [${n.time}] Node: ${n.type} (ID: ${n.id})\n`;
			});
		} else {
			log += "  - No nodes registered during execution.\n";
		}
		
		log += `[${dateStr}] === PIPELINE EXECUTION FINISHED | Total Time: ${totalTime.toFixed(2)}s | Status: ${endStatus} ===\n`;
		
		api.fetchApi("/hex/log_pipeline", {
			method: "POST",
			headers: {
				"Content-Type": "application/json"
			},
			body: JSON.stringify({ log })
		}).catch(err => {
			console.error("[HEX Performance Log] Failed to send pipeline log:", err);
		});
	}

	// Registrar listeners en la instancia API del WebSocket de ComfyUI
	api.addEventListener("execution_start", (event) => {
		const detail = event.detail;
		const promptId = detail ? detail.prompt_id : null;
		logEvent("execution_start", detail);
		startTimer(promptId);
	});

	api.addEventListener("executing", (event) => {
		const detail = event.detail;
		logEvent("executing", detail);
		
		const nodeId = detail ? detail.node : null;
		const promptId = detail ? detail.prompt_id : null;
		
		if (promptId && currentPromptId && promptId !== currentPromptId) {
			logEvent("executing_ignored_wrong_prompt", { promptId, currentPromptId });
			return;
		}

		if (nodeId) {
			if (!pipelineActive) {
				resetPipelineTracking(promptId || currentPromptId);
			}
			const node = app.graph.getNodeById(nodeId);
			const nodeType = node ? node.type : `Node ${nodeId}`;
			const timestamp = new Date().toLocaleTimeString();
			executedNodes.push({ type: nodeType, id: nodeId, time: timestamp });
		}
	});

	api.addEventListener("executed", (event) => {
		const detail = event.detail;
		logEvent("executed", detail);
	});

	api.addEventListener("execution_success", (event) => {
		const detail = event.detail;
		const promptId = detail ? detail.prompt_id : null;
		logEvent("execution_success", detail);
		
		if (promptId && currentPromptId && promptId !== currentPromptId) {
			logEvent("execution_success_ignored_wrong_prompt", { promptId, currentPromptId });
			return;
		}

		if (pipelineActive) {
			sendPipelineLog(true, "SUCCESS");
			stopTimer(true);
			pipelineActive = false;
			currentPromptId = null;
		}
	});

	api.addEventListener("execution_error", (event) => {
		const detail = event.detail;
		const promptId = detail ? detail.prompt_id : null;
		logEvent("execution_error", detail);
		
		if (promptId && currentPromptId && promptId !== currentPromptId) {
			logEvent("execution_error_ignored_wrong_prompt", { promptId, currentPromptId });
			return;
		}

		if (pipelineActive) {
			sendPipelineLog(false, "ERROR");
			stopTimer(false);
			pipelineActive = false;
			currentPromptId = null;
		}
	});

	api.addEventListener("execution_interrupted", (event) => {
		const detail = event.detail;
		const promptId = detail ? detail.prompt_id : null;
		logEvent("execution_interrupted", detail);
		
		if (promptId && currentPromptId && promptId !== currentPromptId) {
			logEvent("execution_interrupted_ignored_wrong_prompt", { promptId, currentPromptId });
			return;
		}

		if (pipelineActive) {
			sendPipelineLog(false, "INTERRUPTED");
			stopTimer(false);
			pipelineActive = false;
			currentPromptId = null;
		}
	});
})();
