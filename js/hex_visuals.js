// Task-Source: T#22 / T#23 / T#24
import { app } from "../../../scripts/app.js";
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
