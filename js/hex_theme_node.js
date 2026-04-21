// Task-Source: T#23 / T#24
import { app } from "../../../scripts/app.js";
import { HEX_SETTINGS } from "./hex_visuals.js";

app.registerExtension({
	name: "Antigravity.HexThemeNode",

	registerCustomNodes() {
		class HexThemeNode extends LGraphNode {
			constructor() {
				super();
				this.title   = "[HEX] Theme";
				this.color   = "#2a1500";
				this.bgcolor = "#1a0e00";
				this.size    = [340, 240];
				this.serialize_widgets = true;

				this.addWidget("text",   "textColor",       "#ff9000",        () => this._apply());
				this.addWidget("text",   "backgroundColor", "#1a1a1a",        () => this._apply());
				this.addWidget("text",   "borderColor",     "#ff9000",        () => this._apply());
				this.addWidget("text",   "font",            "bold 11px Arial",() => this._apply());
				this.addWidget("number", "lineWidth",       1,                () => this._apply(), { min: 0.5, max: 5,   step: 0.5, precision: 1 });
				this.addWidget("number", "minWidth",        400,              () => this._apply(), { min: 200, max: 800, step: 10,  precision: 0 });
			}

			_apply() {
				if (!app._hexTheme) return;
				const [textColor, backgroundColor, borderColor, font, lineWidth, minWidth] =
					this.widgets.map(w => w.value);
				app._hexTheme = {
					section: { textColor, backgroundColor, borderColor, font, lineWidth: Number(lineWidth) },
					node:    { minWidth: Number(minWidth) },
				};
				app._hexThemeNodeActive = true;
				app._hexBroadcast?.(app._hexTheme);
			}

			onConfigure() {
				requestAnimationFrame(() => this._apply());
			}

			onRemoved() {
				// Restaura el tema desde el panel de Settings al eliminar el nodo
				app._hexThemeNodeActive = false;
				if (!app.ui?.settings || !app._hexTheme) return;
				for (const s of HEX_SETTINGS) {
					const val = app.ui.settings.getSettingValue(s.id, s.def);
					app._hexTheme[s.group][s.key] = s.type === "number" ? Number(val) : val;
				}
				app._hexBroadcast?.(app._hexTheme);
			}

			onDrawForeground(ctx) {
				const swatches = [
					{ label: "text",   color: this.widgets[0].value },
					{ label: "bg",     color: this.widgets[1].value },
					{ label: "border", color: this.widgets[2].value },
				];
				const y = this.size[1] - 32;
				swatches.forEach((s, i) => {
					const x = 12 + i * 100;
					ctx.fillStyle = s.color;
					ctx.fillRect(x, y, 28, 16);
					ctx.strokeStyle = "#555";
					ctx.lineWidth = 0.5;
					ctx.strokeRect(x, y, 28, 16);
					ctx.fillStyle = "#aaa";
					ctx.font = "9px Arial";
					ctx.textAlign = "left";
					ctx.fillText(s.label, x + 32, y + 12);
				});
			}
		}

		HexThemeNode.title    = "[HEX] Theme";
		HexThemeNode.category = "HEX/Utils";
		LiteGraph.registerNodeType("HexThemeHex", HexThemeNode);
	},
});
