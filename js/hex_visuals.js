import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "Antigravity.HexVisuals",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		const hexNodes = [
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
			"GlobalSeedHex"
		];

		if (hexNodes.includes(nodeData.name)) {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
				this.size[0] = 400;
				this.widgets.forEach(w => {
					if (w.name && w.name.startsWith("section_")) {
						// Ocultamos el input del widget pero mantenemos el valor para la validación
						if (w.element) w.element.style.display = "none";
						
						w.draw = function(ctx, node, widget_width, y, widget_height) {
                            const margin = 10;
                            ctx.fillStyle = "#1a1a1a";
                            ctx.fillRect(0, y, widget_width, widget_height);
                            
                            ctx.fillStyle = "#ff9000"; 
                            ctx.font = "bold 11px Arial";
                            ctx.textAlign = "center";
                            
                            let label = w.value || w.name.replace("section_", "").replace("_", " ").toUpperCase();
                            ctx.fillText(label, widget_width / 2, y + widget_height / 1.5);
                            
                            ctx.strokeStyle = "#ff9000";
                            ctx.lineWidth = 1;
                            ctx.beginPath();
                            ctx.moveTo(margin, y + widget_height - 1);
                            ctx.lineTo(widget_width - margin, y + widget_height - 1);
                            ctx.stroke();
                        };
					}
				});
				return r;
			};

			const onConfigure = nodeType.prototype.onConfigure;
			nodeType.prototype.onConfigure = function () {
				const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
				this.size[0] = 400;
				return r;
			};

			const computeSize = nodeType.prototype.computeSize;
			nodeType.prototype.computeSize = function () {
				const size = computeSize ? computeSize.apply(this, arguments) : [400, 200];
				size[0] = Math.max(size[0], 400);
				return size;
			};
		}
	},
});
