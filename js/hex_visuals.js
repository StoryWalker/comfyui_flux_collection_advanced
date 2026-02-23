import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "Antigravity.HexVisuals",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		// Aplicar a todos nuestros nodos hexagonales
		if (nodeData.category && nodeData.category.includes("flux_collection_advanced/hex")) {
			
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				// Buscar widgets que son separadores
				this.widgets.forEach(w => {
					if (w.name && w.name.startsWith("section_")) {
						w.type = "HEX_HEADER"; // Cambiar tipo para evitar interacciones
						if (w.inputEl) w.inputEl.style.display = "none"; 
						
						w.draw = function(ctx, node, widget_width, y, widget_height) {
                            ctx.fillStyle = "#333";
                            ctx.fillRect(0, y, widget_width, widget_height);
                            ctx.fillStyle = "#ff9000"; // Naranja Antigravity
                            ctx.font = "bold 12px Arial";
                            ctx.textAlign = "center";
                            ctx.fillText(w.value.toString().replace("[ ", "").replace(" ]", ""), widget_width / 2, y + widget_height / 1.5);
                            
                            // Línea inferior naranja
                            ctx.strokeStyle = "#ff9000";
                            ctx.beginPath();
                            ctx.moveTo(10, y + widget_height - 1);
                            ctx.lineTo(widget_width - 10, y + widget_height - 1);
                            ctx.stroke();
                        };
					}
				});

				return r;
			};
		}
	},
});
