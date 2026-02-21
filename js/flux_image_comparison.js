import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/**
 * Flux Image Comparison (Minimalist Slider):
 * Simplest possible implementation to ensure it works.
 */

app.registerExtension({
	name: "flux_collection_advanced.FluxImageComparison",
	async beforeRegisterNodeDef(nodeDef, nodeData, app) {
		if (nodeData.name === "FluxImageComparison") {
			
			nodeDef.prototype.onNodeCreated = function() {
				this.setSize([400, 400]);
				this.slider = 0.5;
                this.imgA = new Image();
                this.imgB = new Image();
			};

			nodeDef.prototype.onExecuted = function(m) {
                if (m.a?.[0] && m.b?.[0]) {
                    this.imgA.src = api.apiURL(`/view?filename=${m.a[0].filename}&type=temp&subfolder=`);
                    this.imgB.src = api.apiURL(`/view?filename=${m.b[0].filename}&type=temp&subfolder=`);
                    this.imgA.onload = () => this.setDirtyCanvas(true);
                    this.imgB.onload = () => this.setDirtyCanvas(true);
                }
			};

			nodeDef.prototype.onDrawBackground = function(ctx) {
				if (!this.imgA?.complete || !this.imgB?.complete) return;
				
                const w = this.size[0];
                const h = this.size[1] - 40; // Espacio para el tÃ­tulo
                const y = 40;

				// Dibujar B (Derecha)
				ctx.drawImage(this.imgB, 0, y, w, h);

				// Dibujar A (Izquierda con recorte)
				ctx.save();
				ctx.beginPath();
				ctx.rect(0, y, w * this.slider, h);
				ctx.clip();
				ctx.drawImage(this.imgA, 0, y, w, h);
				ctx.restore();

				// LÃ­nea del Slider
				ctx.strokeStyle = "#0F0";
				ctx.lineWidth = 3;
				ctx.beginPath();
				ctx.moveTo(w * this.slider, y);
				ctx.lineTo(w * this.slider, y + h);
				ctx.stroke();
			};

			nodeDef.prototype.onMouseDown = function(e, pos) {
				if (pos[1] > 40) {
					this.dragging = true;
					this.slider = Math.max(0, Math.min(1, pos[0] / this.size[0]));
					this.setDirtyCanvas(true);
					return true;
				}
			};

			nodeDef.prototype.onMouseMove = function(e, pos) {
				if (this.dragging) {
					this.slider = Math.max(0, Math.min(1, pos[0] / this.size[0]));
					this.setDirtyCanvas(true);
				}
			};

			nodeDef.prototype.onMouseUp = function() {
				this.dragging = false;
			};
		}
	}
});
