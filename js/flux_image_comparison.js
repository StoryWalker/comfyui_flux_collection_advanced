import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/**
 * Flux Image Comparison (Architectural Fix):
 * Uses a persistent widget and manual canvas drawing for total control.
 */

app.registerExtension({
	name: "flux_collection_advanced.FluxImageComparison",
	async beforeRegisterNodeDef(nodeDef, nodeData, app) {
		if (nodeData.name === "FluxImageComparison") {
			
			// 1. Set default size
			const onNodeCreated = nodeDef.prototype.onNodeCreated;
			nodeDef.prototype.onNodeCreated = function() {
				if (onNodeCreated) onNodeCreated.apply(this, arguments);
				this.setSize([400, 480]);
				this.sliderPos = 0.5;
                this.imgs = [null, null];
			};

			// 2. Handle execution data
			const onExecuted = nodeDef.prototype.onExecuted;
			nodeDef.prototype.onExecuted = function(message) {
				onExecuted?.apply(this, arguments);
				if (message.flux_compare_data) {
					const urls = message.flux_compare_data.map(img => 
						api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${encodeURIComponent(img.subfolder)}`)
					);
                    
                    this.imgs = [new Image(), new Image()];
                    this.imgs[0].onload = () => this.setDirtyCanvas(true);
                    this.imgs[1].onload = () => this.setDirtyCanvas(true);
                    this.imgs[0].src = urls[0];
                    this.imgs[1].src = urls[1];
				}
			};

			// 3. Robust Background Drawing
			nodeDef.prototype.onDrawBackground = function(ctx) {
				if (!this.imgs || !this.imgs[0]?.complete || !this.imgs[1]?.complete) {
                    ctx.fillStyle = "#222";
                    ctx.fillRect(10, 40, this.size[0]-20, this.size[1]-50);
                    ctx.fillStyle = "#888";
                    ctx.textAlign = "center";
                    ctx.fillText("Waiting for Images...", this.size[0]/2, this.size[1]/2);
                    return;
                }

                const margin = 10;
                const top_y = 45;
                const w = this.size[0] - margin * 2;
                const h = this.size[1] - margin - top_y;

                // Calc scale (contain)
                const imgW = this.imgs[0].naturalWidth;
                const imgH = this.imgs[0].naturalHeight;
                const ratio = Math.min(w / imgW, h / imgH);
                const dw = imgW * ratio;
                const dh = imgH * ratio;
                const dx = margin + (w - dw) / 2;
                const dy = top_y + (h - dh) / 2;

                this.imgRect = { x: dx, y: dy, w: dw, h: dh };

                // Draw B (Right)
                ctx.drawImage(this.imgs[1], dx, dy, dw, dh);

                // Draw A (Left - Clipped)
                const clipX = dw * (this.sliderPos ?? 0.5);
                ctx.save();
                ctx.beginPath();
                ctx.rect(dx, dy, clipX, dh);
                ctx.clip();
                ctx.drawImage(this.imgs[0], dx, dy, dw, dh);
                ctx.restore();

                // Slider Line
                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(dx + clipX, dy);
                ctx.lineTo(dx + clipX, dy + dh);
                ctx.stroke();

                // Label
                ctx.fillStyle = "rgba(0,0,0,0.7)";
                ctx.fillRect(dx + clipX - 35, dy + 10, 70, 20);
                ctx.fillStyle = "#0FF";
                ctx.font = "bold 10px Arial";
                ctx.textAlign = "center";
                ctx.fillText(this.sliderPos > 0.5 ? "REF: A" : "RES: B", dx + clipX, dy + 24);
			};

			// 4. Input handling
			nodeDef.prototype.onMouseDown = function(e, local_pos) {
				if (this.imgRect && local_pos[0] >= this.imgRect.x && local_pos[0] <= this.imgRect.x + this.imgRect.w) {
					this.isDragging = true;
					this.sliderPos = (local_pos[0] - this.imgRect.x) / this.imgRect.w;
					this.setDirtyCanvas(true);
					return true;
				}
			};

			nodeDef.prototype.onMouseMove = function(e, local_pos) {
				if (this.isDragging) {
					this.sliderPos = Math.max(0, Math.min(1, (local_pos[0] - this.imgRect.x) / this.imgRect.w));
					this.setDirtyCanvas(true);
				}
			};

			nodeDef.prototype.onMouseUp = function() {
				this.isDragging = false;
			};
		}
	},
});
