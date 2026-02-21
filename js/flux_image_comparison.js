import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/**
 * Flux Image Comparison (Hover Mode):
 * Interactive comparison that activates on hover once the node is used.
 * Gray slider, auto-reset, and no-click requirement.
 */

app.registerExtension({
	name: "flux_collection_advanced.FluxImageComparison",
	async beforeRegisterNodeDef(nodeDef, nodeData, app) {
		if (nodeData.name === "FluxImageComparison") {
			
			nodeDef.prototype.onNodeCreated = function() {
				this.setSize([400, 400]);
				this.slider = 1.0; 
                this.imgA = new Image();
                this.imgB = new Image();
                this.imgRect = { x: 0, y: 0, w: 1, h: 1 };
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
                const h = this.size[1] - 40; 
                const y_start = 40;

                const imgW = this.imgA.naturalWidth;
                const imgH = this.imgA.naturalHeight;
                const ratio = Math.min(w / imgW, h / imgH);
                
                const dw = imgW * ratio;
                const dh = imgH * ratio;
                const dx = (w - dw) / 2;
                const dy = y_start + (h - dh) / 2;

                this.imgRect = { x: dx, y: dy, w: dw, h: dh };

                // 1. Draw Image B (Target/Right)
				ctx.drawImage(this.imgB, dx, dy, dw, dh);

				// 2. Draw Image A (Reference/Left with Clip)
                const clipX = dw * this.slider;
				ctx.save();
				ctx.beginPath();
				ctx.rect(dx, dy, clipX, dh);
				ctx.clip();
				ctx.drawImage(this.imgA, dx, dy, dw, dh);
				ctx.restore();

				// 3. Gray Slider Line
				ctx.strokeStyle = "#888";
				ctx.lineWidth = 2;
				ctx.beginPath();
				ctx.moveTo(dx + clipX, dy);
				ctx.lineTo(dx + clipX, dy + dh);
				ctx.stroke();
			};

			// Instant alignment on mouse move (No click needed)
			nodeDef.prototype.onMouseMove = function(e, pos) {
				if (this.imgRect) {
                    // Check if mouse is within horizontal image bounds
                    if (pos[0] >= this.imgRect.x && pos[0] <= this.imgRect.x + this.imgRect.w) {
					    this.slider = (pos[0] - this.imgRect.x) / this.imgRect.w;
					    this.setDirtyCanvas(true);
                        return true;
                    }
				}
			};

            // Ensure node gets focus on click
            nodeDef.prototype.onMouseDown = function() {
                return true; 
            };

            // Auto-Reset when mouse leaves the node area
            nodeDef.prototype.onMouseLeave = function() {
                this.slider = 1.0;
                this.setDirtyCanvas(true);
            };
		}
	}
});
