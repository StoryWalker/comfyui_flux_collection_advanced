import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/**
 * Flux Image Comparison (Aspect Ratio & Precision Fix):
 * Perfect for 672x384 or any non-square resolution.
 */

app.registerExtension({
	name: "flux_collection_advanced.FluxImageComparison",
	async beforeRegisterNodeDef(nodeDef, nodeData, app) {
		if (nodeData.name === "FluxImageComparison") {
			const onExecuted = nodeDef.prototype.onExecuted;
			nodeDef.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

				if (this.widgets) {
					this.widgets = this.widgets.filter(w => w.name !== "compare_preview");
				}

				if (message.a_images && message.b_images && message.a_images.length > 0 && message.b_images.length > 0) {
					const imgA = message.a_images[0];
					const imgB = message.b_images[0];

					const urlA = api.apiURL(`/view?filename=${encodeURIComponent(imgA.filename)}&type=${imgA.type}&subfolder=${encodeURIComponent(imgA.subfolder)}`);
					const urlB = api.apiURL(`/view?filename=${encodeURIComponent(imgB.filename)}&type=${imgB.type}&subfolder=${encodeURIComponent(imgB.subfolder)}`);

					const widget = {
						type: "custom_preview",
						name: "compare_preview",
						sliderPos: 0.5,
                        // Store dimensions for mouse precision
                        imgRect: { x: 0, y: 0, w: 1, h: 1 }, 
						draw(ctx, node, widget_width, y, widget_height) {
							const margin = 10;
							const top_margin = 30;
							const drawWidth = widget_width - margin * 2;
							const drawHeight = widget_height - margin * 2 - top_margin;

							if (!this.imgA_obj || this.imgA_obj._src !== urlA) {
								this.imgA_obj = new Image();
								this.imgA_obj._src = urlA;
								this.imgA_obj.onload = () => node.setDirtyCanvas(true);
								this.imgA_obj.src = urlA;
							}

							if (!this.imgB_obj || this.imgB_obj._src !== urlB) {
								this.imgB_obj = new Image();
								this.imgB_obj._src = urlB;
								this.imgB_obj.onload = () => node.setDirtyCanvas(true);
								this.imgB_obj.src = urlB;
							}

							ctx.fillStyle = "#000";
							ctx.fillRect(margin, y + top_margin, drawWidth, drawHeight);

							if (this.imgA_obj.complete && this.imgB_obj.complete) {
                                const imgW = this.imgA_obj.naturalWidth;
                                const imgH = this.imgA_obj.naturalHeight;
                                const ratio = Math.min(drawWidth / imgW, drawHeight / imgH);
                                
                                this.imgRect.w = imgW * ratio;
                                this.imgRect.h = imgH * ratio;
                                this.imgRect.x = margin + (drawWidth - this.imgRect.w) / 2;
                                this.imgRect.y = y + top_margin + (drawHeight - this.imgRect.h) / 2;

								const clipX = this.imgRect.w * this.sliderPos;

								// Draw B
								ctx.drawImage(this.imgB_obj, this.imgRect.x, this.imgRect.y, this.imgRect.w, this.imgRect.h);
								
								// Draw A (Clipped)
								ctx.save();
								ctx.beginPath();
								ctx.rect(this.imgRect.x, this.imgRect.y, clipX, this.imgRect.h);
								ctx.clip();
								ctx.drawImage(this.imgA_obj, this.imgRect.x, this.imgRect.y, this.imgRect.w, this.imgRect.h);
								ctx.restore();

								// Slider UI
								ctx.strokeStyle = "#4CAF50";
								ctx.lineWidth = 2;
								ctx.beginPath();
								ctx.moveTo(this.imgRect.x + clipX, this.imgRect.y);
								ctx.lineTo(this.imgRect.x + clipX, this.imgRect.y + this.imgRect.h);
								ctx.stroke();

                                // Label
                                const label = this.sliderPos > 0.5 ? "A (Left)" : "B (Right)";
                                ctx.fillStyle = "rgba(0,0,0,0.7)";
                                ctx.fillRect(this.imgRect.x + clipX - 35, this.imgRect.y + 5, 70, 18);
                                ctx.fillStyle = "#FFF";
                                ctx.font = "bold 10px Arial";
                                ctx.textAlign = "center";
                                ctx.fillText(label, this.imgRect.x + clipX, this.imgRect.y + 17);
							}
						},
						mouse(event, pos, node) {
							if (event.type === "mousedown" || (event.type === "mousemove" && event.buttons & 1)) {
                                // Real-time calculation based on image bounds
								this.sliderPos = Math.max(0, Math.min(1, (pos[0] - this.imgRect.x) / this.imgRect.w));
								node.setDirtyCanvas(true);
								return true;
							}
						},
                        computeSize(width) {
                            return [width, width * (384/672) + 60]; // Dynamic height based on standard ratio
                        }
					};

					this.addCustomWidget(widget);
					this.setDirtyCanvas(true);
				}
			};
		}
	},
});
