import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/**
 * Flux Image Comparison:
 * Optimized JS extension for interactive A/B image comparison with slider.
 */

app.registerExtension({
	name: "flux_collection_advanced.FluxImageComparison",
	async beforeRegisterNodeDef(nodeDef, nodeData, app) {
		if (nodeData.name === "FluxImageComparison") {
			const onExecuted = nodeDef.prototype.onExecuted;
			nodeDef.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

				// 1. Remove existing compare widget if any
				if (this.widgets) {
					this.widgets = this.widgets.filter(w => w.name !== "compare_preview");
				}

				if (message.images && message.images.length >= 2) {
					const imgA = message.images[0];
					const imgB = message.images[1];

					// Correct URL formation using ComfyUI API
					const urlA = api.apiURL(`/view?filename=${encodeURIComponent(imgA.filename)}&type=${imgA.type}&subfolder=${encodeURIComponent(imgA.subfolder)}`);
					const urlB = api.apiURL(`/view?filename=${encodeURIComponent(imgB.filename)}&type=${imgB.type}&subfolder=${encodeURIComponent(imgB.subfolder)}`);

					const widget = {
						type: "custom_preview",
						name: "compare_preview",
						sliderPos: 0.5,
						draw(ctx, node, widget_width, y, widget_height) {
							const margin = 15;
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

							// Draw background placeholder
							ctx.fillStyle = "#1a1a1a";
							ctx.fillRect(margin, y + top_margin, drawWidth, drawHeight);

							if (this.imgA_obj.complete && this.imgB_obj.complete) {
								const clipX = drawWidth * this.sliderPos;

								// Draw Image B (Right/Background)
								ctx.drawImage(this.imgB_obj, margin, y + top_margin, drawWidth, drawHeight);
								
								// Draw Image A (Left/Clipped)
								ctx.save();
								ctx.beginPath();
								ctx.rect(margin, y + top_margin, clipX, drawHeight);
								ctx.clip();
								ctx.drawImage(this.imgA_obj, margin, y + top_margin, drawWidth, drawHeight);
								ctx.restore();

								// Draw Slider Line
								ctx.strokeStyle = "#00FF00";
								ctx.lineWidth = 3;
								ctx.beginPath();
								ctx.moveTo(margin + clipX, y + top_margin);
								ctx.lineTo(margin + clipX, y + top_margin + drawHeight);
								ctx.stroke();

                                // Draw Label Indicator
                                ctx.fillStyle = "rgba(0,0,0,0.7)";
                                ctx.fillRect(margin + clipX - 40, y + top_margin + 5, 80, 20);
                                ctx.fillStyle = "#00FF00";
                                ctx.font = "bold 11px Arial";
                                ctx.textAlign = "center";
                                const label = this.sliderPos > 0.5 ? "A (Left)" : "B (Right)";
                                ctx.fillText(label, margin + clipX, y + top_margin + 19);
							} else {
                                // Loading state
                                ctx.fillStyle = "#AAAAAA";
                                ctx.textAlign = "center";
                                ctx.fillText("Cargando imÃ¡genes...", widget_width / 2, y + (widget_height / 2));
                            }
						},
						mouse(event, pos, node) {
							if (event.type === "mousedown" || (event.type === "mousemove" && event.buttons & 1)) {
								const margin = 15;
								const width = node.size[0] - margin * 2;
								this.sliderPos = Math.max(0, Math.min(1, (pos[0] - margin) / width));
								node.setDirtyCanvas(true);
								return true;
							}
						},
                        computeSize(width) {
                            return [width, width + 40]; 
                        }
					};

					this.addCustomWidget(widget);
					this.onResize?.(this.size);
					this.setDirtyCanvas(true);
				}
			};
		}
	},
});
