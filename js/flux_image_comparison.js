import { app } from "../../../scripts/app.js";

/**
 * Flux Image Comparison:
 * This JS extension handles the visual rendering of the 'Flux Image Comparison' node.
 * It creates a custom div with a slider that reveals Image A (left) vs Image B (right).
 */

app.registerExtension({
	name: "flux_collection_advanced.FluxImageComparison",
	async beforeRegisterNodeDef(nodeDef, nodeData, app) {
		if (nodeData.name === "FluxImageComparison") {
			const onExecuted = nodeDef.prototype.onExecuted;
			nodeDef.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

				// 1. Clean previous preview if exists
				if (this.widgets) {
					for (let i = 0; i < this.widgets.length; i++) {
						if (this.widgets[i].name === "compare_preview") {
							this.widgets.splice(i, 1);
							break;
						}
					}
				}

				if (message.images && message.images.length >= 2) {
					const imgA = message.images[0];
					const imgB = message.images[1];

					// URL formation
					const urlA = `./view?filename=${imgA.filename}&type=${imgA.type}&subfolder=${imgA.subfolder}`;
					const urlB = `./view?filename=${imgB.filename}&type=${imgB.type}&subfolder=${imgB.subfolder}`;

					// 2. Create Custom Widget
					const widget = {
						type: "custom_preview",
						name: "compare_preview",
						draw(ctx, node, widget_width, y, widget_height) {
							const margin = 10;
							const drawWidth = widget_width - margin * 2;
							const drawHeight = widget_height - margin * 2;

							if (!this.imgA_obj || this.imgA_obj._src !== urlA) {
								this.imgA_obj = new Image();
								this.imgA_obj._src = urlA;
								this.imgA_obj.src = urlA;
							}

							if (!this.imgB_obj || this.imgB_obj._src !== urlB) {
								this.imgB_obj = new Image();
								this.imgB_obj._src = urlB;
								this.imgB_obj.src = urlB;
							}

							if (!this.sliderPos) this.sliderPos = 0.5;

							// Draw Images
							if (this.imgA_obj.complete && this.imgB_obj.complete) {
								const clipX = drawWidth * this.sliderPos;

								// Draw Right Image (B) - Background
								ctx.drawImage(this.imgB_obj, margin, y + margin, drawWidth, drawHeight);
								
								// Draw Left Image (A) - Clipped
								ctx.save();
								ctx.beginPath();
								ctx.rect(margin, y + margin, clipX, drawHeight);
								ctx.clip();
								ctx.drawImage(this.imgA_obj, margin, y + margin, drawWidth, drawHeight);
								ctx.restore();

								// Draw Slider Line
								ctx.strokeStyle = "#FFFFFF";
								ctx.lineWidth = 2;
								ctx.beginPath();
								ctx.moveTo(margin + clipX, y + margin);
								ctx.lineTo(margin + clipX, y + margin + drawHeight);
								ctx.stroke();

                                // Draw Indicator Tooltip
                                ctx.fillStyle = "rgba(0,0,0,0.6)";
                                ctx.fillRect(margin + clipX - 50, y + margin + 10, 100, 20);
                                ctx.fillStyle = "#FFFFFF";
                                ctx.font = "12px sans-serif";
                                ctx.textAlign = "center";
                                const label = this.sliderPos < 0.5 ? "Right: B" : "Left: A";
                                ctx.fillText(label, margin + clipX, y + margin + 25);
							}
						},
						mouse(event, pos, node) {
							if (event.type === "mousedown" || (event.type === "mousemove" && event.buttons & 1)) {
								const widget_y = node.widgets.indexOf(this) * 20; // Approx
								const relX = pos[0];
                                const margin = 10;
                                const width = node.size[0] - margin * 2;
								this.sliderPos = Math.max(0, Math.min(1, (relX - margin) / width));
								node.setDirtyCanvas(true);
								return true;
							}
						},
                        computeSize(width) {
                            return [width, width]; // Square preview
                        }
					};

					this.addCustomWidget(widget);
					this.setSize(this.computeSize());
					this.setDirtyCanvas(true);
				}
			};
		}
	},
});
