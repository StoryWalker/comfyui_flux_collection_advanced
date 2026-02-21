import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/**
 * Flux Image Comparison (Basado en arquitectura estable):
 * Maneja a_images y b_images para crear una comparativa interactiva.
 */

app.registerExtension({
	name: "flux_collection_advanced.FluxImageComparison",
	async beforeRegisterNodeDef(nodeDef, nodeData, app) {
		if (nodeData.name === "FluxImageComparison") {
			
			const onExecuted = nodeDef.prototype.onExecuted;
			nodeDef.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

				if (message.a_images && message.b_images && message.a_images.length > 0 && message.b_images.length > 0) {
					const imgA = message.a_images[0];
					const imgB = message.b_images[0];

					const urlA = api.apiURL(`/view?filename=${encodeURIComponent(imgA.filename)}&type=${imgA.type}&subfolder=${encodeURIComponent(imgA.subfolder)}`);
					const urlB = api.apiURL(`/view?filename=${encodeURIComponent(imgB.filename)}&type=${imgB.type}&subfolder=${encodeURIComponent(imgB.subfolder)}`);

					// Buscar si ya existe el widget para no duplicarlo
					let widget = this.widgets?.find(w => w.name === "cmp_widget");
					
					if (!widget) {
						widget = {
							type: "custom_preview",
							name: "cmp_widget",
							sliderPos: 0.5,
							imgA: null,
							imgB: null,
							draw(ctx, node, widget_width, y, widget_height) {
								const margin = 5;
								const top_y = y + margin;
								const w = widget_width - margin * 2;
								// Altura dinámica basada en el nodo para evitar "franjas"
								const h = node.size[1] - y - margin * 2;

								if (h < 20) return;

								// Fondo negro
								ctx.fillStyle = "#000";
								ctx.fillRect(margin, top_y, w, h);

								if (this.imgA?.complete && this.imgB?.complete) {
									// Calcular proporciones (Contain)
									const imgW = this.imgA.naturalWidth;
									const imgH = this.imgA.naturalHeight;
									const ratio = Math.min(w / imgW, h / imgH);
									const dw = imgW * ratio;
									const dh = imgH * ratio;
									const dx = margin + (w - dw) / 2;
									const dy = top_y + (h - dh) / 2;

									this.rect = { x: dx, y: dy, w: dw, h: dh };

									const clipX = dw * this.sliderPos;

									// Dibujar imagen B (Fondo)
									ctx.drawImage(this.imgB, dx, dy, dw, dh);

									// Dibujar imagen A (Corte)
									ctx.save();
									ctx.beginPath();
									ctx.rect(dx, dy, clipX, dh);
									ctx.clip();
									ctx.drawImage(this.imgA, dx, dy, dw, dh);
									ctx.restore();

									// Slider
									ctx.strokeStyle = "#00FF00";
									ctx.lineWidth = 2;
									ctx.beginPath();
									ctx.moveTo(dx + clipX, dy);
									ctx.lineTo(dx + clipX, dy + dh);
									ctx.stroke();

									// Tooltip
									const label = this.sliderPos > 0.5 ? "A" : "B";
									ctx.fillStyle = "rgba(0,0,0,0.5)";
									ctx.fillRect(dx + clipX - 15, dy + 5, 30, 15);
									ctx.fillStyle = "#FFF";
									ctx.font = "10px Arial";
									ctx.textAlign = "center";
									ctx.fillText(label, dx + clipX, dy + 16);
								}
							},
							mouse(event, pos, node) {
								if (this.rect && (event.type === "mousedown" || (event.type === "mousemove" && event.buttons & 1))) {
									this.sliderPos = Math.max(0, Math.min(1, (pos[0] - this.rect.x) / this.rect.w));
									node.setDirtyCanvas(true);
									return true;
								}
							},
							computeSize(width) {
								return [width, 400]; // Asegura altura mínima
							}
						};
						this.addCustomWidget(widget);
					}

					// Cargar imágenes
					widget.imgA = new Image();
					widget.imgA.onload = () => this.setDirtyCanvas(true);
					widget.imgA.src = urlA;
					widget.imgB = new Image();
					widget.imgB.onload = () => this.setDirtyCanvas(true);
					widget.imgB.src = urlB;

					this.setDirtyCanvas(true);
				}
			};
		}
	},
});
