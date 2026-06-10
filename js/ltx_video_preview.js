/**
 * @fileoverview
 * Este script añade una previsualización de video interactiva al nodo LTXVideoSaver.
 * Crea un elemento de video HTML5 real y lo posiciona sobre el nodo en el lienzo de ComfyUI,
 * reproduciendo el video final una vez que el pipeline ha terminado.
 */

console.info("[LTXVideoPreview Script] Iniciando ejecución.");

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// Asegurar que LiteGraph esté disponible
const LGraph = window.LiteGraph;
if (!LGraph) {
    console.error("[LTXVideoPreview Script] LiteGraph no encontrado en el objeto window.");
}

/**
 * Crea y gestiona el widget de previsualización de video.
 */
function createVideoPreviewWidget(node, widgetName = "videoPreviewWidget") {
    const widget = {
        name: widgetName,
        type: "VIDEO_PREVIEW",
        value: null, // Almacena la URL del video actual
        options: {},
        _videoElement: null,

        /**
         * LiteGraph llama a este método para dibujar el widget.
         * Lo usamos para posicionar y redimensionar el elemento <video> real de HTML.
         */
        draw: function(ctx, node, widget_width, widget_y, widget_height) {
            if (!this._videoElement) return;

            this._videoElement.hidden = !this.value;

            if (this.value) {
                // Obtener transformaciones del canvas (escala y desplazamiento)
                const scale = app.canvas.ds.scale;
                const offset = app.canvas.ds.offset;

                // Calcular posición del nodo y del widget en coordenadas globales del canvas
                const canvasNodePos = node.localToGlobal([0, 0]);
                const canvasWidgetPos = node.localToGlobal([0, widget_y]);

                // Traducir coordenadas de canvas a coordenadas de pantalla (píxeles reales del navegador)
                const screenWidgetX = canvasNodePos[0] * scale + offset[0];
                const screenWidgetY = canvasWidgetPos[1] * scale + offset[1];
                const screenWidgetWidth = widget_width * scale;
                const screenWidgetHeight = widget_height * scale;

                // Aplicar posición y tamaño al elemento <video>
                this._videoElement.style.left = `${screenWidgetX}px`;
                this._videoElement.style.top = `${screenWidgetY}px`;
                this._videoElement.style.width = `${screenWidgetWidth}px`;
                this._videoElement.style.height = `${screenWidgetHeight}px`;
            }
        },

        /**
         * Determina el tamaño que debe ocupar el widget en el nodo.
         */
        computeSize: function(width) {
            // Si hay un video cargado, le asignamos una altura por defecto (256px) en el nodo
            return [width, this.value ? (this.options?.height || 256) : 0];
        },

        /**
         * Asigna la URL del video al reproductor.
         */
        setVideoUrl: function(url) {
            if (this.value === url) return;
            this.value = url;

            if (this._videoElement) {
                if (url) {
                    this._videoElement.src = url;
                    this._videoElement.hidden = false;
                    this._videoElement.play().catch(e => {
                        console.log("[LTXVideoPreview] Reproducción automática bloqueada o fallida, reintentando silenciado:", e);
                        this._videoElement.muted = true;
                        this._videoElement.play().catch(err => console.error("[LTXVideoPreview] Error crítico al reproducir video:", err));
                    });
                } else {
                    this._videoElement.pause();
                    this._videoElement.src = "";
                    this._videoElement.hidden = true;
                }
            }
            app.graph.setDirtyCanvas(true, true);
        },

        /**
         * Limpieza cuando el widget es eliminado.
         */
        onRemoved: function() {
            if (this._videoElement) {
                this._videoElement.pause();
                this._videoElement.remove();
                this._videoElement = null;
                console.log("[LTXVideoPreview] Elemento de video eliminado del DOM.");
            }
        }
    };

    // Crear el elemento HTML <video> interactivo
    const videoElement = document.createElement("video");
    videoElement.setAttribute("draggable", "false");
    videoElement.controls = true;
    videoElement.loop = true;
    videoElement.autoplay = true;
    videoElement.muted = true; // Silenciado para evitar el bloqueo de autoplay del navegador
    videoElement.playsInline = true;

    // Estilo básico para posicionarse por encima del canvas de LiteGraph
    Object.assign(videoElement.style, {
        position: "absolute",
        objectFit: "contain",
        backgroundColor: "#111",
        border: "1px solid #ff9000",
        borderRadius: "4px",
        boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
        zIndex: "10",
        pointerEvents: "auto", // Habilitar clics en controles de reproducción
    });
    videoElement.hidden = true;

    // Añadir al DOM
    const parentElement = document.body || app.canvas?.parentNode || document.body;
    parentElement.appendChild(videoElement);
    console.log("[LTXVideoPreview] Elemento <video> creado y adjuntado al DOM.");

    widget._videoElement = videoElement;
    return widget;
}

// Registrar la extensión en ComfyUI
app.registerExtension({
    name: "FluxCollection.LTXVideoSaver.PreviewLogic",

    async nodeCreated(node) {
        const targetNodeType = "LTXVideoSaver";

        if (node.comfyClass === targetNodeType || node.type === targetNodeType) {
            console.info(`[LTXVideoPreview] Nodo LTXVideoSaver detectado (ID: ${node.id}). Agregando previsualización de video...`);

            try {
                // Crear e inyectar el widget de video
                const widget = createVideoPreviewWidget(node, "videoPreviewWidget");
                node.addCustomWidget(widget);
                node.videoPreviewWidget = widget;

                // Dimensiones por defecto razonables para el nodo con previsualizador
                node.size[0] = Math.max(node.size[0], 340);
                node.size[1] = Math.max(node.size[1], 380);
            } catch (e) {
                console.error("[LTXVideoPreview] Error al inicializar el widget de video en el nodo:", e);
                return;
            }

            // Sobrescribir onExecuted del backend
            const original_onExecuted = node.onExecuted;
            node.onExecuted = function(message) {
                console.log(`[LTXVideoPreview] onExecuted para nodo ${this.id}. Mensaje recibido:`, message);

                // Limpiar previsualización previa
                if (this.videoPreviewWidget) {
                    this.videoPreviewWidget.setVideoUrl(null);
                }

                let uiData = null;
                // ComfyUI puede devolver ui data en message o en message[1] dependiendo del tipo de nodo
                if (message?.ui?.gifs) {
                    uiData = message.ui;
                } else if (Array.isArray(message) && message.length === 2 && message[1]?.ui?.gifs) {
                    uiData = message[1].ui;
                } else if (message?.gifs) {
                    uiData = message;
                }

                if (uiData?.gifs && uiData.gifs.length > 0) {
                    const videoInfo = uiData.gifs[0];
                    if (videoInfo.filename) {
                        // Construir la URL de visualización a través de la API oficial de ComfyUI
                        const videoUrl = api.apiURL(`/view?filename=${encodeURIComponent(videoInfo.filename)}&type=${videoInfo.type || 'output'}&subfolder=${encodeURIComponent(videoInfo.subfolder || '')}&t=${+new Date()}`);
                        console.log(`[LTXVideoPreview] Cargando URL de video final:`, videoUrl);

                        if (this.videoPreviewWidget) {
                            this.videoPreviewWidget.setVideoUrl(videoUrl);
                        }
                    }
                }

                try {
                    original_onExecuted?.apply(this, arguments);
                } catch (e) {
                    console.error("[LTXVideoPreview] Error al llamar onExecuted original:", e);
                }
            };

            // Sobrescribir onRemoved para limpiar el DOM
            const original_onRemoved = node.onRemoved;
            node.onRemoved = function() {
                console.log(`[LTXVideoPreview] Eliminando nodo ${this.id}, limpiando recursos.`);
                if (this.videoPreviewWidget?.onRemoved) {
                    this.videoPreviewWidget.onRemoved();
                }
                this.videoPreviewWidget = null;
                original_onRemoved?.apply(this, arguments);
            };
        }
    }
});
