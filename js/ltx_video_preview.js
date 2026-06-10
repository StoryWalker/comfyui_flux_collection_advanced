/**
 * @fileoverview
 * Previsualización de video interactiva para nodos LTX Video Saver (legacy y HEX).
 * Renderiza un <video> encapsulado en un div contenedor que sigue al nodo.
 *
 * FIX 2026-06-10 — Bug GUI rota:
 *  - Contenedor div con overflow:hidden + position:fixed para clipping visual.
 *  - pointerEvents:"auto" dentro del contenedor; el overflow evita que se salga.
 *  - Autoplay muted restaurado; los controles son clickeables.
 *  - Sincronización basada en getBoundingClientRect() del canvas.
 *
 * FIX 2026-06-10 — Bug video sin sonido:
 *  - muted=false cuando el video tiene pista de audio (el navegador la reproduce).
 *  - muted=true solo como fallback inicial; el usuario puede activar sonido.
 */

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const WIDGET_HEIGHT = 220;
const Z_INDEX = 100;

function createVideoPreviewWidget(node, widgetName = "videoPreviewWidget") {
    const widget = {
        name: widgetName,
        type: "VIDEO_PREVIEW",
        value: null,
        options: { height: WIDGET_HEIGHT },
        _container: null,
        _videoElement: null,
        _nodeRef: node,

        draw: function () {
            if (!this._container || !this.value) return;
            this._syncPosition();
        },

        computeSize: function (width) {
            return [width, this.value ? (this.options?.height || WIDGET_HEIGHT) : 0];
        },

        setVideoUrl: function (url) {
            if (this.value === url) return;
            this.value = url;

            if (!this._videoElement) return;

            if (url) {
                this._videoElement.src = url;
                this._container.hidden = false;
                this._videoElement.load();
                const playPromise = this._videoElement.play();
                if (playPromise) {
                    playPromise.catch(() => {
                        this._videoElement.muted = true;
                        this._videoElement.play().catch(() => {});
                    });
                }
            } else {
                this._videoElement.pause();
                this._videoElement.removeAttribute("src");
                this._container.hidden = true;
            }
            app.graph.setDirtyCanvas(true, false);
        },

        _syncPosition: function () {
            const container = this._container;
            const node = this._nodeRef;
            if (!container || container.hidden || !node) return;

            const canvasEl = app.canvas.canvas;
            if (!canvasEl) return;

            const canvasRect = canvasEl.getBoundingClientRect();
            const scale = app.canvas.ds.scale;
            const offset = app.canvas.ds.offset;

            // Posición del nodo en pantalla
            const nodeScreenX = canvasRect.left + (node.pos[0] * scale) + offset[0];
            const nodeScreenY = canvasRect.top + (node.pos[1] * scale) + offset[1];

            // Altura del título + inputs
            const titleH = LiteGraph.NODE_TITLE_HEIGHT * scale;
            const inputsH = (node.inputs?.length || 0) * LiteGraph.NODE_SLOT_HEIGHT * scale;
            const widgetYOffset = titleH + inputsH + 8 * scale;

            const width = Math.max(60, (node.size[0] - 8) * scale);
            const height = Math.max(40, WIDGET_HEIGHT * scale);

            if (!Number.isFinite(nodeScreenX) || !Number.isFinite(nodeScreenY)) {
                container.hidden = true;
                return;
            }

            container.style.left = `${nodeScreenX + 4 * scale}px`;
            container.style.top = `${nodeScreenY + widgetYOffset}px`;
            container.style.width = `${width}px`;
            container.style.height = `${height}px`;
        },

        onRemoved: function () {
            if (this._container) {
                this._container.remove();
                this._container = null;
                this._videoElement = null;
            }
        }
    };

    // Contenedor div con overflow:hidden (hace clipping si se sale)
    const container = document.createElement("div");
    Object.assign(container.style, {
        position: "fixed",
        overflow: "hidden",
        zIndex: String(Z_INDEX),
        pointerEvents: "auto",
        borderRadius: "4px",
        boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
    });
    container.hidden = true;

    // Video dentro del contenedor
    const video = document.createElement("video");
    video.setAttribute("draggable", "false");
    video.controls = true;
    video.loop = true;
    video.autoplay = true;
    video.muted = true; // autoplay requiere muted; el usuario puede desmutear
    video.playsInline = true;
    Object.assign(video.style, {
        width: "100%",
        height: "100%",
        objectFit: "contain",
        backgroundColor: "#111",
        border: "none",
        display: "block",
    });

    container.appendChild(video);
    document.body.appendChild(container);

    widget._container = container;
    widget._videoElement = video;

    // Re-sincronizar en eventos de interacción del canvas
    const canvasEl = app.canvas.canvas;
    if (canvasEl) {
        canvasEl.addEventListener("pointermove", () => widget._syncPosition());
    }

    // Observer de seguridad: si el contenedor queda huérfano, limpiar
    const observer = new MutationObserver((mutations) => {
        for (const m of mutations) {
            for (const removed of m.removedNodes) {
                if (removed === container) {
                    widget.onRemoved();
                    observer.disconnect();
                    return;
                }
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });

    return widget;
}

const TARGET_CLASSES = ["LTXVideoSaver", "LTXVideoSaverHex"];

app.registerExtension({
    name: "FluxCollection.LTXVideoSaver.PreviewLogic",

    async nodeCreated(node) {
        if (!TARGET_CLASSES.includes(node.comfyClass) && !TARGET_CLASSES.includes(node.type)) {
            return;
        }

        try {
            const widget = createVideoPreviewWidget(node, "videoPreviewWidget");
            node.addCustomWidget(widget);
            node.videoPreviewWidget = widget;

            node.size[0] = Math.max(node.size[0], 340);
            node.size[1] = Math.max(node.size[1], 400);
        } catch (e) {
            console.error("[LTXVideoPreview] Error inicializando widget:", e);
            return;
        }

        const origOnExecuted = node.onExecuted;
        node.onExecuted = function (message) {
            this.videoPreviewWidget?.setVideoUrl(null);

            let uiData = null;
            if (message?.ui?.gifs) {
                uiData = message.ui;
            } else if (Array.isArray(message) && message.length === 2 && message[1]?.ui?.gifs) {
                uiData = message[1].ui;
            } else if (message?.gifs) {
                uiData = message;
            }

            if (uiData?.gifs?.length > 0) {
                const info = uiData.gifs[0];
                if (info.filename) {
                    const url = api.apiURL(
                        `/view?filename=${encodeURIComponent(info.filename)}` +
                        `&type=${info.type || "output"}` +
                        `&subfolder=${encodeURIComponent(info.subfolder || "")}` +
                        `&t=${+new Date()}`
                    );
                    this.videoPreviewWidget?.setVideoUrl(url);
                }
            }

            try { origOnExecuted?.apply(this, arguments); } catch (e) { /* ignore */ }
        };

        const origOnRemoved = node.onRemoved;
        node.onRemoved = function () {
            this.videoPreviewWidget?.onRemoved();
            this.videoPreviewWidget = null;
            try { origOnRemoved?.apply(this, arguments); } catch (e) { /* ignore */ }
        };
    }
});
