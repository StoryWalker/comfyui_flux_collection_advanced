/**
 * @fileoverview This script provides client-side logic for the custom ComfyUI node 'FluxControlNetApplyPreview'.
 * It adds a custom widget to the node that displays the input hint image preview
 * fetched from the backend after execution. This uses DOM manipulation for the preview element.
 *
 * Target Node Class Name (Python): FluxControlNetApplyPreview
 */

console.info("[FluxControlNetApplyPreview Script] Starting execution.");

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// Ensure LiteGraph is available
const LGraph = window.LiteGraph;
if (!LGraph) {
    console.error("[FluxControlNetApplyPreview Script] LiteGraph not found on window object! Node previews will not function.");
}

/**
 * Creates and manages the Image Preview Widget.
 * This widget doesn't draw on the canvas itself but manages an HTML <img> element.
 */
function createImagePreviewWidget(node, widgetName = "previewWidget") {
    const widget = {
        name: widgetName,
        type: "IMG_PREVIEW", // Custom widget type
        value: null, // Stores the current image URL or null
        options: { }, // Placeholder for potential options
        _imageElement: null, // Stores the reference to the created <img> element

        /**
         * Called by LiteGraph when the node is drawn. We use it mainly to update
         * the position and visibility of our external <img> element.
         * @param {CanvasRenderingContext2D} ctx The canvas rendering context (not used for drawing here).
         * @param {LGraphNode} node The LiteGraph node instance.
         * @param {number} widget_width The width allocated for the widget.
         * @param {number} widget_y The vertical position of the widget on the canvas.
         * @param {number} widget_height The height allocated for the widget.
         */
        draw: function(ctx, node, widget_width, widget_y, widget_height) {
            // Don't draw anything on the canvas directly.
            // Instead, update the position and visibility of the associated HTML element.
            if (!this._imageElement) return; // Element not created yet

            // Update element visibility based on the stored value (URL)
            this._imageElement.hidden = !this.value;

            if (this.value) {
                // Calculate the absolute position on the page for the widget area
                const transform = ctx.getTransform(); // Gets canvas transform (scale, translation)
                // Calculate node's top-left position in canvas coordinates
                const canvasNodePos = node.localToGlobal([0, 0]);
                // Calculate widget's top-left position in canvas coordinates
                const canvasWidgetPos = node.localToGlobal([0, widget_y]); // widget_y is relative to node's top

                // Translate canvas coordinates to screen/offset coordinates
                // Note: This calculation might need refinement depending on ComfyUI's exact canvas setup
                // We use the graph canvas's offset and scale.
                const scale = app.canvas.ds.scale; // ds is the DrawState (usually)
                const screenWidgetX = canvasNodePos[0] * scale + app.canvas.ds.offset[0];
                const screenWidgetY = canvasWidgetPos[1] * scale + app.canvas.ds.offset[1];
                const screenWidgetWidth = widget_width * scale;
                // Use the computed widget height for the image element's max height
                const screenWidgetHeight = widget_height * scale;

                // Apply styles to position the <img> element
                this._imageElement.style.left = `${screenWidgetX}px`;
                this._imageElement.style.top = `${screenWidgetY}px`;
                // Let the image scale within the allocated space, maintaining aspect ratio
                this._imageElement.style.maxWidth = `${screenWidgetWidth}px`;
                this.imgElement.style.maxHeight = `${screenWidgetHeight}px`; // Use max-height
                // console.log(`[FluxControlNetApplyPreview Script] Updating image element position: L=${this._imageElement.style.left}, T=${this._imageElement.style.top}, MaxW=${this._imageElement.style.maxWidth}, MaxH=${this._imageElement.style.maxHeight}`); // Verbose
            }
        },

        /**
         * Computes the size the widget should occupy within the node.
         * @param {number} width The available width for the node.
         * @returns {[number, number]} The computed [width, height] for the widget.
         */
        computeSize: function(width) {
            // If we have an image URL, allocate space; otherwise, take no vertical space.
            const allocatedHeight = this.value ? (this.options?.height || 256) : 0; // Default height or 0
            // Return node width (minus padding perhaps) and calculated height
            // Widget width typically matches node width unless handled differently
            return [width, allocatedHeight];
        },

        /**
         * Custom method to update the image source URL for the widget.
         * @param {string|null} url The new image URL, or null to clear the image.
         */
        setImageUrl: function(url) {
            if (this.value === url) return; // No change needed

            this.value = url; // Store the new URL

            if (this._imageElement) {
                if (url) {
                    this._imageElement.src = url;
                    this._imageElement.hidden = false; // Make visible
                    console.log(`[FluxControlNetApplyPreview Script] Widget updated image src to: ${url}`);
                } else {
                    this._imageElement.src = ""; // Clear src
                    this._imageElement.hidden = true; // Hide element
                    console.log(`[FluxControlNetApplyPreview Script] Widget image cleared.`);
                }
            }
            // Request graph redraw to potentially resize the node based on computeSize changes
            app.graph.setDirtyCanvas(true, true);
        },

        /**
         * Custom method called when the parent node is removed.
         * Ensures the associated HTML element is removed from the DOM.
         */
        onRemoved: function() {
            if (this._imageElement) {
                this._imageElement.remove();
                this._imageElement = null;
                 console.log(`[FluxControlNetApplyPreview Script] Removed preview image element from DOM.`);
            }
        }
    };

    // --- Create and Setup the HTML Image Element ---
    // Create the img element only once and store it
    const imgElement = document.createElement("img");
    imgElement.setAttribute("draggable", "false"); // Prevent dragging the image itself
    // Basic styling - position absolute relative to the graph container/body
    Object.assign(imgElement.style, {
        position: "absolute",
        objectFit: "contain", // Scale image while preserving aspect ratio
        objectPosition: "center center",
        border: "1px solid #333", // Optional border
        zIndex: "5", // Try to keep it above node canvas but below other UI? Adjust as needed
        pointerEvents: "none", // Prevent img from interfering with mouse events on canvas
        width: "auto", // Let max-width/max-height control size
        height: "auto",
        maxWidth: "100%", // Default max size (will be updated in draw)
        maxHeight: "100%",
        display: "block", // Use block display
        visibility: "visible", // Control presence with 'hidden' property
    });
    imgElement.hidden = true; // Start hidden
    imgElement.src = ""; // Start empty

    // Append the element to the main graph container or body
    // Appending to document.body is simpler but might have issues with stacking contexts or removal
    // Appending to app.canvas.parentNode might be better but needs confirmation it exists
    const parentElement = document.body || app.canvas?.parentNode || document.body; // Fallback strategy
    parentElement.appendChild(imgElement);
    console.log(`[FluxControlNetApplyPreview Script] Created and appended hidden <img> element for preview.`);

    widget._imageElement = imgElement; // Store reference in the widget object

    return widget;
}


// --- ComfyUI Extension Registration ---
app.registerExtension({
    name: "FluxCollection.FluxControlNetApplyPreview.WidgetLogic", // Specific name

    /**
     * Called when a node is created on the graph.
     * Identifies the target node and adds the custom image preview widget.
     * @param {LGraphNode} node The created LiteGraph node.
     */
    async nodeCreated(node) {
        // console.log(`[FluxControlNetApplyPreview Script] nodeCreated checking: ${node?.comfyClass}`); // Verbose

        const targetNodeType = "FluxControlNetApplyPreview";
        let isTargetNode = false;

        if (node.comfyClass === targetNodeType || node.type === targetNodeType) {
            isTargetNode = true;
            console.info(`[FluxControlNetApplyPreview Script] MATCH! Found target node (${targetNodeType}) ID ${node.id}. Adding preview widget...`);
        }

        if (isTargetNode) {
            // --- Add the Custom Image Preview Widget ---
            try {
                 const widget = createImagePreviewWidget(node, "previewWidget"); // Create our widget instance
                 node.addCustomWidget(widget); // Add it to the node's widget list
                 node.previewWidget = widget; // Store a reference on the node for easy access later
                 console.log(`[FluxControlNetApplyPreview Script] Added custom preview widget to node ${node.id}.`);

                 // We no longer need to override onDrawForeground for drawing the image
                 // The widget's draw method handles positioning the DOM element

            } catch (e) {
                console.error(`[FluxControlNetApplyPreview Script] Failed to create or add custom widget for node ${node.id}:`, e);
                return; // Stop initialization if widget fails
            }


            // --- Override onExecuted ---
            // Handles messages from the backend after execution.
            const original_onExecuted = node.onExecuted;
            node.onExecuted = function(message) {
                console.log(`[FluxControlNetApplyPreview Script] onExecuted called for node ${this.id}. Raw message:`, message);
                // Clear previous preview immediately via the widget method
                if (this.previewWidget) {
                     this.previewWidget.setImageUrl(null);
                } else {
                     console.warn(`[FluxControlNetApplyPreview Script] Node ${this.id} - previewWidget reference missing in onExecuted!`);
                }

                let uiData = null;
                let foundUiData = false;

                // Expect message as Array [output_data, {ui: {images: ...}}]
                if (Array.isArray(message) && message.length === 2 && message[1]?.ui?.images) {
                    const lastElement = message[1];
                     if (lastElement && typeof lastElement === 'object' && lastElement.hasOwnProperty('ui') && Array.isArray(lastElement.ui.images)) {
                        uiData = lastElement.ui;
                        foundUiData = true;
                        console.log(`[FluxControlNetApplyPreview Script] Node ${this.id} - Extracted ui data from message[1].`);
                     } else {
                         console.warn(`[FluxControlNetApplyPreview Script] Node ${this.id} - message[1] lacked expected {ui: {images: [...]}} structure.`);
                     }
                } else {
                    console.warn(`[FluxControlNetApplyPreview Script] Node ${this.id} - Received message is NOT expected Array(2).`);
                }

                // If valid UI data with images was found, update the widget
                if (foundUiData && uiData.images.length > 0) {
                    const imgData = uiData.images[0];
                    console.log(`[FluxControlNetApplyPreview Script] Node ${this.id} - Image data:`, imgData);

                    if (!imgData.filename || !imgData.type) {
                        console.warn(`[FluxControlNetApplyPreview Script] Node ${this.id} - Image data invalid. Clearing widget.`);
                        if (this.previewWidget) this.previewWidget.setImageUrl(null);
                    } else {
                        // Construct URL and update the widget
                        const imageUrl = api.apiURL(`/view?filename=${encodeURIComponent(imgData.filename)}&type=${imgData.type}&subfolder=${encodeURIComponent(imgData.subfolder || '')}&t=${+new Date()}`);
                        console.log(`[FluxControlNetApplyPreview Script] Node ${this.id} - Updating widget with Image URL:`, imageUrl);
                        if (this.previewWidget) {
                            this.previewWidget.setImageUrl(imageUrl); // Update the widget's value (URL)
                        } else {
                             console.warn(`[FluxControlNetApplyPreview Script] Node ${this.id} - previewWidget reference missing when trying to set URL!`);
                        }
                        // Note: We don't need img.onload here as the <img> element handles loading.
                        // Error handling would need to be attached to the img element itself if needed.
                    }
                } else {
                    // No valid image data found, ensure widget is cleared
                    console.log(`[FluxControlNetApplyPreview Script] Node ${this.id} - No valid image data. Clearing widget.`);
                    if (this.previewWidget) this.previewWidget.setImageUrl(null);
                }

                // Call original onExecuted
                try {
                    original_onExecuted?.apply(this, arguments);
                } catch (e) {
                    console.error(`[FluxControlNetApplyPreview Script] Error calling original onExecuted for node ${this.id}:`, e);
                }
            }; // End of onExecuted override


            // --- Override onRemoved ---
            // Clean up the widget and its DOM element when the node is removed
            const original_onRemoved = node.onRemoved;
            node.onRemoved = function() {
                console.log(`[FluxControlNetApplyPreview Script] Node ${this.id} removed. Cleaning up preview widget.`);
                if (this.previewWidget?.onRemoved) {
                    this.previewWidget.onRemoved(); // Call widget's cleanup method
                }
                this.previewWidget = null; // Remove reference from node
                original_onRemoved?.apply(this, arguments); // Call original handler
            };

            console.log(`[FluxControlNetApplyPreview Script] Custom widget logic initialization finished for node ${node.id}.`);
        } // End of if(isTargetNode)
    } // End of nodeCreated function
});

// --- DEBUG LOG: Confirm script execution end ---
console.log("[FluxControlNetApplyPreview Script] script execution finished.");