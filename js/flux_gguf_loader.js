import { app } from "../../../scripts/app.js";

/**
 * Extension to dynamically hide/show widgets in the FluxGGUFLoader node.
 * Specifically hides clip_name2 when clip_type is set to 'flux2'.
 */
app.registerExtension({
    name: "FluxCollection.FluxGGUFLoader.Logic",
    async nodeCreated(node) {
        if (node.comfyClass === "FluxGGUFLoader") {
            // Find the widgets we want to work with
            const clipTypeWidget = node.widgets.find(w => w.name === "clip_type");
            const clip2Widget = node.widgets.find(w => w.name === "clip_name2");

            if (clipTypeWidget && clip2Widget) {
                // Function to update visibility based on current value
                const updateVisibility = () => {
                    const isFlux2 = clipTypeWidget.value === "flux2";
                    
                    if (isFlux2) {
                        // Hide widget and set value to None for the backend
                        clip2Widget.type = "hidden";
                        clip2Widget.value = "None";
                        if (clip2Widget.linked_widgets) clip2Widget.linked_widgets.forEach(w => w.type = "hidden");
                    } else {
                        // Show widget
                        clip2Widget.type = "combo";
                        if (clip2Widget.linked_widgets) clip2Widget.linked_widgets.forEach(w => w.type = "combo");
                    }
                    
                    // Trigger node resize to fit changes
                    node.setSize(node.computeSize());
                    app.canvas.setDirty(true);
                };

                // Listen for changes
                const originalCallback = clipTypeWidget.callback;
                clipTypeWidget.callback = function() {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    updateVisibility();
                };

                // Initial check
                setTimeout(updateVisibility, 10);
            }
        }
    }
});
