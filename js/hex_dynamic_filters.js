import { app } from "../../../scripts/app.js";

/**
 * Extension to dynamically filter and sync widgets based on base_type selection.
 * Handles FluxGGUFLoader and NVFP4NativeLoaderHex.
 */
app.registerExtension({
    name: "Hex.DynamicFilters.Logic",
    async nodeCreated(node) {
        if (node.comfyClass === "FluxGGUFLoader" || node.comfyClass === "NVFP4NativeLoaderHex") {
            const baseTypeWidget = node.widgets.find(w => w.name === "base_type");
            const clipTypeWidget = node.widgets.find(w => w.name === "clip_type");
            const clip1Widget = node.widgets.find(w => w.name === "clip_name1");
            const clip2Widget = node.widgets.find(w => w.name === "clip_name2");

            if (baseTypeWidget && clipTypeWidget && clip1Widget) {
                
                // Guardar la lista original de opciones que envió el backend
                if (!node.originalClip1Options) {
                    node.originalClip1Options = [...(clip1Widget.options.values || [])];
                }
                
                const updateLogic = () => {
                    const isFlux2 = baseTypeWidget.value === "flux2";
                    
                    // 1. Sincronizar clip_type
                    if (isFlux2) {
                        clipTypeWidget.value = "flux2";
                    } else if (baseTypeWidget.value === "flux") {
                        clipTypeWidget.value = "flux";
                    }
                    
                    // 2. Filtrar clip_name1
                    if (isFlux2) {
                        // Solo dejar mistral
                        const filtered1 = node.originalClip1Options.filter(name => 
                            name.toLowerCase().includes("mistral")
                        );
                        clip1Widget.options.values = filtered1;
                        if (!filtered1.includes(clip1Widget.value) && filtered1.length > 0) {
                            clip1Widget.value = filtered1[0];
                        }
                    } else {
                        // Solo dejar clip/l14
                        const filtered1 = node.originalClip1Options.filter(name => 
                            name.toLowerCase().includes("clip") || name.toLowerCase().includes("l14")
                        );
                        clip1Widget.options.values = filtered1;
                        if (!filtered1.includes(clip1Widget.value) && filtered1.length > 0) {
                            clip1Widget.value = filtered1[0];
                        }
                    }

                    // 3. Ocultar/Mostrar clip_name2
                    if (clip2Widget) {
                        if (isFlux2) {
                            clip2Widget.type = "hidden";
                            clip2Widget.value = "None";
                            if (clip2Widget.linked_widgets) clip2Widget.linked_widgets.forEach(w => w.type = "hidden");
                        } else {
                            clip2Widget.type = "combo";
                            if (clip2Widget.linked_widgets) clip2Widget.linked_widgets.forEach(w => w.type = "combo");
                        }
                    }
                    
                    // Asegurar que la UI se redibuje
                    node.setSize(node.computeSize());
                    app.canvas.setDirty(true);
                };

                // Escuchar cambios en base_type
                const originalBaseCallback = baseTypeWidget.callback;
                baseTypeWidget.callback = function() {
                    if (originalBaseCallback) originalBaseCallback.apply(this, arguments);
                    updateLogic();
                };

                // Auto-detectar base_type basado en unet_name
                const unetWidget = node.widgets.find(w => w.name === "unet_name");
                if (unetWidget) {
                    const originalUnetCallback = unetWidget.callback;
                    unetWidget.callback = function() {
                        if (originalUnetCallback) originalUnetCallback.apply(this, arguments);
                        
                        const val = (unetWidget.value || "").toLowerCase();
                        if (val.includes("flux2") || val.includes("flux-2")) {
                            if (baseTypeWidget.value !== "flux2") {
                                baseTypeWidget.value = "flux2";
                                updateLogic();
                            }
                        } else if (val.includes("flux1") || val.includes("flux-1")) {
                            if (baseTypeWidget.value !== "flux") {
                                baseTypeWidget.value = "flux";
                                updateLogic();
                            }
                        }
                    };
                }

                // Ejecución inicial
                setTimeout(() => {
                    // Si el unet_name ya está poblado al crear el nodo, intentar inferir
                    if (unetWidget && unetWidget.value) {
                        const val = (unetWidget.value || "").toLowerCase();
                        if ((val.includes("flux2") || val.includes("flux-2")) && baseTypeWidget.value !== "flux2") {
                            baseTypeWidget.value = "flux2";
                        } else if ((val.includes("flux1") || val.includes("flux-1")) && baseTypeWidget.value !== "flux") {
                            baseTypeWidget.value = "flux";
                        }
                    }
                    updateLogic();
                }, 10);
            }
        }
    }
});
