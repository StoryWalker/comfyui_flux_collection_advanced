import { app } from "../../../scripts/app.js";

/**
 * [HEX] Global Seed — JS Extension
 *
 * - control_after_generate (auto ComfyUI): controla el seed de este nodo.
 * - mode: valor que se propaga a todos los FluxSamplerParametersHex al sincronizar.
 * - auto_sync ON: sincroniza automaticamente al ejecutar o al activar el toggle.
 */

const TARGET_NODE = "FluxSamplerParametersHex";
const SEED_WIDGET = "seed";

/**
 * Busca el control_after_generate auto-añadido por ComfyUI en un nodo.
 * Funciona tanto en GlobalSeedHex como en FluxSamplerParametersHex.
 */
function findControlWidget(node, seedWidget) {
	// 1. Por nombre estandar
	const byName = node.widgets?.find(
		w => w.name === "control_after_generate" ||
		     w.name === `${SEED_WIDGET}_control_after_generate`
	);
	if (byName) return byName;

	// 2. Fallback: widget inmediatamente despues del seed con opciones fixed/randomize
	const idx = node.widgets?.indexOf(seedWidget);
	if (idx !== undefined && idx >= 0) {
		const next = node.widgets[idx + 1];
		if (next?.options?.values?.includes?.("fixed") &&
		    next?.options?.values?.includes?.("randomize")) {
			return next;
		}
	}
	return null;
}

/**
 * Propaga seed y control_after_generate a todos los FluxSamplerParametersHex.
 * inheritedControl es el valor del widget "mode" de GlobalSeedHex.
 */
function broadcastSeed(resolvedSeed, inheritedControl) {
	if (resolvedSeed === undefined || resolvedSeed === null) return;

	const targets = app.graph._nodes.filter(n => n.type === TARGET_NODE);
	for (const node of targets) {
		const seedW = node.widgets?.find(w => w.name === SEED_WIDGET);
		if (!seedW) continue;

		seedW.value = resolvedSeed;

		if (inheritedControl) {
			const controlW = findControlWidget(node, seedW);
			if (controlW) controlW.value = inheritedControl;
		}

		node.setDirtyCanvas(true);
	}
}

app.registerExtension({
	name: "flux_collection_advanced.GlobalSeedHex",

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name !== "GlobalSeedHex") return;

		// ── onNodeCreated ────────────────────────────────────────────────
		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
			this.size[0] = 400;

			// Sincronizar inmediatamente cuando auto_sync cambia a ON
			const autoSyncW = this.widgets?.find(w => w.name === "auto_sync");
			if (autoSyncW) {
				const origCallback = autoSyncW.callback;
				autoSyncW.callback = (value, ...args) => {
					if (origCallback) origCallback.call(this, value, ...args);
					if (value === true) {
						const seedW = this.widgets?.find(w => w.name === SEED_WIDGET);
						const modeW = this.widgets?.find(w => w.name === "mode");
						broadcastSeed(seedW?.value, modeW?.value ?? null);
					}
				};
			}

			return r;
		};

		// ── onExecuted: recibe seed resuelto y propaga si auto_sync ON ───
		nodeType.prototype.onExecuted = function (message) {
			const resolved  = message?.seed?.[0];
			const mode      = message?.mode?.[0];
			const autoSync  = message?.auto_sync?.[0] === true;

			if (resolved === undefined) return;

			// Actualizar seed widget local
			const seedWidget = this.widgets?.find(w => w.name === SEED_WIDGET);
			if (seedWidget) {
				seedWidget.value = resolved;
				this.setDirtyCanvas(true);
			}

			if (autoSync) {
				broadcastSeed(resolved, mode);
			}
		};

		// ── onConfigure ──────────────────────────────────────────────────
		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
			this.size[0] = 400;
			return r;
		};

		// ── computeSize ──────────────────────────────────────────────────
		const computeSize = nodeType.prototype.computeSize;
		nodeType.prototype.computeSize = function () {
			const size = computeSize ? computeSize.apply(this, arguments) : [400, 200];
			size[0] = Math.max(size[0], 400);
			return size;
		};
	},
});
