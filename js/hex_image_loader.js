import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "Antigravity.ImageLoaderHex",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "ImageLoaderHex") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				const node = this;
				const widget = node.widgets.find((w) => w.name === "image");

				widget.callback = function (value) {
					if (value && !value.startsWith("[")) {
						node.imgs = [new Image()];
						node.imgs[0].src = `/view?filename=${encodeURIComponent(value)}&type=input&subfolder=`;
					} else {
						node.imgs = null;
					}
					node.setDirtyCanvas(true, true);
				};

				return r;
			};
		}
	},
});
