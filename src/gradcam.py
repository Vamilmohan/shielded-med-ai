import torch
import numpy as np
import cv2

def get_last_conv_layer(model):
    for layer in reversed(list(model.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    raise RuntimeError("No Conv2D layer found")

def generate_heatmap(model, image_tensor):
    model.eval()

    features = []
    gradients = []

    target_layer = get_last_conv_layer(model)

    def forward_hook(module, inp, out):
        features.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    class_idx = output.argmax(dim=1)

    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0].mean(dim=(2, 3))[0]
    fmap = features[0][0]

    cam = torch.zeros(fmap.shape[1:], dtype=torch.float32)

    for i, w in enumerate(grads):
        cam += w * fmap[i]

    cam = cam.detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / (cam.max() + 1e-8)

    return cam