import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from transformer_net import TransformerNet
import os
import time

# Cesta k modelum
MODEL_PATHS = {
    '1': ("models/mosaic.pth", "Mosaic"),
    '2': ("models/candy.pth", "Candy"),
    '3': ("models/rain_princess.pth", "Rain Princess"),
    '4': ("models/starry-night.pth", "Starry night"),
    '5': ("models/udnie.pth", "Udnie"),
    '6': ("models/bayanihan.pth", "bayanihan")
}

# Aktuální model
current_model_key = '1'
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print(f"🧠 Výpočty poběží na zařízení: {device}")

# Složka pro uložení snímků
output_dir = "snapshots"
os.makedirs(output_dir, exist_ok=True)

# Zjištění čítače z existujících souborů
existing_files = [f for f in os.listdir(output_dir) if f.startswith("snapshot_") and f.endswith(".jpg")]
existing_numbers = [int(f.split("_")[1]) for f in existing_files if f.split("_")[1].isdigit()]
snapshot_counter = max(existing_numbers, default=0) + 1

# Načtení modelu
def load_model(model_path):
    style_model = TransformerNet()

    # Vždy načti model do CPU
    state_dict = torch.load(model_path, map_location="cpu")

    # Odstranění nepodporovaných klíčů z InstanceNorm2d
    ignored = [k for k in state_dict.keys() if "running_mean" in k or "running_var" in k]
    for k in ignored:
        del state_dict[k]

    # Přejmenování klíčů (pokud existují)
    for k in list(state_dict.keys()):
        if k.startswith("saved_model."):
            state_dict[k.replace("saved_model.", "")] = state_dict.pop(k)

    style_model.load_state_dict(state_dict, strict=False)
    style_model.to(device)
    style_model.eval()
    return style_model

print("\n🎨 Stylizace pomocí neuronových sítí (fast-neural-style)")
print("-------------------------------------------------------")
print("1 = Mosaic")
print("2 = Candy")
print("3 = Rain Princess")
print("4 = Starry night")
print("5 = Udnie")
print("s = Uložit snímek")
print("q = Ukončit\n")

# Inicializace modelu
model, model_name = load_model(MODEL_PATHS[current_model_key][0]), MODEL_PATHS[current_model_key][1]

# Transformace vstupu
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

# Kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(img_tensor).cpu()

    output_data = output_tensor.squeeze().clamp(0, 255).detach().numpy()
    output_data = output_data.transpose(1, 2, 0).astype('uint8')
    output_bgr = cv2.cvtColor(output_data, cv2.COLOR_RGB2BGR)

    # 🧼 Vytvoříme čistou verzi výstupu bez textu pro snapshot
    clean_output = output_bgr.copy()

    # Split-screen: vlevo originál, vpravo stylizace
    original_resized = cv2.resize(frame, (output_bgr.shape[1], output_bgr.shape[0]))
    combined = cv2.hconcat([original_resized, output_bgr])

    # Elegantní horní pruh s názvem stylu
    panel_height = 40
    panel = np.zeros((panel_height, combined.shape[1], 3), dtype=np.uint8)

    fps = 1.0 / (time.time() - start_time)


    cv2.putText(panel, f"Styl: {model_name}   FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(panel, f"Styl: {model_name}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Spojíme panel + video
    output_with_panel = np.vstack([panel, combined])

    fps = 1.0 / (time.time() - start_time)

    cv2.imshow("Stylizace - vlevo originál | vpravo stylizace", output_with_panel)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = os.path.join(output_dir, f"snapshot_{snapshot_counter}_{model_name.lower().replace(' ', '_')}.jpg")
        cv2.imwrite(filename, clean_output)  # Ukládáme čistou verzi
        print(f"🖼️ Uložen snímek jako {filename}")
        snapshot_counter += 1
    elif chr(key) in MODEL_PATHS:
        current_model_key = chr(key)
        model, model_name = load_model(MODEL_PATHS[current_model_key][0]), MODEL_PATHS[current_model_key][1]
        print(f"\n🔁 Přepnuto na styl: {model_name}")

cap.release()
cv2.destroyAllWindows()
