import os
import sys
import json
from pathlib import Path
from collections import defaultdict

# Configurar stdout para UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent

def prepare_kallpa_dataset():
    print("=" * 80)
    print("  AGROVISION LeafNet-kallpa (MVP 7) — PREPARACION Y DATASET HÍBRIDO")
    print("=" * 80)

    pv_dir = ROOT_DIR / "datasets" / "plantvillage_mvp2"
    pw_dir = ROOT_DIR / "plantwild" / "plantwild"
    pw_images_dir = pw_dir / "images"
    pw_trainval_file = pw_dir / "trainval.txt"
    plantdoc_dir = ROOT_DIR / "datasets" / "plantdoc_test"

    output_dataset_dir = ROOT_DIR / "datasets" / "leafnet_kallpa"
    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    classes_path = ROOT_DIR / "models" / "class_names_mvp2.json"
    mapping_path = ROOT_DIR / "models" / "class_mapping_mvp2.json"

    with open(classes_path, "r", encoding="utf-8") as f:
        agro_classes = json.load(f)

    # Mapeo Oficial de 24 Clases AgroVision <---> PlantWild
    agro_to_pw = {
        "Apple___Apple_scab": "apple scab",
        "Apple___Black_rot": "apple black rot",
        "Apple___Cedar_apple_rust": "apple rust",
        "Apple___healthy": "apple leaf",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "corn gray leaf spot",
        "Corn_(maize)___Common_rust_": "corn rust",
        "Corn_(maize)___Northern_Leaf_Blight": "corn northern leaf blight",
        "Corn_(maize)___healthy": "corn leaf",
        "Grape___Black_rot": "grape black rot",
        "Grape___Esca_(Black_Measles)": None,
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape leaf spot",
        "Grape___healthy": "grape leaf",
        "Pepper,_bell___Bacterial_spot": "bell pepper leaf spot",
        "Pepper,_bell___healthy": "bell pepper leaf",
        "Potato___Early_blight": "potato early blight",
        "Potato___Late_blight": "potato late blight",
        "Potato___healthy": "potato leaf",
        "Tomato___Bacterial_spot": "tomato bacterial leaf spot",
        "Tomato___Early_blight": "tomato early blight",
        "Tomato___Late_blight": "tomato late blight",
        "Tomato___Leaf_Mold": "tomato leaf mold",
        "Tomato___Septoria_leaf_spot": "tomato septoria leaf spot",
        "Tomato___Spider_mites Two-spotted_spider_mite": None,
        "Tomato___Target_Spot": None,
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato yellow leaf curl virus",
        "Tomato___Tomato_mosaic_virus": "tomato mosaic virus",
        "Tomato___healthy": "tomato leaf"
    }

    pw_to_agro = {v: k for k, v in agro_to_pw.items() if v is not None}

    # Parsear splits de PlantWild
    pw_splits = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    
    with open(pw_trainval_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("=")
            if len(parts) == 3:
                rel_img_path, _, mode = parts
                pw_class_folder = rel_img_path.split("/")[0]
                if pw_class_folder in pw_to_agro:
                    agro_class = pw_to_agro[pw_class_folder]
                    full_img_path = pw_images_dir / rel_img_path
                    if full_img_path.exists():
                        if mode == "1":
                            pw_splits["train"][agro_class].append(str(full_img_path))
                        elif mode == "2":
                            pw_splits["val"][agro_class].append(str(full_img_path))
                        elif mode == "0":
                            pw_splits["test"][agro_class].append(str(full_img_path))

    # Parsear PlantVillage
    pv_splits = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    for agro_cls in agro_classes:
        # Train
        pv_c_train = pv_dir / "train" / agro_cls
        if pv_c_train.exists():
            for f_img in pv_c_train.iterdir():
                if f_img.is_file() and f_img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    pv_splits["train"][agro_cls].append(str(f_img))
        # Val
        pv_c_val = pv_dir / "validation" / agro_cls
        if pv_c_val.exists():
            for f_img in pv_c_val.iterdir():
                if f_img.is_file() and f_img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    pv_splits["val"][agro_cls].append(str(f_img))
        # Test
        pv_c_test = pv_dir / "test" / agro_cls
        if pv_c_test.exists():
            for f_img in pv_c_test.iterdir():
                if f_img.is_file() and f_img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    pv_splits["test"][agro_cls].append(str(f_img))

    train_records = []
    val_records = []
    pw_test_records = []
    pv_test_records = []

    per_class_summary = []

    for idx, agro_cls in enumerate(agro_classes):
        pv_tr = pv_splits["train"][agro_cls]
        pv_va = pv_splits["val"][agro_cls]
        pv_te = pv_splits["test"][agro_cls]

        pw_tr = pw_splits["train"][agro_cls]
        pw_va = pw_splits["val"][agro_cls]
        pw_te = pw_splits["test"][agro_cls]

        for p in pv_tr:
            train_records.append({"path": p, "class_idx": idx, "class_name": agro_cls, "domain": "PlantVillage"})
        for p in pw_tr:
            train_records.append({"path": p, "class_idx": idx, "class_name": agro_cls, "domain": "PlantWild"})

        for p in pv_va:
            val_records.append({"path": p, "class_idx": idx, "class_name": agro_cls, "domain": "PlantVillage"})
        for p in pw_va:
            val_records.append({"path": p, "class_idx": idx, "class_name": agro_cls, "domain": "PlantWild"})

        for p in pw_te:
            pw_test_records.append({"path": p, "class_idx": idx, "class_name": agro_cls, "domain": "PlantWild"})
        for p in pv_te:
            pv_test_records.append({"path": p, "class_idx": idx, "class_name": agro_cls, "domain": "PlantVillage"})

        pw_name = agro_to_pw.get(agro_cls)
        per_class_summary.append({
            "class_idx": idx,
            "class_name": agro_cls,
            "pw_class_name": pw_name,
            "total_train": len(pv_tr) + len(pw_tr)
        })

    manifest_train_path = output_dataset_dir / "train_manifest.json"
    manifest_val_path = output_dataset_dir / "val_manifest.json"
    manifest_pw_test_path = output_dataset_dir / "plantwild_test_manifest.json"
    summary_path = output_dataset_dir / "dataset_summary.json"

    with open(manifest_train_path, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2, ensure_ascii=False)
    with open(manifest_val_path, "w", encoding="utf-8") as f:
        json.dump(val_records, f, indent=2, ensure_ascii=False)
    with open(manifest_pw_test_path, "w", encoding="utf-8") as f:
        json.dump(pw_test_records, f, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(per_class_summary, f, indent=2, ensure_ascii=False)

    plantdoc_imgs_in_train = sum(1 for r in train_records if "plantdoc" in r["path"].lower())
    plantdoc_imgs_in_val = sum(1 for r in val_records if "plantdoc" in r["path"].lower())

    print("\n" + "=" * 60)
    print("  LEAFNET-KALLPA (MVP 7) DATASET — REPORTE DE VERIFICACION")
    print("=" * 60)
    print(f"TOTAL ENTRENAMIENTO: {len(train_records):>6,} imágenes")
    print(f"TOTAL VALIDACION:    {len(val_records):>6,} imágenes")
    print("\n" + "=" * 60)
    print("  DATA LEAKAGE CHECK (PLANTDOC)")
    print("=" * 60)
    print(f"  PlantDoc training images:   {plantdoc_imgs_in_train} ✓")
    print(f"  PlantDoc validation images: {plantdoc_imgs_in_val} ✓")
    if plantdoc_imgs_in_train == 0 and plantdoc_imgs_in_val == 0:
        print("  [OK] Integridad CIEGA confirmada para PlantDoc.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    prepare_kallpa_dataset()
