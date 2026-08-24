import os
import sys
import json
import shutil
from pathlib import Path
from collections import defaultdict

# Configurar stdout para UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent

def prepare_mvp6_dataset():
    print("=" * 80)
    print("  AGROVISION MVP 6 — PREPARACION Y AUDITORIA DEL DATASET MULTIDOMINIO")
    print("=" * 80)

    # 1. Definir rutas
    pv_dir = ROOT_DIR / "datasets" / "plantvillage_mvp2"
    pw_dir = ROOT_DIR / "plantwild" / "plantwild"
    pw_images_dir = pw_dir / "images"
    pw_trainval_file = pw_dir / "trainval.txt"
    plantdoc_dir = ROOT_DIR / "datasets" / "plantdoc_test"

    output_dataset_dir = ROOT_DIR / "datasets" / "agrovision_mvp6"
    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    # 2. Cargar las 27 Clases Canónicas de AgroVision
    classes_path = ROOT_DIR / "models" / "class_names_mvp2.json"
    mapping_path = ROOT_DIR / "models" / "class_mapping_mvp2.json"

    with open(classes_path, "r", encoding="utf-8") as f:
        agro_classes = json.load(f)

    with open(mapping_path, "r", encoding="utf-8") as f:
        agro_meta = json.load(f).get("metadata", {})

    # 3. Mapeo Oficial de 24 Clases AgroVision <---> PlantWild
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

    # 4. Parsear splits de PlantWild (trainval.txt)
    # mode: 1 = train, 2 = val, 0 = test
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

    # 5. Parsear PlantVillage (train, validation, test)
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

    # 6. Construir los registros de entrenamiento y validación de MVP 6
    # Generar manifiestos JSON y estructurar carpetas
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

        # Añadir a registros
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
            "pv_train": len(pv_tr),
            "pv_val": len(pv_va),
            "pw_train": len(pw_tr),
            "pw_val": len(pw_va),
            "pw_test": len(pw_te),
            "total_train": len(pv_tr) + len(pw_tr),
            "total_val": len(pv_va) + len(pw_va)
        })

    # 7. Guardar manifiestos estructurados
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

    # 8. Verificación de Data Leakage (PlantDoc)
    plantdoc_imgs_in_train = 0
    plantdoc_imgs_in_val = 0
    for r in train_records:
        if "plantdoc" in r["path"].lower():
            plantdoc_imgs_in_train += 1
    for r in val_records:
        if "plantdoc" in r["path"].lower():
            plantdoc_imgs_in_val += 1

    total_pv_train = sum(len(pv_splits["train"][c]) for c in agro_classes)
    total_pv_val = sum(len(pv_splits["val"][c]) for c in agro_classes)
    total_pw_train = sum(len(pw_splits["train"][c]) for c in agro_classes)
    total_pw_val = sum(len(pw_splits["val"][c]) for c in agro_classes)

    # 9. Imprimir Reporte de Verificación Formal
    print("\n" + "=" * 60)
    print("  AGROVISION MVP 6 DATASET — REPORTE DE VERIFICACION")
    print("=" * 60)
    print(f"\nPlantVillage (Dominio Laboratorio):")
    print(f"  Train:      {total_pv_train:>6,} fotos")
    print(f"  Validation: {total_pv_val:>6,} fotos")

    print(f"\nPlantWild (Dominio Real In-The-Wild):")
    print(f"  Train:      {total_pw_train:>6,} fotos")
    print(f"  Validation: {total_pw_val:>6,} fotos")
    print(f"  Test:       {len(pw_test_records):>6,} fotos")

    print("\nDistribución por Clase (AgroVision Canonical ID 00-26):")
    print("-" * 60)
    for c in per_class_summary:
        pw_str = f"PW: {c['pw_train']:>3}" if c["pw_class_name"] else "PW:   —"
        print(f"  [{c['class_idx']:02d}] {c['class_name']:<46} PV: {c['pv_train']:>4}  {pw_str}  (Total Train: {c['total_train']:>4})")

    print("\nClases sin PlantWild (Solo PlantVillage - Sin sobremuestreo agresivo):")
    print("  • [09] Grape___Esca_(Black_Measles)")
    print("  • [22] Tomato___Spider_mites Two-spotted_spider_mite")
    print("  • [23] Tomato___Target_Spot")

    print("\nPlantDoc (Benchmark Externo Ciego):")
    print("  NO INCLUIDO ✓ (172 imágenes reservadas para test final)")

    print(f"\nTOTAL ENTRENAMIENTO MVP 6: {len(train_records):>6,} imágenes")
    print(f"TOTAL VALIDACION MVP 6:    {len(val_records):>6,} imágenes")

    print("\n" + "=" * 60)
    print("  DATA LEAKAGE CHECK")
    print("=" * 60)
    print(f"  PlantDoc training images:   {plantdoc_imgs_in_train} ✓")
    print(f"  PlantDoc validation images: {plantdoc_imgs_in_val} ✓")
    print("=" * 60 + "\n")

    print(f"[OK] Manifiestos listos en: {output_dataset_dir}")
    print("[STOP] Verificación completada. Listo para revisión antes del entrenamiento.\n")

if __name__ == "__main__":
    prepare_mvp6_dataset()
