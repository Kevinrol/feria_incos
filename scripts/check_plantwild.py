import json
import os
import sys
from pathlib import Path

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def analyze_plantwild():
    # 1. Cargar las clases de PlantVillage (nuestras 27 clases)
    with open('models/class_names_mvp2.json', 'r', encoding='utf-8') as f:
        pv_classes = json.load(f)

    with open('models/class_mapping_mvp2.json', 'r', encoding='utf-8') as f:
        pv_meta = json.load(f)['metadata']

    # 2. Cargar las 89 clases de PlantWild
    pw_json_path = 'MVPDR/gpt_files_plt44/plantwild_prompts_50_25.json'
    with open(pw_json_path, 'r', encoding='utf-8') as f:
        pw_prompts = json.load(f)
    pw_classes = list(pw_prompts.keys())

    target_crops = {
        'Papa': ['potato'],
        'Tomate': ['tomato'],
        'Maíz': ['corn', 'maize'],
        'Manzana': ['apple'],
        'Vid': ['grape', 'grapevine'],
        'Pimiento': ['pepper', 'bell pepper']
    }

    print("=" * 80)
    print("  VERIFICACION DE CULTIVOS EN PLANTWILD (MVPDR)")
    print("=" * 80)

    for crop, kws in target_crops.items():
        found = [c for c in pw_classes if any(k in c.lower() for k in kws)]
        print(f"\n🌿 ¿Tiene {crop}? -> {'SÍ ✅' if found else 'NO ❌'} ({len(found)} clases encontradas en PlantWild):")
        for c in found:
            print(f"    • {c}")

    print("\n" + "=" * 80)
    print("  MAPEO Y COINCIDENCIA CON NUESTRAS 27 CLASES DE AGROVISION")
    print("=" * 80)

    # Mapeo sistemático
    mapping = {
        # Manzana (Apple)
        "Apple___Apple_scab": "apple scab",
        "Apple___Black_rot": "apple black rot",
        "Apple___Cedar_apple_rust": "apple rust",
        "Apple___healthy": "apple leaf",
        
        # Maíz (Corn)
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "corn gray leaf spot",
        "Corn_(maize)___Common_rust_": "corn rust",
        "Corn_(maize)___Northern_Leaf_Blight": "corn northern leaf blight",
        "Corn_(maize)___healthy": "corn leaf",
        
        # Vid (Grape)
        "Grape___Black_rot": "grape black rot",
        "Grape___Esca_(Black_Measles)": None,  # No existe Esca en PlantWild
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape leaf spot",  # Leaf spot / Blight
        "Grape___healthy": "grape leaf",
        
        # Pimiento (Pepper)
        "Pepper,_bell___Bacterial_spot": "bell pepper leaf spot",
        "Pepper,_bell___healthy": "bell pepper leaf",
        
        # Papa (Potato)
        "Potato___Early_blight": "potato early blight",
        "Potato___Late_blight": "potato late blight",
        "Potato___healthy": "potato leaf",
        
        # Tomate (Tomato)
        "Tomato___Bacterial_spot": "tomato bacterial leaf spot",
        "Tomato___Early_blight": "tomato early blight",
        "Tomato___Late_blight": "tomato late blight",
        "Tomato___Leaf_Mold": "tomato leaf mold",
        "Tomato___Septoria_leaf_spot": "tomato septoria leaf spot",
        "Tomato___Spider_mites Two-spotted_spider_mite": None,  # No existe en PlantWild
        "Tomato___Target_Spot": None,  # No existe en PlantWild
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato yellow leaf curl virus",
        "Tomato___Tomato_mosaic_virus": "tomato mosaic virus",
        "Tomato___healthy": "tomato leaf"
    }

    matched = []
    unmatched = []

    for idx, pv_c in enumerate(pv_classes):
        m = pv_meta.get(pv_c, {})
        crop_es = m.get('crop_es', pv_c.split('___')[0])
        dis_es = m.get('disease_es', pv_c.split('___')[1])
        disp = f"{crop_es}: {dis_es}"
        
        pw_target = mapping.get(pv_c)
        if pw_target and pw_target in pw_classes:
            matched.append((idx, pv_c, disp, pw_target))
        else:
            unmatched.append((idx, pv_c, disp))

    print(f"\n✅ CLASES COINCIDENTES: {len(matched)} de 27 ({len(matched)/27*100:.1f}%)")
    print("-" * 80)
    for idx, pv_c, disp, pw_c in matched:
        print(f"[{idx:02d}] {disp:<42} <---> PlantWild: \"{pw_c}\"")

    print(f"\n❌ CLASES SIN EQUIVALENTE EN PLANTWILD: {len(unmatched)} de 27 ({len(unmatched)/27*100:.1f}%)")
    print("-" * 80)
    for idx, pv_c, disp in unmatched:
        print(f"[{idx:02d}] {disp:<42} (Clase PV: {pv_c})")

    # Clases adicionales en PlantWild para estos 6 cultivos
    print("\n" + "=" * 80)
    print("  CLASES ADICIONALES EN PLANTWILD (No presentes en PlantVillage)")
    print("=" * 80)
    
    used_pw = set([m[3] for m in matched])
    extra_pw = []
    for crop, kws in target_crops.items():
        found = [c for c in pw_classes if any(k in c.lower() for k in kws) and c not in used_pw]
        for c in found:
            extra_pw.append((crop, c))
            
    for crop, c in extra_pw:
        print(f"  • {crop}: \"{c}\"")

if __name__ == "__main__":
    analyze_plantwild()
