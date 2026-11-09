#!/usr/bin/env python
"""
Acceso directo para predecir con Ensemble MVP 4 + MVP 6 desde la raíz del proyecto:
python predict_ensemble.py --image test-plant.jpg
"""
import sys
import argparse
from src.predict_ensemble import predict_ensemble

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgroVision: Diagnóstico por Ensemble MVP 4 + MVP 6")
    parser.add_argument("--image", type=str, default="test-plant.jpg", help="Ruta de la imagen de prueba")
    parser.add_argument("--model4", type=str, default="models/agrovision_mobilenetv2_mvp4.keras", help="Ruta del modelo MVP 4")
    parser.add_argument("--model6", type=str, default="models/agrovision_mobilenetv2_mvp6.keras", help="Ruta del modelo MVP 6")
    parser.add_argument("--classes", type=str, default="models/class_names_mvp2.json", help="Ruta de class_names")
    parser.add_argument("--mapping", type=str, default="models/class_mapping_mvp2.json", help="Ruta de class_mapping")
    parser.add_argument("--weight4", type=float, default=0.5, help="Peso para MVP 4 (por defecto 0.5)")
    parser.add_argument("--weight6", type=float, default=0.5, help="Peso para MVP 6 (por defecto 0.5)")
    parser.add_argument("--top", type=int, default=5, help="Número de probabilidades a mostrar")
    args = parser.parse_args()

    predict_ensemble(
        image_path=args.image,
        model4_path=args.model4,
        model6_path=args.model6,
        classes_path=args.classes,
        mapping_path=args.mapping,
        weight4=args.weight4,
        weight6=args.weight6,
        top_k=args.top
    )
