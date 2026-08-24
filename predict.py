#!/usr/bin/env python
"""
Acceso directo para predecir desde la raíz del proyecto AgroVision:
python predict.py [--image test-plant.jpg]
"""
import sys
import argparse
from src.predict import predict_single_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgroVision: Predecir estado/enfermedad de una planta")
    parser.add_argument("--image", type=str, default="test-plant.jpg", help="Ruta de la imagen de prueba")
    parser.add_argument("--model", type=str, default="models/agrovision_mobilenetv2_mvp4.keras", help="Ruta del modelo")
    parser.add_argument("--classes", type=str, default="models/class_names_mvp2.json", help="Ruta de class_names")
    parser.add_argument("--mapping", type=str, default="models/class_mapping_mvp2.json", help="Ruta de class_mapping")
    parser.add_argument("--top", type=int, default=5, help="Número de probabilidades a mostrar")
    args = parser.parse_args()

    predict_single_image(
        image_path=args.image,
        model_path=args.model,
        classes_path=args.classes,
        mapping_path=args.mapping,
        top_k=args.top
    )
