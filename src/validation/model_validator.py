"""
Validación y monitoreo de modelos de riesgo crediticio.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import json
import joblib
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Clase para validar y monitorear modelos de riesgo crediticio.
    """

    def __init__(self, model_path: str, metadata_path: str):
        """
        Inicializa el validador.

        Args:
            model_path: Ruta al modelo guardado (.pkl)
            metadata_path: Ruta a metadata del modelo (.json)
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        # Cargar modelo y metadata
        self.model = joblib.load(self.model_path)
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata.get("feature_names", [])
        self.target_name = self.metadata.get("target_name", "default")

        logger.info(f"Validador inicializado para modelo: {self.model_path.name}")

    def validate_on_new_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida el modelo en nuevos datos.

        Args:
            df: DataFrame con nuevos datos

        Returns:
            Diccionario con métricas de validación
        """
        logger.info("Validando modelo en nuevos datos...")

        # Verificar que tenemos todas las features necesarias
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            logger.warning(f"Features faltantes en nuevos datos: {missing_features}")
            # Intentar imputar o manejar features faltantes
            for feature in missing_features:
                if feature in df.columns:
                    continue
                # Si la feature es categórica dummy creada, puede no existir en nuevos datos
                if "_" in feature and any(
                    base in feature for base in ["estado", "nivel", "tipo"]
                ):
                    df[feature] = 0  # Asumir categoría no presente
                else:
                    df[feature] = np.nan

        # Asegurar el orden correcto de features
        X = df[self.feature_names].copy() if self.feature_names else df.copy()

        # Verificar si hay target
        if self.target_name in df.columns:
            y_true = df[self.target_name]
            has_target = True
        else:
            y_true = None
            has_target = False

        # Predecir
        y_pred = self.model.predict(X)
        y_proba = (
            self.model.predict_proba(X)[:, 1]
            if hasattr(self.model, "predict_proba")
            else None
        )

        # Calcular métricas si hay target
        metrics = {}
        if has_target and y_true is not None:
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                roc_auc_score,
                confusion_matrix,
            )

            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
                "roc_auc": (
                    float(roc_auc_score(y_true, y_proba))
                    if y_proba is not None
                    else 0.5
                ),
                "sample_size": len(df),
            }

            # Calcular matriz de confusión
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            metrics["true_negatives"] = int(cm[0, 0])
            metrics["false_positives"] = int(cm[0, 1])
            metrics["false_negatives"] = int(cm[1, 0])
            metrics["true_positives"] = int(cm[1, 1])

            logger.info(f"Métricas de validación: {metrics}")

        # Crear DataFrame con predicciones
        predictions_df = df.copy()
        predictions_df["prediction"] = y_pred
        if y_proba is not None:
            predictions_df["probability_default"] = y_proba
            predictions_df["risk_category"] = pd.cut(
                y_proba,
                bins=[0, 0.3, 0.5, 0.7, 1.0],
                labels=["BAJO", "MODERADO", "ALTO", "MUY_ALTO"],
            )

        return {
            "metrics": metrics,
            "predictions_df": predictions_df,
            "has_target": has_target,
        }

    def calculate_drift_metrics(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calcula métricas de drift de datos.

        Args:
            reference_data: Datos de referencia (entrenamiento)
            current_data: Datos actuales (producción)

        Returns:
            Diccionario con métricas de drift
        """
        logger.info("Calculando métricas de drift...")

        drift_metrics = {}

        # Para cada feature numérica
        numeric_cols = reference_data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in current_data.columns:
                ref_mean = reference_data[col].mean()
                ref_std = reference_data[col].std()
                curr_mean = current_data[col].mean()

                # Calcular Z-score de la diferencia
                if ref_std > 0:
                    z_score = abs(curr_mean - ref_mean) / ref_std
                else:
                    z_score = 0

                drift_metrics[col] = {
                    "reference_mean": float(ref_mean),
                    "current_mean": float(curr_mean),
                    "z_score": float(z_score),
                    "has_drift": z_score > 2.0,  # 2 desviaciones estándar
                }

        # Calcular drift en distribución de target si existe
        if (
            self.target_name in reference_data.columns
            and self.target_name in current_data.columns
        ):
            ref_target_dist = reference_data[self.target_name].value_counts(
                normalize=True
            )
            curr_target_dist = current_data[self.target_name].value_counts(
                normalize=True
            )

            drift_metrics["target_distribution"] = {
                "reference": ref_target_dist.to_dict(),
                "current": curr_target_dist.to_dict(),
                "js_divergence": self._jensen_shannon_divergence(
                    ref_target_dist.values, curr_target_dist.values
                ),
            }

        # Resumen de drift
        drifted_features = [
            col
            for col, metrics in drift_metrics.items()
            if isinstance(metrics, dict) and metrics.get("has_drift", False)
        ]

        drift_summary = {
            "total_features_checked": len(numeric_cols),
            "drifted_features": drifted_features,
            "drift_percentage": len(drifted_features) / max(len(numeric_cols), 1) * 100,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Drift detectado en {len(drifted_features)}/{len(numeric_cols)} features"
        )

        return {"drift_metrics": drift_metrics, "drift_summary": drift_summary}

    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calcula la divergencia de Jensen-Shannon entre dos distribuciones."""
        # Normalizar
        p = np.asarray(p)
        q = np.asarray(q)
        p = p / p.sum()
        q = q / q.sum()

        m = 0.5 * (p + q)

        # Calcular KL divergence
        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))

        js_divergence = 0.5 * (kl_pm + kl_qm)
        return float(js_divergence)

    def generate_validation_report(
        self, validation_results: Dict[str, Any], output_dir: str = "reports"
    ) -> str:
        """
        Genera un reporte de validación.

        Args:
            validation_results: Resultados de validación
            output_dir: Directorio para guardar reportes

        Returns:
            Ruta al reporte generado
        """
        logger.info("Generando reporte de validación...")

        # Crear directorio si no existe
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(output_dir) / f"validation_report_{timestamp}.txt"

        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("📋 REPORTE DE VALIDACIÓN DEL MODELO\n")
            f.write("=" * 70 + "\n\n")

            # Información del modelo
            f.write(f"🤖 INFORMACIÓN DEL MODELO:\n")
            f.write(f"  Modelo: {self.model_path.name}\n")
            f.write(f"  Tipo: {self.metadata.get('model_type', 'N/A')}\n")
            f.write(f"  Fecha entrenamiento: {self.metadata.get('timestamp', 'N/A')}\n")
            f.write(f"  Features: {len(self.feature_names)}\n\n")

            # Métricas de validación
            if validation_results.get("has_target", False):
                metrics = validation_results["metrics"]
                f.write(f"📊 MÉTRICAS DE VALIDACIÓN:\n")
                for metric_name, value in metrics.items():
                    if metric_name not in [
                        "confusion_matrix",
                        "sample_size",
                        "true_negatives",
                        "false_positives",
                        "false_negatives",
                        "true_positives",
                    ]:
                        f.write(f"  {metric_name}: {value:.4f}\n")

                # Matriz de confusión
                if "confusion_matrix" in metrics:
                    f.write(f"\n🎯 MATRIZ DE CONFUSIÓN:\n")
                    f.write(
                        f"  TN: {metrics['true_negatives']} | FP: {metrics['false_positives']}\n"
                    )
                    f.write(
                        f"  FN: {metrics['false_negatives']} | TP: {metrics['true_positives']}\n"
                    )

                f.write(f"  Tamaño de muestra: {metrics.get('sample_size', 'N/A')}\n")

            # Distribución de predicciones
            predictions_df = validation_results.get("predictions_df")
            if predictions_df is not None and "prediction" in predictions_df.columns:
                f.write(f"\n📈 DISTRIBUCIÓN DE PREDICCIONES:\n")
                pred_dist = predictions_df["prediction"].value_counts()
                for value, count in pred_dist.items():
                    f.write(
                        f"  Clase {value}: {count} ({count/len(predictions_df)*100:.1f}%)\n"
                    )

                if "risk_category" in predictions_df.columns:
                    f.write(f"\n📊 DISTRIBUCIÓN DE CATEGORÍAS DE RIESGO:\n")
                    risk_dist = predictions_df["risk_category"].value_counts()
                    for category, count in risk_dist.items():
                        f.write(
                            f"  {category}: {count} ({count/len(predictions_df)*100:.1f}%)\n"
                        )

            # Comparación con métricas originales
            original_metrics = self.metadata.get("metrics", {})
            if original_metrics and validation_results.get("has_target", False):
                f.write(f"\n🔄 COMPARACIÓN CON MÉTRICAS ORIGINALES:\n")
                if "test" in original_metrics:
                    f.write(
                        f"  ROC AUC Original: {original_metrics['test'].get('roc_auc', 'N/A'):.4f}\n"
                    )
                    f.write(
                        f"  ROC AUC Validación: {validation_results['metrics'].get('roc_auc', 'N/A'):.4f}\n"
                    )
                    diff = validation_results["metrics"].get(
                        "roc_auc", 0
                    ) - original_metrics["test"].get("roc_auc", 0)
                    f.write(f"  Diferencia: {diff:+.4f}\n")

        logger.info(f"Reporte generado en: {report_path}")
        return str(report_path)

    def create_validation_plots(
        self, validation_results: Dict[str, Any], output_dir: str = "reports/plots"
    ):
        """
        Crea gráficos de validación.

        Args:
            validation_results: Resultados de validación
            output_dir: Directorio para guardar gráficos
        """
        logger.info("Creando gráficos de validación...")

        # Crear directorio si no existe
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_df = validation_results.get("predictions_df")

        if predictions_df is None:
            logger.warning("No hay datos de predicciones para crear gráficos")
            return

        # 1. Distribución de probabilidades
        if "probability_default" in predictions_df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(
                data=predictions_df, x="probability_default", bins=30, kde=True
            )
            plt.title("Distribución de Probabilidades de Default")
            plt.xlabel("Probabilidad de Default")
            plt.ylabel("Frecuencia")
            plt.tight_layout()
            plt.savefig(Path(output_dir) / f"prob_distribution_{timestamp}.png")
            plt.close()

        # 2. Distribución de categorías de riesgo
        if "risk_category" in predictions_df.columns:
            plt.figure(figsize=(10, 6))
            risk_counts = predictions_df["risk_category"].value_counts().sort_index()
            colors = ["green", "yellow", "orange", "red"]
            risk_counts.plot(kind="bar", color=colors)
            plt.title("Distribución de Categorías de Riesgo")
            plt.xlabel("Categoría de Riesgo")
            plt.ylabel("Número de Clientes")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(Path(output_dir) / f"risk_categories_{timestamp}.png")
            plt.close()

        # 3. Curva de calibración (si hay target real)
        if (
            validation_results.get("has_target", False)
            and "probability_default" in predictions_df.columns
            and self.target_name in predictions_df.columns
        ):

            y_true = predictions_df[self.target_name]
            y_proba = predictions_df["probability_default"]

            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

            plt.figure(figsize=(10, 6))
            plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Modelo")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfecto")
            plt.title("Curva de Calibración del Modelo")
            plt.xlabel("Probabilidad Predicha Promedio")
            plt.ylabel("Fracción de Positivos")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(Path(output_dir) / f"calibration_curve_{timestamp}.png")
            plt.close()

        logger.info(f"Gráficos guardados en: {output_dir}")


# Función principal para validación desde CLI
def validate_model_cli():
    """Función principal para validar modelo desde CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Validar modelo de riesgo crediticio")
    parser.add_argument(
        "--model", type=str, required=True, help="Ruta al modelo guardado (.pkl)"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Archivo con datos para validación"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="Ruta a metadata del modelo (inferida si no se proporciona)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directorio para guardar reportes",
    )

    args = parser.parse_args()

    # Inferir metadata path si no se proporciona
    if args.metadata is None:
        metadata_path = Path(args.model).with_suffix(".json")
        if metadata_path.exists():
            args.metadata = str(metadata_path)
        else:
            # Buscar metadata con mismo nombre base
            model_dir = Path(args.model).parent
            model_name = Path(args.model).stem
            metadata_files = list(model_dir.glob(f"*{model_name}*metadata*.json"))
            if metadata_files:
                args.metadata = str(metadata_files[0])
            else:
                logger.error("No se pudo encontrar archivo de metadata")
                return

    # Inicializar validador
    validator = ModelValidator(args.model, args.metadata)

    # Cargar datos de validación
    logger.info(f"Cargando datos de validación desde: {args.data}")
    if args.data.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    logger.info(f"Datos cargados: {len(df)} registros")

    # Validar modelo
    validation_results = validator.validate_on_new_data(df)

    # Generar reporte
    report_path = validator.generate_validation_report(
        validation_results, args.output_dir
    )

    # Crear gráficos
    validator.create_validation_plots(validation_results, f"{args.output_dir}/plots")

    print(f"\n✅ Validación completada exitosamente")
    print(f"   Reporte generado: {report_path}")

    # Mostrar métricas si hay target
    if validation_results.get("has_target", False):
        metrics = validation_results["metrics"]
        print(f"\n📊 Métricas de validación:")
        for metric_name, value in metrics.items():
            if metric_name not in ["confusion_matrix", "sample_size"]:
                print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    validate_model_cli()
