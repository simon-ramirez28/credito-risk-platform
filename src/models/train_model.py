"""
Entrenamiento de modelo de machine learning para riesgo crediticio.
Enfoque simple y reproducible para ingenieros de datos.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import json
import joblib
from typing import Dict, Any, Tuple, Optional, List

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CreditRiskModel:
    """
    Clase para entrenar y gestionar modelos de riesgo crediticio.
    """

    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        """
        Inicializa el modelo.

        Args:
            model_type: Tipo de modelo ('random_forest', 'gradient_boosting', 'logistic')
            random_state: Semilla para reproducibilidad
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names: Optional[List[str]] = None
        self.target_name = "default"
        self.metrics: Dict[str, Any] = {}
        self.best_params: Dict[str, Any] = {}

        # Inicializar modelo basado en tipo
        self._initialize_model()

        logger.info(f"Modelo {model_type} inicializado con random_state={random_state}")

    def _initialize_model(self):
        """Inicializa el modelo según el tipo especificado."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced",
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
            )
        elif self.model_type == "logistic":
            self.model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
            )
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
        """
        Prepara datos para entrenamiento.

        Args:
            df: DataFrame con features y target
            test_size: Proporción para test split

        Returns:
            Tuple con X_train, X_test, y_train, y_test
        """
        logger.info("Preparando datos para entrenamiento...")

        # Verificar que existe la columna target
        if self.target_name not in df.columns:
            raise ValueError(f"Target column '{self.target_name}' no encontrada")

        # Separar features y target
        X = df.drop(columns=[self.target_name])
        y = df[self.target_name]

        # Guardar nombres de features
        self.feature_names = X.columns.tolist()
        if self.feature_names is not None:
            logger.info(f"Número de features: {len(self.feature_names)}")
        logger.info(f"Distribución de clases: {dict(y.value_counts())}")

        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,  # Mantener proporción de clases
            shuffle=True,
        )

        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def handle_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple:
        """
        Maneja desbalance de clases usando SMOTE.

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento

        Returns:
            Tupla con datos balanceados
        """
        logger.info("Aplicando SMOTE para balancear clases...")

        # Verificar desbalance
        class_counts = y_train.value_counts()
        logger.info(f"Distribución antes de SMOTE: {dict(class_counts)}")

        # Aplicar SMOTE solo si hay desbalance significativo
        if class_counts.min() / class_counts.max() < 0.3:
            smote = SMOTE(
                random_state=self.random_state,
                sampling_strategy="auto",
                k_neighbors=min(5, class_counts.min() - 1),
            )

            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

            logger.info(
                f"Distribución después de SMOTE: {dict(pd.Series(y_train_balanced).value_counts())}"
            )
            return X_train_balanced, y_train_balanced

        logger.info("No se aplicó SMOTE (clases relativamente balanceadas)")
        return X_train, y_train

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_cv: bool = True,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Entrena el modelo.

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            use_cv: Usar cross-validation
            cv_folds: Número de folds para CV

        Returns:
            Diccionario con métricas de entrenamiento
        """
        logger.info(f"Entrenando modelo {self.model_type}...")

        # Manejar desbalance de clases
        X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train)

        # Entrenar modelo
        if use_cv:
            # Cross-validation para evaluación robusta
            cv_scores = cross_val_score(
                self.model,
                X_train_balanced,
                y_train_balanced,
                cv=cv_folds,
                scoring="roc_auc",
                n_jobs=-1,
            )

            logger.info(
                f"Cross-validation ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
            )
            self.metrics["cv_scores"] = cv_scores.tolist()
            self.metrics["cv_mean"] = float(cv_scores.mean())
            self.metrics["cv_std"] = float(cv_scores.std())

        # Entrenar con todos los datos
        if self.model is None:
            raise ValueError("Modelo no inicializado")
        self.model.fit(X_train_balanced, y_train_balanced)

        # Calcular métricas en training
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        y_train_pred = self.model.predict(X_train_balanced)
        y_train_proba = self.model.predict_proba(X_train_balanced)[:, 1]

        train_metrics = self._calculate_metrics(
            y_train_balanced, y_train_pred, y_train_proba
        )
        self.metrics["train"] = train_metrics

        logger.info(
            f"Entrenamiento completado. Train ROC AUC: {train_metrics['roc_auc']:.4f}"
        )
        return self.metrics

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evalúa el modelo en datos de test.

        Args:
            X_test: Features de test
            y_test: Target de test

        Returns:
            Diccionario con métricas de evaluación
        """
        logger.info("Evaluando modelo en test set...")

        if self.model is None:
            raise ValueError("Modelo no entrenado")
        
        # Predecir
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)[:, 1]

        # Calcular métricas
        test_metrics = self._calculate_metrics(y_test, y_test_pred, y_test_proba)
        self.metrics["test"] = test_metrics

        # Guardar predicciones para análisis
        self.y_test = y_test
        self.y_test_pred = y_test_pred
        self.y_test_proba = y_test_proba

        # Mostrar reporte de clasificación
        logger.info("\n📊 Classification Report:")
        logger.info(f"\n{classification_report(y_test, y_test_pred)}")

        # Mostrar matriz de confusión
        cm = confusion_matrix(y_test, y_test_pred)
        logger.info(f"\n🎯 Confusion Matrix:\n{cm}")

        logger.info(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")
        return test_metrics

    def _calculate_metrics(
        self, y_true: pd.Series, y_pred: pd.Series, y_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcula múltiples métricas de evaluación.

        Args:
            y_true: Valores reales
            y_pred: Predicciones
            y_proba: Probabilidades predichas

        Returns:
            Diccionario con métricas
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": (
                roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5
            ),
        }

        # Calcular balanced accuracy para datos desbalanceados
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics["balanced_accuracy"] = (specificity + sensitivity) / 2
        else:
            metrics["balanced_accuracy"] = metrics["accuracy"]

        return {k: float(v) for k, v in metrics.items()}

    def hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Realiza búsqueda de hiperparámetros.

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            param_grid: Grid de parámetros a buscar

        Returns:
            Mejores parámetros encontrados
        """
        logger.info("Realizando búsqueda de hiperparámetros...")

        # Grid de parámetros por defecto
        if param_grid is None:
            if self.model_type == "random_forest":
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                }
            elif self.model_type == "gradient_boosting":
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                }
            elif self.model_type == "logistic":
                param_grid = {
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"],
                }

        # Búsqueda en grid con cross-validation
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        # Actualizar modelo con mejores parámetros
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        logger.info(f"Mejores parámetros encontrados: {self.best_params}")
        logger.info(f"Mejor score CV: {grid_search.best_score_:.4f}")

        self.metrics["best_params"] = self.best_params
        self.metrics["best_cv_score"] = float(grid_search.best_score_)

        return self.best_params

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Obtiene importancia de features del modelo entrenado.

        Returns:
            DataFrame con importancia de features
        """
        if self.model is None:
            logger.warning("Modelo no entrenado")
            return pd.DataFrame()

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_

            # Crear DataFrame con importancias
            feature_importance = pd.DataFrame(
                {"feature": self.feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)

            logger.info(f"Top 5 features más importantes:")
            for i, row in feature_importance.head().iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

            return feature_importance

        elif hasattr(self.model, "coef_"):
            # Para modelos lineales como Logistic Regression
            if len(self.model.coef_.shape) > 1:
                importances = np.abs(self.model.coef_[0])
            else:
                importances = np.abs(self.model.coef_)

            feature_importance = pd.DataFrame(
                {"feature": self.feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)

            return feature_importance

        else:
            logger.warning("Modelo no tiene atributo de importancia de features")
            return pd.DataFrame()

    def save_model(self, output_dir: str = "models"):
        """
        Guarda el modelo entrenado y metadata.

        Args:
            output_dir: Directorio donde guardar el modelo
        """
        # Crear directorio si no existe
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.model_type}_model_{timestamp}"

        # Guardar modelo
        model_path = Path(output_dir) / f"{model_name}.pkl"
        joblib.dump(self.model, model_path)

        # Guardar metadata
        metadata = {
            "model_type": self.model_type,
            "model_name": model_name,
            "timestamp": timestamp,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "best_params": self.best_params,
            "metrics": self.metrics,
            "model_path": str(model_path),
            "feature_count": len(self.feature_names) if self.feature_names else 0,
        }

        metadata_path = Path(output_dir) / f"{model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Guardar importancia de features
        feature_importance = self.get_feature_importance()
        if not feature_importance.empty:
            importance_path = Path(output_dir) / f"{model_name}_feature_importance.csv"
            feature_importance.to_csv(importance_path, index=False)
            metadata["feature_importance_path"] = str(importance_path)

        logger.info(f"Modelo guardado en: {model_path}")
        logger.info(f"Metadata guardada en: {metadata_path}")

        return {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "metadata": metadata,
        }

    def create_model_report(self) -> str:
        """
        Crea un reporte detallado del modelo.

        Returns:
            String con reporte del modelo
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"📊 REPORTE DEL MODELO DE RIESGO CREDITICIO")
        report_lines.append("=" * 60)

        # Información del modelo
        report_lines.append(f"\n🤖 INFORMACIÓN DEL MODELO:")
        report_lines.append(f"  Tipo: {self.model_type}")
        report_lines.append(f"  Random State: {self.random_state}")

        if self.best_params:
            report_lines.append(f"  Mejores parámetros: {self.best_params}")

        # Métricas
        if "train" in self.metrics:
            report_lines.append(f"\n📈 MÉTRICAS DE ENTRENAMIENTO:")
            for metric, value in self.metrics["train"].items():
                report_lines.append(f"  {metric}: {value:.4f}")

        if "test" in self.metrics:
            report_lines.append(f"\n🧪 MÉTRICAS DE TEST:")
            for metric, value in self.metrics["test"].items():
                report_lines.append(f"  {metric}: {value:.4f}")

        if "cv_scores" in self.metrics:
            report_lines.append(f"\n🔄 CROSS-VALIDATION:")
            report_lines.append(f"  Mean ROC AUC: {self.metrics['cv_mean']:.4f}")
            report_lines.append(f"  Std ROC AUC: {self.metrics['cv_std']:.4f}")

        # Feature importance
        feature_importance = self.get_feature_importance()
        if not feature_importance.empty:
            report_lines.append(f"\n🎯 TOP 10 FEATURES MÁS IMPORTANTES:")
            for i, row in feature_importance.head(10).iterrows():
                report_lines.append(
                    f"  {i+1:2d}. {row['feature'][:40]:40s} : {row['importance']:.4f}"
                )

        report_lines.append("\n" + "=" * 60)

        return "\n".join(report_lines)


# Función principal para entrenamiento desde línea de comandos
def train_model_from_cli():
    """Función principal para entrenar modelo desde CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Entrenar modelo de riesgo crediticio")
    parser.add_argument(
        "--data", type=str, required=True, help="Archivo con features (CSV o Parquet)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "gradient_boosting", "logistic"],
        help="Tipo de modelo a entrenar",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directorio para guardar modelo",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Proporción para test split"
    )
    parser.add_argument(
        "--tune-hyperparams",
        action="store_true",
        help="Realizar búsqueda de hiperparámetros",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Semilla para reproducibilidad"
    )

    args = parser.parse_args()

    # Cargar datos
    logger.info(f"Cargando datos desde: {args.data}")
    if args.data.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    logger.info(f"Datos cargados: {len(df)} registros, {len(df.columns)} columnas")

    # Verificar que existe la columna target
    if "default" not in df.columns:
        logger.error("Columna 'default' no encontrada en los datos")
        return

    # Inicializar y entrenar modelo
    model = CreditRiskModel(model_type=args.model_type, random_state=args.random_state)

    # Preparar datos
    X_train, X_test, y_train, y_test = model.prepare_data(df, test_size=args.test_size)

    # Búsqueda de hiperparámetros si se solicita
    if args.tune_hyperparams:
        model.hyperparameter_tuning(X_train, y_train)

    # Entrenar modelo
    model.train(X_train, y_train, use_cv=True)

    # Evaluar modelo
    model.evaluate(X_test, y_test)

    # Guardar modelo
    saved_files = model.save_model(output_dir=args.output_dir)

    # Mostrar reporte
    print("\n" + model.create_model_report())

    print(f"\n✅ Modelo entrenado y guardado exitosamente")
    print(f"   Modelo: {saved_files['model_path']}")
    print(f"   Metadata: {saved_files['metadata_path']}")

    return model


if __name__ == "__main__":
    train_model_from_cli()
