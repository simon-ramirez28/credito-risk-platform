"""
Feature engineering avanzado para datos crediticios.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import category_encoders as ce
from feature_engine import encoding as fe_encoding
from feature_engine import imputation as fe_imputation

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CreditFeatureEngineer:
    """
    Clase para feature engineering avanzado de datos crediticios.
    """

    def __init__(self, target_col: str = "default", random_state: int = 42):
        """
        Inicializa el feature engineer.

        Args:
            target_col: Nombre de la columna objetivo
            random_state: Semilla para reproducibilidad
        """
        self.target_col = target_col
        self.random_state = random_state
        self.scaler = None
        self.imputer = None
        self.encoder = None
        self.feature_names = None
        self.feature_metadata = {}

        logger.info(f"Feature engineer inicializado con target: {target_col}")

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features avanzadas basadas en dominio financiero.

        Args:
            df: DataFrame con datos procesados básicos

        Returns:
            DataFrame con features avanzadas
        """
        logger.info("Creando features avanzadas...")
        df_features = df.copy()

        # 1. FEATURES DE RIESGO FINANCIERO
        df_features = self._create_financial_risk_features(df_features)

        # 2. FEATURES DE COMPORTAMIENTO
        df_features = self._create_behavioral_features(df_features)

        # 3. FEATURES TEMPORALES
        df_features = self._create_temporal_features(df_features)

        # 4. FEATURES DE INTERACCIÓN
        df_features = self._create_interaction_features(df_features)

        # 5. FEATURES AGRUPADAS
        df_features = self._create_grouped_features(df_features)

        # 6. FEATURES DE SCORE COMPUESTO
        df_features = self._create_composite_score_features(df_features)

        logger.info(f"Features creadas: {len(df_features.columns)} columnas totales")
        return df_features

    def _create_financial_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de riesgo financiero."""

        # 1. Capacidad de pago
        if all(col in df.columns for col in ["ingreso_mensual", "gastos_mensuales"]):
            df["capacidad_pago"] = df["ingreso_mensual"] - df["gastos_mensuales"]
            df["ratio_capacidad_pago"] = np.where(
                df["gastos_mensuales"] > 0,
                df["capacidad_pago"] / df["gastos_mensuales"],
                0,
            )

        # 2. Niveles de endeudamiento
        if all(col in df.columns for col in ["total_adeudado", "ingreso_anual"]):
            df["deuda_ingreso_anual_ratio"] = np.where(
                df["ingreso_anual"] > 0, df["total_adeudado"] / df["ingreso_anual"], 0
            )

        # 3. Líquidez
        if all(col in df.columns for col in ["ahorros", "gastos_mensuales"]):
            df["meses_liquidez"] = np.where(
                df["gastos_mensuales"] > 0, df["ahorros"] / df["gastos_mensuales"], 0
            )

        # 4. Estabilidad de ingresos
        if "antiguedad_empleo" in df.columns:
            df["estabilidad_empleo"] = np.where(
                df["antiguedad_empleo"] >= 24,  # 2 años
                "alta",
                np.where(df["antiguedad_empleo"] >= 12, "media", "baja"),
            )

        # 5. Score bancario categorizado
        if "score_bancario" in df.columns:
            bins = [300, 580, 670, 740, 800, 850]
            labels = ["muy_bajo", "bajo", "regular", "bueno", "excelente"]
            df["score_bancario_cat"] = pd.cut(
                df["score_bancario"], bins=bins, labels=labels, include_lowest=True
            )

        return df

    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de comportamiento crediticio."""

        # 1. Historial de mora
        if all(col in df.columns for col in ["max_dias_mora", "num_creditos_previos"]):
            df["tiene_mora_grave"] = (df["max_dias_mora"] > 90).astype(int)
            df["frecuencia_mora"] = np.where(
                df["num_creditos_previos"] > 0,
                df["creditos_problematicos"] / df["num_creditos_previos"],
                0,
            )

        # 2. Patrón de uso de crédito
        if all(col in df.columns for col in ["num_creditos_previos", "edad"]):
            df["edad_primer_credito"] = np.where(
                df["num_creditos_previos"] > 0,
                df["edad"] - (df["num_creditos_previos"] * 2),  # Estimación
                df["edad"],
            )

            df["intensidad_crediticia"] = np.where(
                df["edad"] > 18, df["num_creditos_previos"] / (df["edad"] - 18), 0
            )

        # 3. Diversificación de créditos
        # (En producción, esto vendría de datos más detallados)
        df["diversificacion_creditos"] = np.random.choice(
            ["baja", "media", "alta"], size=len(df), p=[0.6, 0.3, 0.1]
        )

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features temporales y de tendencia."""

        # 1. Edad en diferentes categorías
        if "edad" in df.columns:
            df["edad_decada"] = (df["edad"] // 10) * 10
            df["es_joven_adulto"] = df["edad"].between(18, 30).astype(int)
            df["es_mediana_edad"] = df["edad"].between(31, 50).astype(int)
            df["es_mayor"] = (df["edad"] > 50).astype(int)

        # 2. Estacionalidad (simulada)
        # En producción, usarías fechas reales
        meses = np.random.choice(range(1, 13), size=len(df))
        df["mes_solicitud"] = meses
        df["es_fin_ano"] = df["mes_solicitud"].isin([11, 12]).astype(int)

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de interacción entre variables."""

        # 1. Interacción edad-ingreso
        if all(col in df.columns for col in ["edad", "ingreso_mensual"]):
            df["edad_ingreso_interaction"] = df["edad"] * df["ingreso_mensual"]
            df["ingreso_por_ano_experiencia"] = np.where(
                df["edad"] > 18, df["ingreso_mensual"] / (df["edad"] - 18), 0
            )

        # 2. Interacción deuda-dependientes
        if all(col in df.columns for col in ["total_adeudado", "dependientes"]):
            df["deuda_por_dependiente"] = np.where(
                df["dependientes"] > 0,
                df["total_adeudado"] / df["dependientes"],
                df["total_adeudado"],
            )

        # 3. Interacción score-mora
        if all(col in df.columns for col in ["score_bancario", "max_dias_mora"]):
            df["score_ajustado_mora"] = df["score_bancario"] - (
                df["max_dias_mora"] * 0.1
            )

        return df

    def _create_grouped_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features agrupadas por categorías."""

        # Agrupaciones por ciudad (si hay suficientes datos)
        if "ciudad" in df.columns and len(df["ciudad"].unique()) > 1:
            # Estadísticas por ciudad
            city_stats = (
                df.groupby("ciudad")
                .agg({"ingreso_mensual": ["mean", "std"], "default": "mean"})
                .round(2)
            )

            city_stats.columns = [
                "ingreso_promedio_ciudad",
                "ingreso_std_ciudad",
                "default_rate_ciudad",
            ]

            df = df.merge(city_stats, on="ciudad", how="left")

            # Comparación con promedio de ciudad
            df["ingreso_vs_ciudad"] = np.where(
                df["ingreso_promedio_ciudad"] > 0,
                df["ingreso_mensual"] / df["ingreso_promedio_ciudad"],
                1,
            )

        # Agrupaciones por nivel educativo
        if "nivel_educacion" in df.columns:
            edu_stats = (
                df.groupby("nivel_educacion")
                .agg({"ingreso_mensual": "mean", "default": "mean"})
                .round(2)
            )

            edu_stats.columns = ["ingreso_promedio_educacion", "default_rate_educacion"]

            df = df.merge(edu_stats, on="nivel_educacion", how="left")

        return df

    def _create_composite_score_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea scores compuestos basados en múltiples factores."""

        scores = []
        weights = []

        # 1. Score de capacidad financiera
        if all(
            col in df.columns
            for col in ["ingreso_mensual", "gastos_mensuales", "ahorros"]
        ):
            capacidad_score = np.where(
                df["gastos_mensuales"] > 0,
                (df["ingreso_mensual"] - df["gastos_mensuales"])
                / df["gastos_mensuales"],
                0,
            )
            # Normalizar a 0-1
            capacidad_score = np.clip(capacidad_score, 0, 2) / 2
            scores.append(capacidad_score)
            weights.append(0.25)

        # 2. Score de historial crediticio
        if all(
            col in df.columns for col in ["max_dias_mora", "creditos_problematicos"]
        ):
            historial_score = 1 - (
                (df["max_dias_mora"].clip(0, 365) / 365) * 0.5
                + (df["creditos_problematicos"].clip(0, 10) / 10) * 0.5
            )
            scores.append(historial_score)
            weights.append(0.35)

        # 3. Score de estabilidad
        if all(
            col in df.columns for col in ["antiguedad_empleo", "antiguedad_residencia"]
        ):
            estabilidad_score = (np.clip(df["antiguedad_empleo"], 0, 60) / 60) * 0.6 + (
                np.clip(df["antiguedad_residencia"], 0, 30) / 30
            ) * 0.4
            scores.append(estabilidad_score)
            weights.append(0.20)

        # 4. Score demográfico
        if all(col in df.columns for col in ["edad", "dependientes"]):
            # Edad óptima entre 30-50, penalizar extremos
            edad_score = np.where(
                df["edad"].between(30, 50),
                1.0,
                np.where(df["edad"].between(25, 55), 0.7, 0.4),
            )

            # Menos dependientes = mejor
            dependientes_score = 1 - (df["dependientes"].clip(0, 5) / 5)

            demografico_score = edad_score * 0.6 + dependientes_score * 0.4
            scores.append(demografico_score)
            weights.append(0.20)

        # Calcular score compuesto final
        if scores:
            # Asegurar que todos los arrays tengan la misma longitud
            min_len = min(len(s) for s in scores)
            scores = [s[:min_len] for s in scores]

            # Calcular promedio ponderado
            scores_array = np.array(scores)
            weights_array = np.array(weights).reshape(-1, 1)

            weighted_scores = scores_array * weights_array
            composite_score = weighted_scores.sum(axis=0) / weights_array.sum()

            df["composite_risk_score"] = composite_score
            df["risk_category"] = pd.cut(
                composite_score,
                bins=[0, 0.3, 0.5, 0.7, 1.0],
                labels=["BAJO", "MODERADO", "ALTO", "MUY_ALTO"],
            )

        return df

    def prepare_features_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara features para machine learning.

        Args:
            df: DataFrame con todas las features

        Returns:
            DataFrame listo para ML
        """
        logger.info("Preparando features para ML...")

        # 1. Separar target si existe
        if self.target_col in df.columns:
            y = df[self.target_col]
            X = df.drop(columns=[self.target_col])
        else:
            X = df.copy()

        # 2. Guardar nombres de columnas originales
        original_columns = X.columns.tolist()
        self.feature_metadata["original_columns"] = original_columns

        # 3. Separar variables numéricas y categóricas
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        logger.info(f"  Columnas numéricas: {len(numeric_cols)}")
        logger.info(f"  Columnas categóricas: {len(categorical_cols)}")

        # 4. Procesar variables numéricas
        if numeric_cols:
            # Imputar valores faltantes
            self.imputer = SimpleImputer(strategy="median")
            X[numeric_cols] = self.imputer.fit_transform(X[numeric_cols])

            # Escalar
            self.scaler = StandardScaler()
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])

        # 5. Procesar variables categóricas
        if categorical_cols:
            # Target encoding para categorías con alta cardinalidad
            high_card_cols = [
                col
                for col in categorical_cols
                if X[col].nunique() > 10 and self.target_col in df.columns
            ]
            low_card_cols = [
                col for col in categorical_cols if col not in high_card_cols
            ]

            # Target encoding para alta cardinalidad
            if high_card_cols and self.target_col in df.columns:
                self.encoder = ce.TargetEncoder(cols=high_card_cols)
                X[high_card_cols] = self.encoder.fit_transform(X[high_card_cols], y)

            # One-hot encoding para baja cardinalidad
            if low_card_cols:
                X = pd.get_dummies(X, columns=low_card_cols, drop_first=True)

        # 6. Guardar nombres de features finales
        self.feature_names = X.columns.tolist()
        self.feature_metadata["final_features"] = self.feature_names
        self.feature_metadata["feature_count"] = len(self.feature_names)

        # 7. Añadir target de nuevo si existía
        if self.target_col in df.columns:
            X[self.target_col] = y.values

        logger.info(f"Features preparadas: {len(self.feature_names)} columnas")
        return X

    def get_feature_importance(
        self, df: pd.DataFrame, target_col: str = "default"
    ) -> pd.DataFrame:
        """
        Calcula importancia de features usando correlación.

        Args:
            df: DataFrame con features
            target_col: Columna objetivo

        Returns:
            DataFrame con importancia de features
        """
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' no encontrada")
            return pd.DataFrame()

        # Calcular correlación solo para columnas numéricas
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]

        if not numeric_cols:
            return pd.DataFrame()

        correlations = df[numeric_cols + [target_col]].corr()[target_col].abs()
        correlations = correlations.drop(target_col).sort_values(ascending=False)

        feature_importance = pd.DataFrame(
            {
                "feature": correlations.index,
                "correlation_with_target": correlations.values,
                "abs_correlation": correlations.abs().values,
            }
        )

        return feature_importance

    def save_feature_metadata(self, filepath: str):
        """Guarda metadata de features en un archivo JSON."""
        import json

        metadata = {
            "feature_names": self.feature_names,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "target_column": self.target_col,
            "random_state": self.random_state,
            "timestamp": datetime.now().isoformat(),
            **self.feature_metadata,
        }

        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata de features guardada en: {filepath}")
        return metadata


# Función principal para ejecutar desde línea de comandos
def main():
    """Función principal para feature engineering."""
    import argparse
    import pandas as pd
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Feature engineering para datos crediticios"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Archivo de entrada (CSV o Parquet)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/features/features_engineered.parquet",
        help="Archivo de salida",
    )
    parser.add_argument(
        "--target", type=str, default="default", help="Columna objetivo"
    )

    args = parser.parse_args()

    # Crear directorio de salida si no existe
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    logger.info(f"Cargando datos desde: {args.input}")
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    logger.info(f"Datos cargados: {len(df)} registros, {len(df.columns)} columnas")

    # Feature engineering
    engineer = CreditFeatureEngineer(target_col=args.target)

    # Crear features avanzadas
    df_features = engineer.create_advanced_features(df)
    logger.info(f"Después de feature engineering: {len(df_features.columns)} columnas")

    # Preparar para ML
    df_ml_ready = engineer.prepare_features_for_ml(df_features)
    logger.info(f"Después de preparación ML: {len(df_ml_ready.columns)} columnas")

    # Calcular importancia de features
    if args.target in df_ml_ready.columns:
        importance = engineer.get_feature_importance(df_ml_ready, args.target)
        if not importance.empty:
            print("\n📊 Top 10 features más importantes:")
            print(importance.head(10).to_string())

    # Guardar resultados
    df_ml_ready.to_parquet(args.output, index=False)
    logger.info(f"Features guardadas en: {args.output}")

    # Guardar metadata
    metadata_path = str(Path(args.output).with_suffix(".json"))
    engineer.save_feature_metadata(metadata_path)

    print(f"\n✅ Feature engineering completado exitosamente")
    print(f"   Features generadas: {len(df_ml_ready.columns)}")
    print(f"   Archivo guardado: {args.output}")
    print(f"   Metadata guardada: {metadata_path}")


if __name__ == "__main__":
    main()
