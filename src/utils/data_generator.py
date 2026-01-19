"""
Generador de datos sintéticos para análisis de riesgo crediticio.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import json
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GeneradorDatosCrediticios:
    """Genera datos sintéticos realistas para análisis crediticio."""

    def __init__(self, n_clientes: int = 1000, seed: int = 42):
        """
        Inicializa el generador.

        Args:
            n_clientes: Número de clientes a generar
            seed: Semilla para reproducibilidad
        """
        self.n_clientes = n_clientes
        self.seed = seed
        self.faker = Faker()

        # Configurar semillas para reproducibilidad
        np.random.seed(seed)
        random.seed(seed)
        self.faker.seed_instance(seed)

        logger.info(f"Inicializando generador para {n_clientes} clientes")

    def generar_datos_demograficos(self) -> pd.DataFrame:
        """Genera información demográfica de clientes."""
        logger.info("Generando datos demográficos...")

        clientes = []
        for i in range(self.n_clientes):
            edad = random.randint(18, 70)

            cliente = {
                "cliente_id": f"CLI{i:06d}",
                "nombre": self.faker.name(),
                "edad": edad,
                "genero": random.choice(["M", "F"]),
                "estado_civil": random.choice(
                    ["soltero", "casado", "divorciado", "viudo"]
                ),
                "dependientes": random.randint(0, min(5, max(0, (edad - 20) // 10))),
                "nivel_educacion": self._asignar_educacion(edad),
                "tipo_vivienda": random.choice(
                    ["propia", "arrendada", "familiar", "hipotecada"]
                ),
                "ciudad": self.faker.city(),
                "antiguedad_residencia": random.randint(0, min(edad - 18, 30)),
                "telefono": self.faker.phone_number(),
                "email": self.faker.email(),
                "fecha_registro": self.faker.date_between(
                    start_date="-10y", end_date="today"
                ),
            }
            clientes.append(cliente)

        df = pd.DataFrame(clientes)
        logger.info(f"Datos demográficos generados: {len(df)} registros")
        return df

    def _asignar_educacion(self, edad: int) -> str:
        """Asigna nivel educativo basado en edad (lógica realista)."""
        if edad < 22:
            return random.choice(["basica", "media"])
        elif edad < 30:
            return random.choice(["media", "universitaria", "tecnica"])
        else:
            return random.choice(["universitaria", "postgrado", "tecnica", "media"])

    def generar_datos_financieros(self) -> pd.DataFrame:
        """Genera información financiera de clientes."""
        logger.info("Generando datos financieros...")

        datos_financieros = []
        for i in range(self.n_clientes):
            edad = random.randint(25, 65)
            tiene_empleo = random.random() > 0.1  # 90% tiene empleo

            if tiene_empleo:
                # Ingreso basado en edad y educación
                ingreso_base = 1000 + (edad - 25) * 30
                variabilidad = np.random.normal(0, 0.25)
                ingreso_mensual = max(600, ingreso_base * (1 + variabilidad))
                antiguedad_empleo = random.randint(0, min(edad - 18, 40))
            else:
                ingreso_mensual = random.uniform(400, 800)
                antiguedad_empleo = 0

            datos = {
                "cliente_id": f"CLI{i:06d}",
                "ingreso_mensual": round(ingreso_mensual, 2),
                "ingreso_anual": round(ingreso_mensual * 12, 2),
                "fuente_ingreso": self._asignar_fuente_ingreso(tiene_empleo),
                "antiguedad_empleo": antiguedad_empleo,  # en meses
                "tipo_contrato": self._asignar_tipo_contrato(tiene_empleo),
                "empresa_tamano": (
                    random.choice(["pequena", "mediana", "grande"])
                    if tiene_empleo
                    else "N/A"
                ),
                "otros_ingresos": round(random.uniform(0, 500), 2),
                "gastos_mensuales": round(
                    ingreso_mensual * random.uniform(0.3, 0.7), 2
                ),
                "ahorros": round(random.uniform(0, ingreso_mensual * 12), 2),
                "score_bancario": random.randint(300, 850),
            }
            datos_financieros.append(datos)

        df = pd.DataFrame(datos_financieros)
        logger.info(f"Datos financieros generados: {len(df)} registros")
        return df

    def _asignar_fuente_ingreso(self, tiene_empleo: bool) -> str:
        """Asigna fuente de ingreso."""
        if tiene_empleo:
            return random.choice(["empleo", "profesional_independiente", "consultor"])
        else:
            return random.choice(
                ["pension", "subsidio", "negocio_propio", "inversiones"]
            )

    def _asignar_tipo_contrato(self, tiene_empleo: bool) -> str:
        """Asigna tipo de contrato."""
        if tiene_empleo:
            return random.choice(["indefinido", "temporal", "obra", "eventual"])
        else:
            return "sin_contrato"

    def generar_historial_crediticio(self) -> pd.DataFrame:
        """Genera historial de créditos anteriores."""
        logger.info("Generando historial crediticio...")

        historial = []
        for i in range(self.n_clientes):
            cliente_id = f"CLI{i:06d}"

            # Número de créditos previos (distribución realista)
            n_creditos = np.random.poisson(2.5)  # Promedio 2.5 créditos

            for j in range(n_creditos):
                # Determinar si el crédito fue problemático
                es_problematico = random.random() < 0.15  # 15% son problemáticos

                monto = random.choice([100, 500, 1000, 5000, 10000, 20000])
                plazo = random.choice([3, 6, 12, 24, 36, 48, 60])

                # Fechas realistas
                fecha_inicio = self.faker.date_between(start_date="-5y", end_date="-6m")
                fecha_fin = fecha_inicio + timedelta(days=plazo * 30)

                # Determinar estado basado en si es problemático
                if es_problematico:
                    if random.random() < 0.3:
                        estado = "incumplido"
                        dias_mora = random.randint(90, 365)
                    else:
                        estado = "moroso"
                        dias_mora = random.randint(30, 89)
                else:
                    estado = "pagado"
                    dias_mora = 0 if random.random() > 0.1 else random.randint(1, 29)

                historial.append(
                    {
                        "cliente_id": cliente_id,
                        "credito_id": f"CRD{i:06d}-{j:03d}",
                        "monto": monto,
                        "plazo_meses": plazo,
                        "tasa_interes": round(random.uniform(0.08, 0.35), 4),
                        "fecha_inicio": fecha_inicio,
                        "fecha_fin": (
                            fecha_fin if fecha_fin < datetime.now().date() else None
                        ),
                        "estado": estado,
                        "dias_mora": dias_mora,
                        "tipo_credito": random.choice(
                            ["personal", "automotriz", "hipotecario", "tarjeta"]
                        ),
                        "institucion": random.choice(
                            ["BancoA", "BancoB", "FinancieraC", "CooperativaD"]
                        ),
                    }
                )

        df = pd.DataFrame(historial)
        logger.info(f"Historial crediticio generado: {len(df)} registros")
        return df

    def generar_target_variable(
        self, df_demografico: pd.DataFrame, df_financiero: pd.DataFrame
    ) -> pd.DataFrame:
        """Genera la variable objetivo (default) con lógica realista."""
        logger.info("Generando variable objetivo...")

        targets = []
        for i in range(self.n_clientes):
            cliente_id = f"CLI{i:06d}"

            # Obtener datos del cliente
            demografico = df_demografico[
                df_demografico["cliente_id"] == cliente_id
            ].iloc[0]
            financiero = df_financiero[df_financiero["cliente_id"] == cliente_id].iloc[
                0
            ]

            # Calcular score de riesgo (0-1)
            score_riesgo = self._calcular_score_riesgo(demografico, financiero)

            # Determinar default (1 = default, 0 = no default)
            # Umbral ajustable
            default = 1 if score_riesgo > 0.65 else 0

            # Añadir algo de ruido (5% de casos contrarios)
            if random.random() < 0.05:
                default = 1 - default

            targets.append(
                {
                    "cliente_id": cliente_id,
                    "default": default,
                    "score_riesgo": round(score_riesgo, 4),
                    "probabilidad_default": round(score_riesgo * 100, 2),
                    "categoria_riesgo": self._categorizar_riesgo(score_riesgo),
                }
            )

        df = pd.DataFrame(targets)
        logger.info(f"Target generado: {df['default'].mean()*100:.1f}% de default")
        return df

    def _calcular_score_riesgo(
        self, demografico: pd.Series, financiero: pd.Series
    ) -> float:
        """Calcula score de riesgo combinando múltiples factores."""
        factores = []

        # 1. Edad (edad extremas = más riesgo)
        if demografico["edad"] < 25 or demografico["edad"] > 65:
            factores.append(0.7)
        else:
            factores.append(0.3)

        # 2. Ingreso (bajo ingreso = más riesgo)
        ingreso_riesgo = min(1.0, max(0, 1 - (financiero["ingreso_mensual"] / 3000)))
        factores.append(ingreso_riesgo)

        # 3. Dependientes (más dependientes = más riesgo)
        dep_riesgo = min(1.0, demografico["dependientes"] / 5)
        factores.append(dep_riesgo * 0.5)

        # 4. Estabilidad laboral
        estabilidad = 1.0 if financiero["antiguedad_empleo"] < 6 else 0.2
        factores.append(estabilidad * 0.8)

        # 5. Score bancario
        score_riesgo = 1 - (financiero["score_bancario"] - 300) / 550
        factores.append(score_riesgo * 0.7)

        # 6. Tipo de vivienda
        vivienda_riesgo = {
            "propia": 0.2,
            "hipotecada": 0.5,
            "arrendada": 0.7,
            "familiar": 0.9,
        }.get(demografico["tipo_vivienda"], 0.5)
        factores.append(vivienda_riesgo)

        # Promedio ponderado
        pesos = [0.15, 0.25, 0.10, 0.20, 0.20, 0.10]
        score = sum(f * p for f, p in zip(factores, pesos))

        # Añadir algo de aleatoriedad
        score = min(1.0, max(0.0, score + np.random.normal(0, 0.1)))

        return score

    def _categorizar_riesgo(self, score: float) -> str:
        """Categoriza el riesgo basado en el score."""
        if score < 0.3:
            return "BAJO"
        elif score < 0.5:
            return "MODERADO"
        elif score < 0.7:
            return "ALTO"
        else:
            return "MUY_ALTO"

    def generar_todos_datos(self, output_dir: str = "data/raw"):
        """Genera todos los datos y los guarda en archivos."""
        logger.info("Iniciando generación de todos los datos...")

        # Crear directorio si no existe
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generar datos
        df_demografico = self.generar_datos_demograficos()
        df_financiero = self.generar_datos_financieros()
        df_historial = self.generar_historial_crediticio()
        df_target = self.generar_target_variable(df_demografico, df_financiero)

        # Guardar datos
        fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        df_demografico.to_csv(
            f"{output_dir}/clientes_demograficos_{fecha_str}.csv", index=False
        )
        df_financiero.to_csv(
            f"{output_dir}/clientes_financieros_{fecha_str}.csv", index=False
        )
        df_historial.to_csv(
            f"{output_dir}/historial_crediticio_{fecha_str}.csv", index=False
        )
        df_target.to_csv(f"{output_dir}/target_{fecha_str}.csv", index=False)

        # Generar metadata
        metadata = {
            "fecha_generacion": fecha_str,
            "n_clientes": self.n_clientes,
            "n_creditos": len(df_historial),
            "seed": self.seed,
            "default_rate": float(df_target["default"].mean()),
            "archivos": [
                f"clientes_demograficos_{fecha_str}.csv",
                f"clientes_financieros_{fecha_str}.csv",
                f"historial_crediticio_{fecha_str}.csv",
                f"target_{fecha_str}.csv",
            ],
        }

        with open(f"{output_dir}/metadata_{fecha_str}.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Datos generados y guardados en {output_dir}")
        logger.info(f"Metadata: {json.dumps(metadata, indent=2)}")

        return {
            "demografico": df_demografico,
            "financiero": df_financiero,
            "historial": df_historial,
            "target": df_target,
            "metadata": metadata,
        }


# Función principal para ejecutar desde línea de comandos
def main():
    """Función principal para generar datos."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generador de datos sintéticos para riesgo crediticio"
    )
    parser.add_argument(
        "--clientes", type=int, default=1000, help="Número de clientes a generar"
    )
    parser.add_argument(
        "--output", type=str, default="data/raw", help="Directorio de salida"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Semilla para reproducibilidad"
    )

    args = parser.parse_args()

    # Generar datos
    generador = GeneradorDatosCrediticios(n_clientes=args.clientes, seed=args.seed)
    generador.generar_todos_datos(output_dir=args.output)

    print(f"✅ Datos generados exitosamente en {args.output}")


if __name__ == "__main__":
    main()
