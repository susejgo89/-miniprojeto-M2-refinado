"""
Módulo de Esquemas Pydantic para Inferência Preditiva (MNIST).
Atende aos requisitos da Fase 4: Tipagem estrita, restrições com Field e validação contra HTTP 422.
"""

import json
import os
from typing import Dict, List
from pydantic import BaseModel, Field, field_validator


def _carregar_exemplo_digito_7() -> List[float]:
    """Carrega o vetor real de 784 pixels do dígito 7 para o exemplo interativo no Swagger (/docs)."""
    caminho = os.path.join(os.path.dirname(__file__), "..", "..", "data", "exemplo_digito_7.json")
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)
            return [round(float(x) / 255.0, 4) for x in dados["features"]]
    except Exception:
        return [0.0] * 784


class MNISTInputSchema(BaseModel):
    """
    Contrato de dados de entrada para o endpoint de predição.
    Exige exatamente 784 valores numéricos (pixels da imagem 28x28 normalizados ou na escala 0-255).
    """
    features: List[float] = Field(
        ...,
        min_length=784,
        max_length=784,
        description="Lista contendo exatamente 784 intensidades de pixels correspondentes à imagem 28x28 achatada (valores entre 0.0 e 255.0).",
        json_schema_extra={
            "example": _carregar_exemplo_digito_7()  # Exemplo real do dígito 7 pronto para testar no Swagger
        }
    )

    @field_validator("features")
    @classmethod
    def validar_dimensao_e_valores(cls, v: List[float]) -> List[float]:
        # 1. Validação estrita da dimensão geométrica da imagem
        if len(v) != 784:
            raise ValueError(
                f"Dimensão inválida: o vetor de entrada deve conter exatamente 784 pixels (recebeu {len(v)})."
            )

        # 2. Validação estrita dos limites de intensidade de cada pixel
        # Aceita escala normalizada [0.0, 1.0] ou bruta inteira [0, 255]
        for idx, pixel in enumerate(v):
            if pixel < 0.0 or pixel > 255.0:
                raise ValueError(
                    f"Pixel no índice {idx} com valor fora dos limites permitidos: {pixel}. "
                    "Valores aceitos devem estar entre 0.0 e 255.0."
                )

        return v


class MNISTOutputSchema(BaseModel):
    """
    Contrato de dados de resposta do endpoint /predict.
    Garante tipagem estrita na devolução do resultado para o cliente.
    """
    predicted_class: int = Field(
        ...,
        ge=0,
        le=9,
        description="Dígito inferido pelo modelo campeão (inteiro de 0 a 9)."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade calibrada da classe vencedora calculada pela função Softmax."
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Vetor completo com as probabilidades atribuídas para cada uma das 10 classes (0 a 9)."
    )
    inference_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Tempo total gasto exclusivamente na inferência matemática (em milissegundos)."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_class": 7,
                "confidence": 0.9984,
                "probabilities": {
                    "0": 0.0000, "1": 0.0001, "2": 0.0002, "3": 0.0001, "4": 0.0003,
                    "5": 0.0000, "6": 0.0000, "7": 0.9984, "8": 0.0001, "9": 0.0008
                },
                "inference_time_ms": 1.45
            }
        }
    }