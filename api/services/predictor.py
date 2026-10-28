"""
Módulo de Serviço Preditivo (Camada de Negócio e Inferência ML).
Atende às Fases 2 e 3: Desacoplamento de HTTP e execução estritamente em memória RAM.
"""

import os
import time
from typing import Dict, Tuple, Any
import numpy as np
import tensorflow as tf


class MNISTPredictorService:
    """
    Serviço responsável pelo carregamento e execução da inferência da Rede Neural MLP.
    O modelo reside em memória RAM após a inicialização pelo ciclo de vida (Lifespan).
    """

    def __init__(self, model_path: str = "models/modelo_mlp_mnist.keras"):
        self.model_path = model_path
        self.model: Any = None
        self.model_version: str = "MLP-Keras-v1.0"
        self.device: str = "CPU"

    def load_model(self) -> None:
        """
        Carrega o artefato serializado do disco para a memória RAM.
        Esta operação deve ser chamada EXCLUSIVAMENTE uma única vez no startup da API.
        """
        if not os.path.exists(self.model_path):
            # Fallback caso o comando uvicorn seja executado dentro da pasta api/
            caminho_alternativo = os.path.join(
                os.path.dirname(__file__), "..", "..", "models", "modelo_mlp_mnist.keras"
            )
            if os.path.exists(caminho_alternativo):
                self.model_path = os.path.abspath(caminho_alternativo)
            else:
                raise FileNotFoundError(
                    f"Artefato serializado do modelo não foi encontrado em: {self.model_path}"
                )

        # Carrega o modelo Keras para a memória RAM
        self.model = tf.keras.models.load_model(self.model_path)

        # Detecta se há suporte a GPU/CUDA ou se operará em CPU
        gpus = tf.config.list_physical_devices("GPU")
        self.device = "CUDA" if len(gpus) > 0 else "CPU"

    def is_ready(self) -> bool:
        """Verifica se o modelo está carregado e pronto para receber inferências."""
        return self.model is not None

    def predict(self, features: list[float]) -> Tuple[int, float, Dict[str, float], float]:
        """
        Executa a inferência preditiva a partir do vetor de 784 intensidades de pixels.

        Retorna:
            - classe_predita (int de 0 a 9)
            - confianca (float de 0.0 a 1.0)
            - probabilidades (dicionário com as 10 classes)
            - tempo_inferencia_ms (float em milissegundos)
        """
        if not self.is_ready():
            raise RuntimeError("O modelo preditivo ainda não foi carregado na memória RAM.")

        # 1. Converte a lista em array numpy float32
        dados_np = np.array(features, dtype=np.float32)

        # 2. Normalização defensiva: se os pixels vierem em escala [0, 255], normaliza para [0.0, 1.0]
        if np.max(dados_np) > 1.0:
            dados_np = dados_np / 255.0

        # 3. Formata para batch de tamanho 1: shape (1, 784)
        tensor_entrada = dados_np.reshape(1, 784)

        # 4. Medição precisa do tempo de inferência
        inicio = time.perf_counter()
        predicoes_raw = self.model.predict(tensor_entrada, verbose=0)
        fim = time.perf_counter()
        tempo_ms = round((fim - inicio) * 1000, 2)

        # 5. Extração das probabilidades da camada Softmax
        vetor_probabilidades = predicoes_raw[0]
        classe_predita = int(np.argmax(vetor_probabilidades))
        confianca = float(vetor_probabilidades[classe_predita])

        # 6. Mapeamento das 10 classes (dígitos 0 a 9)
        distribuicao_probs = {
            str(digito): round(float(vetor_probabilidades[digito]), 4)
            for digito in range(10)
        }

        return classe_predita, round(confianca, 4), distribuicao_probs, tempo_ms
    