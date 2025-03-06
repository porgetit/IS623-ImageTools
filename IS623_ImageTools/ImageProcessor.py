from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

def _normalizar_datos(datos: np.ndarray) -> np.ndarray:
    """
    Normaliza el arreglo de imagen a rango [0, 1] si es necesario.

    Parámetros:
        datos (np.ndarray): Arreglo de imagen.

    Retorna:
        np.ndarray: Arreglo normalizado.
    """
    return datos.astype(np.float32) / 255.0 if datos.max() > 1 else datos.astype(np.float32)


@dataclass
class Imagen:
    """
    Clase que representa una imagen y encapsula operaciones sobre ella.

    Permite el encadenamiento de métodos para realizar transformaciones de forma fluida.
    """
    datos: np.ndarray

    @classmethod
    def desde_archivo(cls, ruta: str) -> 'Imagen':
        """
        Crea una instancia de Imagen a partir de un archivo.

        Parámetros:
            ruta (str): Ruta al archivo de imagen.

        Retorna:
            Imagen: Instancia de Imagen.
        """
        try:
            img = Image.open(ruta).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error al cargar la imagen desde {ruta}: {e}")
        datos = np.array(img)
        return cls(datos)

    def normalizar(self) -> 'Imagen':
        """
        Normaliza la imagen para que sus valores estén en el rango [0, 1].

        Retorna:
            Imagen: La instancia actual (para encadenamiento).
        """
        self.datos = self.datos / 255.0
        return self

    def desnormalizar(self) -> 'Imagen':
        """
        Desnormaliza la imagen para que sus valores estén en el rango [0, 255].

        Retorna:
            Imagen: La instancia actual (para encadenamiento).
        """
        self.datos = (self.datos * 255).astype(np.uint8)
        return self

    def invertir(self) -> 'Imagen':
        """
        Invierte los colores de la imagen.

        Retorna:
            Imagen: La instancia actual (para encadenamiento).
        """
        self.datos = 1 - self.datos
        return self

    def colorear_pixel(self, row: int, col: int, color: list[int]) -> 'Imagen':
        """
        Colorea un píxel específico de la imagen.

        Parámetros:
            row (int): Índice de la fila.
            col (int): Índice de la columna.
            color (list[int]): Valores de color (debe coincidir con el número de canales).

        Retorna:
            Imagen: La instancia actual (para encadenamiento).

        Raises:
            IndexError: Si los índices están fuera de rango.
            ValueError: Si la longitud de la lista de color es incorrecta.
        """
        if row < 0 or row >= self.datos.shape[0] or col < 0 or col >= self.datos.shape[1]:
            raise IndexError("El índice de píxel está fuera de rango")
        if len(color) != self.datos.shape[2]:
            raise ValueError("La longitud de la lista de color no coincide con el número de canales")
        self.datos[row, col, :] = color
        return self

    def extraer_capa_rgb(self, indice: int) -> 'Imagen':
        """
        Extrae una capa de la imagen en formato RGB.

        Parámetros:
            indice (int): Índice de la capa a extraer (0: R, 1: G, 2: B).

        Retorna:
            Imagen: Nueva imagen con la capa especificada.

        Raises:
            ValueError: Si la imagen no tiene 3 canales o el índice es inválido.
        """
        if self.datos.shape[2] != 3:
            raise ValueError("La imagen debe tener 3 canales")
        if not (0 <= indice <= 2):
            raise ValueError("El índice debe estar entre 0 y 2")
        capa = np.zeros_like(self.datos)
        capa[:, :, indice] = self.datos[:, :, indice]
        return Imagen(capa)

    def extraer_capa_cmyk(self, indice: int) -> 'Imagen':
        """
        Extrae una capa de la imagen simulando el formato CMYK.

        Nota:
            Se asume que la imagen original está en formato RGB.

        Parámetros:
            indice (int): Índice de la capa a extraer (0: cyan, 1: magenta, 2: yellow, 3: black).

        Retorna:
            Imagen: Nueva imagen con la capa especificada.

        Raises:
            ValueError: Si la imagen no tiene 3 canales o el índice es inválido.
        """
        if self.datos.shape[2] != 3:
            raise ValueError("La imagen debe tener 3 canales para RGB")
        if not (0 <= indice <= 3):
            raise ValueError("El índice debe estar entre 0 y 3")
        capa = np.zeros_like(self.datos)
        if indice == 0:  # cyan: conservar G y B
            capa[:, :, 1] = self.datos[:, :, 1]
            capa[:, :, 2] = self.datos[:, :, 2]
        elif indice == 1:  # magenta: conservar R y B
            capa[:, :, 0] = self.datos[:, :, 0]
            capa[:, :, 2] = self.datos[:, :, 2]
        elif indice == 2:  # yellow: conservar R y G
            capa[:, :, 0] = self.datos[:, :, 0]
            capa[:, :, 1] = self.datos[:, :, 1]
        elif indice == 3:  # black: retorna matriz de ceros
            pass
        return Imagen(capa)

    def mean_filter(self, kernel_size: int = 3) -> 'Imagen':
        """
        Aplica un filtro de promedio a la imagen.

        Parámetros:
            kernel_size (int): Tamaño del kernel (debe ser impar).

        Retorna:
            Imagen: La instancia actual (para encadenamiento).

        Raises:
            ValueError: Si el tamaño del kernel es par.
        """
        if kernel_size % 2 == 0:
            raise ValueError("El tamaño del kernel debe ser impar")
        pad_size = kernel_size // 2
        datos_padded = np.pad(self.datos, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="reflect")
        filtered = np.zeros_like(self.datos)
        for i in range(self.datos.shape[0]):
            for j in range(self.datos.shape[1]):
                for k in range(self.datos.shape[2]):
                    filtered[i, j, k] = np.mean(datos_padded[i:i + kernel_size, j:j + kernel_size, k])
        self.datos = filtered
        return self

    def gris_promedio(self) -> 'Imagen':
        """
        Convierte la imagen a escala de grises usando el promedio de los canales.

        Retorna:
            Imagen: Nueva imagen en escala de grises (3 canales).
        """
        gray = np.mean(self.datos, axis=2)
        gray_3ch = np.stack((gray, gray, gray), axis=-1)
        return Imagen(gray_3ch)

    def gris_luminosidad(self) -> 'Imagen':
        """
        Convierte la imagen a escala de grises usando la fórmula de luminosidad.

        Retorna:
            Imagen: Nueva imagen en escala de grises (3 canales).

        Raises:
            ValueError: Si la imagen no tiene al menos 3 canales.
        """
        if self.datos.shape[2] < 3:
            raise ValueError("La imagen debe tener al menos 3 canales")
        R = self.datos[:, :, 0]
        G = self.datos[:, :, 1]
        B = self.datos[:, :, 2]
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        gray_3ch = np.stack((gray, gray, gray), axis=-1)
        return Imagen(gray_3ch)

    def gris_tonalidad(self) -> 'Imagen':
        """
        Convierte la imagen a escala de grises usando el método de tonalidad.

        Retorna:
            Imagen: Nueva imagen en escala de grises (3 canales).

        Raises:
            ValueError: Si la imagen no tiene al menos 3 canales.
        """
        if self.datos.shape[2] < 3:
            raise ValueError("La imagen debe tener al menos 3 canales")
        max_val = np.max(self.datos, axis=2)
        min_val = np.min(self.datos, axis=2)
        gray = (max_val + min_val) / 2.0
        gray_3ch = np.stack((gray, gray, gray), axis=-1)
        return Imagen(gray_3ch)

    def ajustar(self, factor: float) -> 'Imagen':
        """
        Ajusta la imagen aplicando un filtro basado en el factor:
          - factor < 0: Realza el contraste.
          - factor > 0: Realza la intensidad.
          - factor == 0: No se aplica transformación.

        Parámetros:
            factor (float): Valor en el rango [-1, 1].

        Retorna:
            Imagen: La imagen ajustada.
        """
        filtro = FiltroFactory.obtener_filtro(factor)
        return filtro.aplicar(self)

    @staticmethod
    def fusionar(imagenes: list['Imagen']) -> 'Imagen':
        """
        Fusiona varias imágenes sumando sus valores pixel a pixel.

        Parámetros:
            imagenes (list[Imagen]): Lista de imágenes a fusionar.

        Retorna:
            Imagen: Imagen resultante de la fusión.

        Raises:
            ValueError: Si las imágenes tienen tamaños diferentes.
        """
        if len({img.datos.shape[:2] for img in imagenes}) != 1:
            raise ValueError("Las imágenes deben tener el mismo tamaño (filas y columnas)")
        datos_fusion = np.zeros_like(imagenes[0].datos)
        for img in imagenes:
            datos_fusion += img.datos
        return Imagen(datos_fusion)

    @staticmethod
    def fusionar_ecualizado(imagenes: list[tuple['Imagen', int]]) -> 'Imagen':
        """
        Fusiona imágenes aplicando un factor de ecualización a cada una.

        Parámetros:
            imagenes (list[tuple[Imagen, int]]): Lista de tuplas (Imagen, factor).

        Retorna:
            Imagen: Imagen resultante de la fusión.

        Raises:
            ValueError: Si las imágenes tienen tamaños diferentes.
        """
        if len({img.datos.shape[:2] for img, _ in imagenes}) != 1:
            raise ValueError("Las imágenes deben tener el mismo tamaño (filas y columnas)")
        datos_fusion = np.zeros_like(imagenes[0][0].datos)
        for img, factor in imagenes:
            datos_fusion += img.datos * factor
        return Imagen(datos_fusion)


class ColorConverter:
    """
    Clase para conversiones entre espacios de color.
    """
    @staticmethod
    def rgb_a_cmyk(imagen: Imagen) -> Imagen:
        """
        Convierte una imagen de RGB a CMYK.

        Parámetros:
            imagen (Imagen): Imagen en formato RGB.

        Retorna:
            Imagen: Imagen en formato CMYK con valores en el rango [0, 1].

        Raises:
            ValueError: Si la imagen no tiene 3 canales.
        """
        datos = imagen.datos
        if datos.shape[-1] != 3:
            raise ValueError("La imagen debe tener 3 canales (RGB)")
        datos_norm = _normalizar_datos(datos)
        R = datos_norm[..., 0]
        G = datos_norm[..., 1]
        B = datos_norm[..., 2]
        K = 1 - np.max(datos_norm, axis=-1)
        C = np.where(K == 1, 0, (1 - R - K) / (1 - K))
        M = np.where(K == 1, 0, (1 - G - K) / (1 - K))
        Y = np.where(K == 1, 0, (1 - B - K) / (1 - K))
        cmyk = np.stack((C, M, Y, K), axis=-1)
        return Imagen(cmyk)

    @staticmethod
    def cmyk_a_rgb(imagen: Imagen) -> Imagen:
        """
        Convierte una imagen de CMYK a RGB.

        Parámetros:
            imagen (Imagen): Imagen en formato CMYK.

        Retorna:
            Imagen: Imagen en formato RGB con valores en el rango [0, 1].

        Raises:
            ValueError: Si la imagen no tiene 4 canales.
        """
        datos = imagen.datos
        if datos.shape[-1] != 4:
            raise ValueError("La imagen debe tener 4 canales (CMYK)")
        C = datos[..., 0]
        M = datos[..., 1]
        Y = datos[..., 2]
        K = datos[..., 3]
        R = (1 - C) * (1 - K)
        G = (1 - M) * (1 - K)
        B = (1 - Y) * (1 - K)
        rgb = np.stack((R, G, B), axis=-1)
        return Imagen(rgb)


class FiltroStrategy(ABC):
    """
    Clase abstracta para estrategias de filtrado.
    """
    @abstractmethod
    def aplicar(self, imagen: Imagen) -> Imagen:
        """
        Aplica el filtro a la imagen.

        Parámetros:
            imagen (Imagen): Imagen a la que aplicar el filtro.

        Retorna:
            Imagen: Imagen filtrada.
        """
        pass


class FiltroIdentity(FiltroStrategy):
    """
    Filtro que no aplica cambios a la imagen.
    """
    def aplicar(self, imagen: Imagen) -> Imagen:
        """
        Retorna una copia de la imagen sin modificaciones.
        """
        return Imagen(imagen.datos.copy())


class FiltroContraste(FiltroStrategy):
    """
    Filtro para realzar el contraste utilizando transformación logarítmica.

    Se espera que el factor sea negativo.
    """
    def __init__(self, factor: float) -> None:
        if factor >= 0:
            raise ValueError("El factor de contraste debe ser negativo")
        self.factor = abs(factor)

    def aplicar(self, imagen: Imagen) -> Imagen:
        """
        Aplica la transformación de contraste a la imagen.

        Retorna:
            Imagen: Imagen con contraste ajustado.
        """
        datos_norm = _normalizar_datos(imagen.datos)
        c = 1.0 / np.log10(2.0)
        log_img = c * np.log10(1.0 + datos_norm)
        resultado = (1 - self.factor) * datos_norm + self.factor * log_img
        imagen.datos = resultado
        return imagen


class FiltroIntensidad(FiltroStrategy):
    """
    Filtro para realzar la intensidad utilizando transformación exponencial.

    Se espera que el factor sea positivo.
    """
    def __init__(self, factor: float) -> None:
        if factor <= 0:
            raise ValueError("El factor de intensidad debe ser positivo")
        self.factor = factor

    def aplicar(self, imagen: Imagen) -> Imagen:
        """
        Aplica la transformación de intensidad a la imagen.

        Retorna:
            Imagen: Imagen con intensidad ajustada.
        """
        datos_norm = _normalizar_datos(imagen.datos)
        exp_img = (np.exp(datos_norm) - 1.0) / (np.e - 1.0)
        resultado = (1 - self.factor) * datos_norm + self.factor * exp_img
        imagen.datos = resultado
        return imagen


class FiltroFactory:
    """
    Fábrica para obtener estrategias de filtrado según un factor.

    Permite seleccionar la estrategia adecuada (contraste, intensidad o identidad).
    """
    @staticmethod
    def obtener_filtro(factor: float) -> FiltroStrategy:
        """
        Retorna la estrategia de filtrado adecuada basada en el factor.

        Parámetros:
            factor (float): Factor de ajuste en el rango [-1, 1].

        Retorna:
            FiltroStrategy: Estrategia de filtrado correspondiente.

        Raises:
            ValueError: Si no se reconoce una estrategia para el factor dado.
        """
        if factor == 0:
            return FiltroIdentity()
        elif factor < 0:
            return FiltroContraste(factor)
        elif factor > 0:
            return FiltroIntensidad(factor)
        else:
            raise ValueError("Factor no reconocido para el filtrado")
