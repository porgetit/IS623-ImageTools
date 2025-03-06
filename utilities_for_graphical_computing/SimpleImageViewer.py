import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Union, Any

class SimpleImageViewer:
    """
    Clase sencilla para mostrar múltiples imágenes en una o varias figuras.
    
    Las imágenes se muestran en una cuadrícula fija de 2x2 por figura, lo que asegura un tamaño
    estándar y suficiente separación entre ellas y sus títulos. Se acepta tanto arreglos NumPy
    como instancias de la clase Imagen (o similares que tengan el atributo 'datos').
    """
    
    def __init__(self, images_dict: Dict[str, Union[np.ndarray, Any]]) -> None:
        """
        Inicializa la clase con un diccionario de imágenes.
        
        Parámetros:
            images_dict (Dict[str, Union[np.ndarray, Imagen]]): Diccionario en el que las claves
            son títulos y los valores son imágenes.
        """
        self.images_dict = images_dict
    
    @staticmethod
    def _get_image_array(image: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Extrae el arreglo de imagen de un objeto.
        
        Si el objeto tiene el atributo 'datos', se asume que es una instancia de la clase Imagen
        y se retorna dicho atributo; de lo contrario, se asume que ya es un arreglo NumPy.
        
        Parámetros:
            image (Union[np.ndarray, Imagen]): Imagen a procesar.
        
        Retorna:
            np.ndarray: Arreglo de la imagen.
        """
        if hasattr(image, 'datos'):
            return image.datos
        return image
    
    def show(self, images_per_figure: int = 4, scale: float = 5) -> None:
        """
        Muestra todas las imágenes en figuras con una cuadrícula fija de 2 filas x 2 columnas.
        
        Esto garantiza que cada imagen se muestre en un tamaño estándar y con suficiente separación,
        evitando solapamientos entre imágenes y títulos.
        
        Parámetros:
            images_per_figure (int): Número máximo de imágenes por figura (por defecto, 4).
            scale (float): Factor de escala para definir el tamaño de la figura.
        """
        images_list = list(self.images_dict.items())
        n_images = len(images_list)
        
        # Se generan figuras de 2x2 para cada grupo de imágenes.
        for i in range(0, n_images, images_per_figure):
            fig, axes = plt.subplots(2, 2, figsize=(scale * 2, scale * 2))
            fig.subplots_adjust(hspace=0.5, wspace=0.5)
            axes = axes.flatten()
            
            current_items = images_list[i:i + images_per_figure]
            for ax, (title, image) in zip(axes, current_items):
                image_array = self._get_image_array(image)
                if image_array.ndim == 2:
                    ax.imshow(image_array, cmap='gray')
                else:
                    ax.imshow(image_array)
                ax.set_title(title)
                ax.axis('off')
            
            # Ocultar los ejes no utilizados en la cuadrícula.
            for ax in axes[len(current_items):]:
                ax.axis('off')
        
        plt.show()
