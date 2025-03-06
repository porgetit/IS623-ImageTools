# IS623-ImageTools

**IS623-ImageTools** es una librería sencilla para el procesamiento y la visualización de imágenes, desarrollada en el marco del curso **IS623 - Computación Gráfica** impartido por el Profesor Francisco Alejandro Medina Aguirre en la Universidad Tecnológica de Pereira (UTP), Colombia. La librería fue creada por **Kevin Esguerra Cardona** con el objetivo de facilitar el aprendizaje de conceptos básicos del procesamiento digital de imágenes, el paradigma de la programación orientada a objetos y el manejo de paquetes en Python según los estándares PEP 609.

---

## Características

- **Procesamiento de Imágenes:**  
  Permite cargar imágenes en formato JPG, normalizarlas, desnormalizarlas, invertir colores, aplicar filtros (por ejemplo, filtro de promedio) y ajustar propiedades como contraste e intensidad mediante una API encadenable.

- **Conversión de Formatos y Operaciones de Color:**  
  Facilita la conversión entre espacios de color, como de RGB a CMYK y viceversa, y permite la extracción de capas específicas en ambos formatos.

- **Filtros Basados en Patrones de Diseño:**  
  Incorpora el uso de patrones *Strategy* y *Factory* para seleccionar dinámicamente la transformación adecuada (por ejemplo, para realzar contraste o intensidad) en función de un factor de ajuste.

- **Visualización de Imágenes:**  
  La clase `SimpleImageViewer` organiza y muestra múltiples imágenes en una cuadrícula fija (2x2) en figuras de tamaño estándar, garantizando que las imágenes y sus títulos se presenten sin solapamientos.

---

## Requisitos

- **Python:** 3.8 o superior  
- **Librerías externas:**  
  - NumPy  
  - Pillow  
  - Matplotlib

> Nota: Solo se listan las dependencias externas a la distribución estándar de Python.

---

<!-- ## Instalación

1. **Clonar el repositorio:**

   ```bash
   git clone https://ruta-del-repositorio/IS623-ImageTools.git
   cd IS623-ImageTools
   ```

2. **Instalar la librería y sus dependencias:**
  
> Nota: La librería aún no se ha construido, por lo que la importación o alocación debe realizarse en una ruta conocida.

   - **Instalación normal:**

     ```bash
     pip install .
     ```

   - **Instalación en modo desarrollo:**

     ```bash
     pip install -e .
     ```

--- -->

## Uso

### Procesamiento de Imágenes

La clase principal para el procesamiento es `Imagen`. Esta clase encapsula el estado de la imagen (almacenado en un arreglo NumPy) y permite encadenar métodos para aplicar diversas transformaciones. Por ejemplo:

```python
from utilities_for_graphical_computing import Imagen, ColorConverter

# Cargar la imagen y normalizarla (valores en el rango [0,1])
imagen = Imagen.desde_archivo("ruta/a/la/imagen.jpg").normalizar()

# Invertir los colores de la imagen
imagen_invertida = Imagen(imagen.datos.copy()).invertir()

# Extraer la capa roja (índice 0) de la imagen RGB
imagen_roja = Imagen(imagen.datos.copy()).extraer_capa_rgb(0)

# Convertir la imagen de RGB a CMYK
imagen_cmyk = ColorConverter.rgb_a_cmyk(imagen)
```

### Visualización de Imágenes

La clase `SimpleImageViewer` permite mostrar múltiples imágenes en una figura organizada en una cuadrícula fija de 2x2. Se aceptan tanto arreglos NumPy como instancias de `Imagen` (o similares) que dispongan del atributo `datos`.

```python
from utilities_for_graphical_computing import SimpleImageViewer

imagenes = {
    "Original": imagen.datos,
    "Invertida": imagen_invertida.datos,
    "Capa Roja": imagen_roja.datos,
    "RGB a CMYK": imagen_cmyk.datos
}

visor = SimpleImageViewer(imagenes)
visor.show()
```

### Ejemplo Completo

El siguiente fragmento de código, basado en el archivo `main.py`, muestra un ejemplo de implementación en el que se crean imágenes personalizadas, se aplican transformaciones y se visualizan los resultados:

```python
from utilities_for_graphical_computing import Imagen, SimpleImageViewer
import numpy as np

def colorear_region(imagen: Imagen, rows, cols, color: list[int]) -> Imagen:
    if isinstance(rows, slice):
        rows = list(range(*rows.indices(imagen.datos.shape[0])))
    elif isinstance(rows, int):
        rows = [rows]
    if isinstance(cols, slice):
        cols = list(range(*cols.indices(imagen.datos.shape[1])))
    elif isinstance(cols, int):
        cols = [cols]
    for r in rows:
        for c in cols:
            imagen.colorear_pixel(r, c, color)
    return imagen

# Ejemplo: Crear una matriz personalizada 3x3
def matriz3x3Personalizada():
    imagen = Imagen(np.zeros((3, 3, 3), dtype=float))
    imagen.colorear_pixel(0, 0, [0, 1, 1])    # cyan
    imagen.colorear_pixel(0, 1, [1, 1, 1])    # blanco
    imagen.colorear_pixel(0, 2, [1, 0, 0])    # rojo
    imagen.colorear_pixel(1, 0, [1, 0, 1])    # magenta
    imagen.colorear_pixel(1, 1, [0.5, 0.5, 0.5])  # gris
    imagen.colorear_pixel(1, 2, [0, 1, 0])    # verde
    imagen.colorear_pixel(2, 0, [1, 1, 0])    # amarillo
    imagen.colorear_pixel(2, 2, [0, 0, 1])    # azul
    return imagen.datos

if __name__ == "__main__":
    matriz3x3 = matriz3x3Personalizada()

    # Cargar y procesar la imagen "paris.jpg"
    paris = Imagen.desde_archivo("paris.jpg").normalizar()
    paris_invertida = Imagen(paris.datos.copy()).invertir().datos

    images = {
        "Matriz 3x3 personalizada": matriz3x3,
        "Paris - Invertida": paris_invertida,
        # ... otros ejemplos de transformación
    }

    viewer = SimpleImageViewer(images)
    viewer.show()
```

---

## Documentación de la API

> Nota: En este documento se habla de la API (Interfaz de Programación de Aplicaciones) porque describe el conjunto de clases, métodos y funciones que la librería pone a disposición de los usuarios para interactuar con ella. Es decir, la API es la "puerta de entrada" que permite utilizar las funcionalidades de IS623-ImageTools en tus propios proyectos.

### Clase `Imagen`
La clase `Imagen` es el núcleo del procesamiento de imágenes en la librería. Esta clase encapsula un arreglo NumPy que representa la imagen y proporciona un conjunto de métodos para transformar, manipular y analizar la imagen de forma encadenable. A continuación se detalla el funcionamiento de cada uno de sus métodos:

- **`Imagen.desde_archivo(ruta: str) -> Imagen`**  
  
  Este método de clase carga una imagen a partir de la ruta especificada, utilizando la librería Pillow para abrir el archivo y convertir la imagen a formato RGB. Luego, transforma la imagen en un arreglo NumPy y devuelve una nueva instancia de `Imagen` con estos datos.

- **`normalizar() -> Imagen`**  
  
  Este método transforma la imagen de modo que todos sus valores de píxel se escalen al rango [0, 1]. Esto se logra dividiendo el arreglo de la imagen por 255. Es útil para realizar operaciones de procesamiento que requieren trabajar con valores flotantes normalizados. Retorna la misma instancia, permitiendo el encadenamiento de métodos.

- **`desnormalizar() -> Imagen`**  
  
  Realiza la operación inversa a `normalizar()`: multiplica el arreglo normalizado por 255 y convierte los valores resultantes a enteros sin signo (uint8), volviendo a la escala de 0 a 255. Esto es útil para visualizar la imagen o guardarla en formatos que requieren esta escala.

- **`invertir() -> Imagen`**  
  
  Invierte los colores de la imagen asumiendo que está en el rango [0, 1]. Cada valor se transforma en su complemento (1 - valor), lo que resulta en una imagen con colores invertidos. Este método modifica la imagen en sitio y retorna la instancia para permitir el encadenamiento.

- **`colorear_pixel(row: int, col: int, color: list[int]) -> Imagen`**  
  
  Permite modificar el color de un píxel específico de la imagen. Se requiere indicar la posición del píxel (fila y columna) y proporcionar una lista con los valores de color, cuya longitud debe coincidir con el número de canales de la imagen (por ejemplo, 3 para imágenes RGB). El método valida que los índices estén dentro del rango de la imagen y que el tamaño del color sea correcto, luego actualiza el valor del píxel y retorna la instancia actual.

- **`extraer_capa_rgb(indice: int) -> Imagen`**
  
  Extrae una capa específica de la imagen en formato RGB. Se espera que el índice sea 0, 1 o 2, correspondientes a los canales R, G y B, respectivamente. El método crea un nuevo arreglo donde sólo se conserva la capa especificada y las demás se ponen a cero, devolviendo una nueva instancia de `Imagen` con este arreglo.

- **`extraer_capa_cmyk(indice: int) -> Imagen`**
  
  Simula la extracción de una capa en formato CMYK a partir de una imagen en RGB. Aunque la imagen original es RGB, el método utiliza reglas específicas para "extraer" las componentes que corresponderían a cyan, magenta, yellow o black según el índice (0 a 3). Devuelve una nueva instancia de `Imagen` con la capa extraída.

- **`mean_filter(kernel_size: int = 3) -> Imagen`**
  
  Aplica un filtro de promedio (o media) sobre la imagen. Para cada píxel, calcula el promedio de los valores en una vecindad definida por un kernel de tamaño `kernel_size` (que debe ser impar) y asigna este valor al píxel. El método utiliza padding con modo 'reflect' para manejar los bordes y retorna la misma instancia modificada.

- **`gris_promedio() -> Imagen`**
  
  Convierte la imagen a escala de grises utilizando el promedio de los tres canales de color. Primero, calcula el promedio para cada píxel y luego replica ese valor en los tres canales para mantener el mismo número de dimensiones. Devuelve una nueva instancia de `Imagen` con la imagen en escala de grises.

- **`gris_luminosidad() -> Imagen`**
  
  Realiza la conversión a escala de grises aplicando la fórmula de luminosidad, que pondera cada canal (R, G y B) de acuerdo con la percepción humana (0.299 para R, 0.587 para G y 0.114 para B). El resultado se replica en los tres canales, devolviendo una nueva instancia de `Imagen` en escala de grises.

- **`gris_tonalidad() -> Imagen`**
  
  Convierte la imagen a escala de grises utilizando el método de tonalidad (midgray). Para cada píxel, toma el promedio entre el valor máximo y mínimo de los canales y replica este valor en los tres canales, retornando una nueva instancia de `Imagen` en escala de grises.

- **`ajustar(factor: float) -> Imagen`**

  Ajusta la imagen aplicando un filtro basado en el valor de factor, que se espera esté en el rango [-1, 1].

  - Si `factor` es negativo, se aplica un filtro que realza el contraste mediante una transformación logarítmica.
  - Si `factor` es positivo, se aplica un filtro que realza la intensidad utilizando una transformación exponencial.
  - Si `factor` es 0, no se realiza ningún ajuste. \
    El método selecciona la estrategia adecuada utilizando un patrón de diseño Factory y retorna la instancia modificada.

- **`fusionar(imagenes: list[Imagen]) -> Imagen`**
  
  Fusiona varias imágenes de las mismas dimensiones realizando una suma pixel a pixel de sus arreglos. Antes de la fusión, valida que todas las imágenes tengan el mismo tamaño (en filas y columnas). Devuelve una nueva instancia de `Imagen` con el resultado de la fusión.

- **`fusionar_ecualizado(imagenes: list[tuple[Imagen, int]]) -> Imagen`**
  
  Similar al método anterior, pero permite aplicar un factor de ecualización a cada imagen antes de sumarlas. Cada imagen se multiplica por el factor especificado en la tupla correspondiente, y luego se realiza la suma pixel a pixel. Retorna una nueva instancia de Imagen con la imagen fusionada.

### Clase `ColorConverter`

- **`rgb_a_cmyk(imagen: Imagen) -> Imagen`**  
  
  Este método transforma una imagen en formato RGB a CMYK. Primero, valida que la imagen tenga exactamente 3 canales (R, G y B). Luego, se normalizan los datos (si no lo están ya) para trabajar en el rango [0, 1]. Se extraen los canales R, G y B y se calcula el canal K como la diferencia entre 1 y el valor máximo de los tres canales para cada píxel. Con K calculado, se determinan los canales C, M y Y utilizando fórmulas que ajustan cada componente de color en función de la luminosidad del píxel. Finalmente, los canales C, M, Y y K se combinan en un arreglo de 4 canales y se retorna una nueva instancia de Imagen que representa la imagen en formato CMYK.

- **`cmyk_a_rgb(imagen: Imagen) -> Imagen`**  
  
  Este método realiza la operación inversa, convirtiendo una imagen en formato CMYK a RGB. Primero, verifica que la imagen tenga 4 canales. Se extraen los valores de los canales C, M, Y y K y se aplican fórmulas que permiten obtener los valores correspondientes de R, G y B. La fórmula utilizada es:

    ```
    R = (1 - C) \cdot (1 - K)
    G = (1 - M) \cdot (1 - K) 
    B = (1 - Y) \cdot (1 - K)
    ```

  Estos cálculos producen una imagen en el rango [0, 1] para cada canal RGB. Finalmente, se retorna una nueva instancia de `Imagen` con estos datos, permitiendo que la imagen convertida pueda ser utilizada para visualización o procesamiento adicional.

### Estrategias de Filtro

- **`FiltroContraste`**  
  
  Se utiliza para realzar el contraste de la imagen mediante una transformación logarítmica. Se espera un factor negativo. En su constructor, verifica que el factor sea negativo y almacena su valor absoluto. En el método `aplicar`, normaliza la imagen y calcula una transformación logarítmica (usando una constante derivada de `log10`) para enfatizar las zonas oscuras, mezclando esta transformación con la imagen original en proporción al factor.

- **`FiltroIntensidad`**  
  
  Realza la intensidad de la imagen usando una transformación exponencial. Requiere un factor positivo y, en su método `aplicar`, normaliza la imagen y aplica la transformación exponencial para potenciar las zonas claras, mezclándola nuevamente con la imagen original.

- **`FiltroIdentity`**  
  
  Es la estrategia por defecto que no realiza ningún cambio en la imagen; simplemente retorna una copia de la misma. Esto se utiliza cuando el factor es 0, lo que indica que no se desea aplicar ningún ajuste.

- **`FiltroFactory.obtener_filtro(factor: float) -> FiltroStrategy`** 

  Este método actúa como una fábrica (*Factory Pattern*) para seleccionar y retornar la estrategia de filtrado adecuada en función del valor del factor proporcionado:

    - Si el factor es 0, retorna una instancia de `FiltroIdentity`.
    - Si el factor es negativo, retorna una instancia de `FiltroContraste`.
    - Si el factor es positivo, retorna una instancia de `FiltroIntensidad`.

  De esta manera, se encapsula la lógica de selección del filtro y se facilita la extensión futura (por ejemplo, añadiendo nuevos tipos de filtros sin modificar el código del cliente).

### Clase `SimpleImageViewer`

- **`show()`**  
  
  El método `show()` de la clase `SimpleImageViewer` se encarga de la visualización de múltiples imágenes organizándolas en una cuadrícula fija de 2x2 por figura. El funcionamiento detallado es el siguiente:

    - Se parte de un diccionario de imágenes, donde las claves son títulos descriptivos y los valores pueden ser arreglos NumPy o instancias de `Imagen` (se extrae el arreglo mediante una función interna).
    - Para cada grupo de hasta 4 imágenes, se crea una figura utilizando `matplotlib.pyplot.subplots` con una cuadrícula de 2 filas y 2 columnas. El tamaño de la figura se define mediante un factor de escala, garantizando un tamaño estándar para todas las imágenes.
    - Se ajusta el espaciado entre subplots (usando `subplots_adjust`) para asegurar que ni las imágenes ni sus títulos se solapen.
    - Cada imagen se muestra en su respectivo subplot: si la imagen es un arreglo 2D se utiliza un mapa de colores (por ejemplo, `cmap="gray"`), y si es un arreglo 3D se muestra con los colores originales.
    - Se ocultan los ejes de aquellos subplots que no se utilizan en caso de que la figura no se llene completamente.
    - Finalmente, se llama a `plt.show()` para renderizar todas las figuras generadas.

  Esta implementación garantiza una presentación consistente y clara de los resultados del procesamiento, facilitando la comparación y evaluación visual de diferentes transformaciones aplicadas a las imágenes.

---

## Contribuciones

Si deseas contribuir a **IS623-ImageTools**, te invitamos a:

- Abrir un *issue* para reportar errores o sugerir mejoras.
- Enviar un *pull request* con tus aportes.

Toda contribución es bienvenida para enriquecer esta herramienta educativa. También te invito a escribirme al correo electrónico *kevin.esguerra@utp.edu.co*

---

## Licencia

Este proyecto se distribuye bajo la **Licencia MIT**. Consulta el archivo [LICENSE](LICENSE.yaml) para más detalles.

---

## Créditos

- **Profesor:** Francisco Alejandro Medina Aguirre  
- **Desarrollador:** Kevin Esguerra Cardona  
- **Universidad:** Universidad Tecnológica de Pereira (UTP)  
- **Tecnologías utilizadas:** VS Code, GitHub Copilot, ChatGPT

---

Disfruta explorando y aprendiendo procesamiento digital de imágenes con **IS623-ImageTools**.