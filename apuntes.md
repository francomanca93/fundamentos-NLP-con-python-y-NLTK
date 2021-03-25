<div align="center">
    <h1>Curso de Fundamentos de Procesamiento de Lenguaje Natural con Python y NLTK</h1>
    <img src="https://imgur.com/V7P8rn2.png" width="">
</div>

### Indice
- [Introducción al Procesamiento de Lenguaje Natural](#introducción-al-procesamiento-de-lenguaje-natural)
  - [Introducción a NLP y definiciones](#introducción-a-nlp-y-definiciones)
    - [Importancia de NLP](#importancia-de-nlp)
    - [NLP y NLU](#nlp-y-nlu)
    - [Usos de NLP](#usos-de-nlp)
    - [Dificultad y Test de Turing](#dificultad-y-test-de-turing)
  - [Evolución del NLP](#evolución-del-nlp)
    - [Linea del tiempo del NPL](#linea-del-tiempo-del-npl)
    - [Avances del NLP](#avances-del-nlp)
    - [Lo que estudiaremos y Roadmap](#lo-que-estudiaremos-y-roadmap)
  - [Conceptos básicos de NLP](#conceptos-básicos-de-nlp)
- [Fundamentos con NLTK](#fundamentos-con-nltk)
  - [Inicio a los fundamentos | Configurar ambiente de trabajo](#inicio-a-los-fundamentos--configurar-ambiente-de-trabajo)
  - [Palabras, textos y vocabularios | Expresiones Regulares](#palabras-textos-y-vocabularios--expresiones-regulares)
  - [Tokenizacion con Expresiones Regulares](#tokenizacion-con-expresiones-regulares)
- [Aplicaciones | Estadísticas del lenguaje](#aplicaciones--estadísticas-del-lenguaje)
  - [Estadísticas básicas del lenguaje](#estadísticas-básicas-del-lenguaje)
  - [Distribuciónes de frecuencia de palabras](#distribuciónes-de-frecuencia-de-palabras)
  - [Refinamiento y visualización de cuerpos de texto](#refinamiento-y-visualización-de-cuerpos-de-texto)

# Introducción al Procesamiento de Lenguaje Natural

## Introducción a NLP y definiciones

### Importancia de NLP

- **¿Por qué es tan importante el procesamiento de lenguaje natural?**
  - Por que creemos que es el autentico camino de lo que creemos que es el ideal de lo que nosotros consideramos inteligencia artificial.

### NLP y NLU

- **¿Qué significa y de que se encarga el NLP?**
  - Significa **Natural Language Processing** y este es un área que combina ciencias de la computación, lingüística, IA para entender como se pueden ejecutar interacciones entre humanos y maquinas por medio le lenguaje natural.

- **¿Qué significa y de que se encarga el NLU?**
  - Significa **Natural Language Understanding** y esta es una sub área del NLP que se encarga de tareas especificas que las maquinas puedan ejecutar en el proceso de comunicación de los seres humanos de manera que esas tarea reflejen que el robot no solo puede procesar nuestro lenguaje, si no entenderlo y las respuestas que nos dé, deben de reflejar que verdaderamente lo entiende.

### Usos de NLP

- **Usos actuales del NLP**:
  - Máquinas de búsqueda o motores de búsqueda.
  - Traductores de texto.
  - Chatbots.
  - Análisis de discurso.
  - Reconociendo del habla.

### Dificultad y Test de Turing

- **¿Por qué es tan difícil el NLP?**
  - El problema mas grande son las ambigüedades.
  - Es difuso.
  - Requiere de mucho contexto

- **¿Qué es el test de Turing?**
  - Es un examen de la capacidad de una máquina para exhibir un comportamiento inteligente similar al de un ser humano o indistinguible de este.
  - Alan Turing propuso que un humano evaluara conversaciones en lenguaje natural entre un humano y una máquina diseñada para generar respuestas similares a las de un humano. El evaluador sabría que uno de los participantes de la conversación es una máquina y los intervinientes serían separados unos de otros. La conversación estaría limitada a un medio únicamente textual como un teclado de computadora y un monitor por lo que sería irrelevante la capacidad de la máquina de transformar texto en habla. En el caso de que el evaluador no pueda distinguir entre el humano y la máquina acertadamente (Turing originalmente sugirió que la máquina debía convencer a un evaluador, después de 5 minutos de conversación, el 70 % del tiempo), la máquina habría pasado la prueba. Esta prueba no evalúa el conocimiento de la máquina en cuanto a su capacidad de responder preguntas correctamente, solo se toma en cuenta la capacidad de esta de generar respuestas similares a las que daría un humano.
  - [Prueba de Turing](https://es.wikipedia.org/wiki/Prueba_de_Turing)

## Evolución del NLP

### Linea del tiempo del NPL

![nlp-evolucion-temporal](https://imgur.com/FqvjOpL.png)

- Entre 1950 y 1990: Sistemas basados en reglas. A las máquinas se las preprogramaba con todas las reglas de la linguistica para poder comunicarse. Esto basado en todo nuestro conocimiento linguistico del lenguaje, reglas, para saber cuando algo esta bien dicho o escrito.
- Entre 1990 y 2000: Estadística de corpus. Algoritmos basados en estadisticas de corpus. Un corpus es una coleccion de diferentes textos.
- Entre 2000 y 2014: Machine Learning.
- Entre 2014 y 2020: Deep Learning.

### Avances del NLP

El NLP ha tenido 2 vertientes muy grandes diferenciadas en el siguiente esquema:

- Entendimiento de texto (bajo nivel): Algoritmos especificos para una tarea concreta.
- Aprendizaje de representaciones: Algoritmos basados en redes neuronales, donde se pueden crear arquitectura para realizar multitareas.

![avances-nlp](https://imgur.com/AdQ8kWa.png)

![avanzes-2](https://imgur.com/DTwfNRv.png)

- LSTM: Redes secuenciales que procesan las oraciones teniendo en cuenta las palabras anterior a la que se esta analizando.
- BiLSTM: Para mejorar el contexto de las oraciones hay que ver las palabras anteriores y posteriores. Estas son una evolución de las redes anteriores.
- Transformer: Las palabras estan dadas por su contexto en la oración, pero podemos simplificar el analisis unicamente analizando ciertas palabras. Analizando palabras especificas en puntos especificos de una oración o un texto, todo esto sin tener que ver toda la oración, esto es llamado **mecanismo de atención**. Los transformer utilizan este concepto.
- Reformer. Evolución de los transformers.

### Lo que estudiaremos y Roadmap

![librerias](https://imgur.com/bfaadDp.png)

![roadmap](https://imgur.com/7QNH1Fi.png)

## Conceptos básicos de NLP

![](https://imgur.com/fRc1TLu.png)

- **NLP**: El procesamiento de lenguaje natural esta más enfocado hacia aplicaciones practicas en la ingeniería
- **LC**: La lingüística computacional estudia el lenguaje desde una perspectiva más científica.
  - Basada en crear modelos que pueden tener dos enfoques de conocimiento o datos.

  ![LP](https://imgur.com/pdFgtKL.png)

- **TEXTO**: El procesamiento de una cadena de texto necesita una Normalización que incluye los siguientes procesos:
![](https://imgur.com/oToe8kw.png)

  - **Tokenización**: Separar en palabras toda la cadena de texto
  
  ![tokenizacion](https://imgur.com/2Z7dGWb.png)
  
  - **Lematización**: Convertir cada una de las palabras a su raiz fundamental
  
  ![lematizacino](https://imgur.com/URrYjzV.png)
  
  - **Segmentación**: Separación en frases (puede ser con las comas)
  
  ![segmentacion](https://imgur.com/92EDLYt.png)

- **CORPUS**: Colección de muchos textos

![corpus](https://imgur.com/gn00ue1.png)

- **CORPORA**: Colección de colecciones de texto

![corpora](https://imgur.com/7RCaaXO.png)

# Fundamentos con NLTK

## Inicio a los fundamentos | Configurar ambiente de trabajo

> [Notebook configurando ambiente de trabajo](1_fundamentos_nlp.ipynb)

**Corpus lingüístico**

Un corpus lingüístico es un conjunto amplio y estructurado de ejemplos reales de uso de la lengua. Estos ejemplos pueden ser textos, o muestras orales.​ Un corpus lingüístico es un conjunto de textos relativamente grande, creado independientemente de sus posibles formas o usos.

**Token**

Un token es un conjunto de caracteres que representan texto. También podemos decir que el token es la unidad análisis de texto, así como un número es la unidad del análisis matemático. Es fácil para nosotros pensar que un token es igual a una palabra, sin embargo esto no es correcto, puesto que la “palabra” es un elemento del lenguaje que posee significado por sí misma, mientras que el token se supone es un elemento abstracto. Dependiendo de la tarea que estemos afrontando, el token puede ser alguna de las siguientes:

    Una sola palabra, como: “jóvenes”, “nivel” o “superior”,
    Un número, como: “1”, “0”, o “10”,
    Un solo caracter, como: “j”, “ó” o “v”,
    Un símbolo, como “¿”, “?” o “#”,
    Un conjunto de caracteres, como “nivel superior” o “escuela técnica”

**Tokenización**

La tokenización es un paso que divide cadenas de texto más largas en piezas más pequeñas o tokens. Los trozos de texto más grandes pueden ser convertidos en oraciones, las oraciones pueden ser tokenizadas en palabras, etc. El procesamiento adicional generalmente se realiza después de que una pieza de texto ha sido apropiadamente concatenada. La tokenización también se conoce como segmentación de texto o análisis léxico. A veces la segmentación se usa para referirse al desglose de un gran trozo de texto en partes más grandes que las palabras (por ejemplo, párrafos u oraciones), mientras que la tokenización se reserva para el proceso de desglose que se produce exclusivamente en palabras.

Algunos enlaces:

- [Corpus lingüístico. Wikipedia](https://es.wikipedia.org/wiki/Corpus_lingüístico)
- [Introducción al análisis de texto. 🌮 tacos de datos](https://tacosdedatos.com/analisis-texto#:%7E:text=Un%20token%20es%20un%20conjunto,la%20unidad%20del%20an%C3%A1lisis%20matem%C3%A1tico.&text=Un%20conjunto%20de%20caracteres%2C%20como,superior%E2%80%9D%20o%20%E2%80%9Cescuela%20t%C3%A9cnica%E2%80%9D)
- [Preprocesamiento de datos de texto: un tutorial en Python. Medium](https://medium.com/datos-y-ciencia/preprocesamiento-de-datos-de-texto-un-tutorial-en-python-5db5620f1767#:%7E:text=single%20curly%20braces.%7D-,Tokenizaci%C3%B3n,ser%20tokenizadas%20en%20palabras%2C%20etc)

## Palabras, textos y vocabularios | Expresiones Regulares

> [Notebook aprendiendo REGEX](1_fundamentos_nlp.ipynb)

- Las Expresiones Regulares o **regex** constituyen un lenguaje estandarizado para definir cadenas de búsqueda de texto.
- Libreria de operaciones con  expresiones regulares de Python [re](https://docs.python.org/3/library/re.html)
- Reglas para escribir expresiones regulares [Wiki](https://es.wikipedia.org/wiki/Expresión_regular)
- [Expresiones Regulares Cheat Sheet from dataquest.io](https://www.dataquest.io/wp-content/uploads/2019/03/python-regular-expressions-cheat-sheet.pdf)

Básicos de REGEX

- Estructura de la funcion `re.search()`: Esta funcion determina si el patron de búsqueda p está contenido en la cadena s `re.search(p, s)`.
- Búsquedas en cadenas de texto con **meta caracteres** básicos:
  - `es`: Buscamos el 'es' en la cadena w **en donde este**.
  - `es$`: Buscamos el 'es' en la cadena w al **final**.
  - `^es`: Buscamos el 'es' en la cadena w al **principio**.

- Patrones de búsqueda usando el concepto de rango:
  - **Rango `[a-z]`**: Determina que el carácter debe estar ubicado entra la a y la z
  - **Rango `[ghi]`**: Determina que el carácter que este en esta posición puede ser cualquier letra entre la g, h e i.
- Clausuras
  - El `*` Esta clausura representa que se puede repetir 0 o más veces.
  - El `+`: Esta clausura representa que s puede repetir 1 o más veces.

## Tokenizacion con Expresiones Regulares

> [Notebook aplicando Tokenizacion con expresiones regulares](1_fundamentos_nlp.ipynb)

Aplicaremos las expresiones regulares para formar algoritmos de tokenización.

**Tokenización** es el proceso mediante el cual se sub-divide un cadena de texto en unidades linguisticas minimas

Aplicamos en nuestro notebook los siguientes casos:

- **Caso 1**: _tokenizar por espacios vacios_
- **Caso 2**: _tokenizacion usando regex_. Por diferentes tipos de espacios como tab, salto, espacio.
- **Caso 3**: _tokenizacion usando regex 2_. Para diferentes tipos de espacios y caracteres no alfanumericos. El metacaracter `\W` lo que hace es hacer match con todo lo que no sea un caracter alfanumérico (como paréntesis, símbolos raros, etc.)
- **Caso 4**: _tokenizacion usando regex mas fofisticada_. En lugar de usar `[\t\n]+` para definir un rango de espacios, tabs, etc. Se puede usar el metacaracter `\s`, específicamente diseñado para reconocer diferentes clases de espacios.
- **Caso 5**: Tokenizar usando la libreria NLTK para casos especiales.

# Aplicaciones | Estadísticas del lenguaje

## Estadísticas básicas del lenguaje

Descargamos una serie de libros que ya se encuentran tokenizados con `nltk.download('book')`. Con estos podemos empezar a aplicar estadistica basicas del lenguaje.

Aplicamos una métrica llamada **riqueza lexica en un texto**. Cuando tenemos un texto, es normal que algunas palabras se repitan. Queremos definir cuantas palabras unicas se utilizaron respecto al total de palabras del texto. Cuanto mas grande sea nuestra riqueza lexica, mas palabras diferentes utilizo nuestro autor. Sabiendo lo anterior creamos funciones para medir la requiza lexica:

```py
def riqueza_lexica(texto):
  vocabulario = sorted(set(texto))
  return len(vocabulario)/len(texto)

```

Tambien podemos saber el **porcentaje de aparicion de una palabra** en nuestro texto:

```py

def porcentaje_palabra(palabra, texto):
  porcentaje_palabra = 100 * texto.count(palabra)/len(texto)
  return porcentaje_palabra

```

Entre otras funciones **aplicando estadísticas al lenguaje** que podemos ver en el [Notebook](2_aplicaciones_estadísticas_del_lenguaje.ipynb)

## Distribuciónes de frecuencia de palabras

Los cálculos estadísticos más simples que se pueden efectuar sobre un texto o un corpus son los relacionados con frecuencia de aparición de palabras.

Cuando hacemos esta operacion sobre todas las palabras que componen el vocabulario de un texto construimos una distribucion de probabilidad de esas palabras dentro del texto.

Podemos hacer esto de 2 formas diferentes:

- **Realizar un for básico**, computacionalmete es muy pesado si tenemos muchos tokens.
  - Estaríamos hablando de un Big **On ** 2**, osea tiene **crecimiento polinominial**, estos son algoritmos que deben usarse cuando el input o la entrada de datos es pequeña.
- Utilizamos el **método FreqDist de NLTK**.
  - Este procesa un objeto de texto de una forma diferente a la que lo haría un for.

Podemos ver estas 2 aplicaciones [Notebook](2_aplicaciones_estadísticas_del_lenguaje.ipynb)

## Refinamiento y visualización de cuerpos de texto

Distribuciones sobre contenido con filtro-fino

- En la sección anterior vimos que los tokens más frecuentes en un texto no son necesariamente las palabras que mas informacion nos arrojan sobre el contenido del mismo.
- Por ello, es mejor filtrar y construir distribuciones de frecuencia que no consideren signos de puntuación o caracteres especiales.

Podemos ver estas aplicaciones y los gráficos en el [Notebook](2_aplicaciones_estadísticas_del_lenguaje.ipynb)