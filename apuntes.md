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
