# Text-Classification-with-spacy
This tool uses a pre-trained model for Spanish-to-English translation, allowing text of any size to be translated without being limited by the model's token restrictions. It handles large texts by splitting them into manageable parts and reassembling the translation seamlessly.
---
## Requirements 

I recomend to install the following libraries in the following order:
- python = 3.9
- transformers = 4.3 (i used 4.32.1)

---
## Code
#### Importing libraries 
```python
#Importing libraries
from transformers import MarianMTModel, MarianTokenizer
import math
```
#### Load the model 
In this case we load a pre-trained model to make traductions from spanish to english
```python
model_name = 'Helsinki-NLP/opus-mt-es-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
```
#### Input the text that you want to translate
```python
text = "Como sabemos, esta semana se celebró un juego de futbol en la ciudad de Monterrey entre los equipos Monterrey e Inter Miami, con Leo Messi en sus filas.  Con motivo de este partido, visitó Monterrey el famoso jugador argentino de futbol Mario Kempes, quién fue campeón del mundo con Argentina en 1978. Durante su visita, Kempes se quejó del tráfico de la ciudad de la siguiente manera: “Lo que hemos visto está bien, lo que pasa es que bueno el tráfico generalmente aquí en México es bastante complicado, y Monterrey no es menos, espero que de cualquier manera que a medida que se vaya acercando esa gran posibilidad de Mundial, se hagan arreglos para que el ida y vuelta sea más fácil”. Por lo demás, sabemos que el tráfico de automóviles no es un problema exclusivo de nuestro país y que lo experimentan muchas ciudades a lo largo del mundo.  Por otro lado, si bien el tráfico no es la única incomodidad que enfrentamos quienes vivimos en áreas urbanas y nos faltarían dedos de las manos para enumerarlas, hemos decidido permanecer en las ciudades por las ventajas que también tienen. De hecho, las ciudades nacieron hace miles de años porque en algún momento resultó conveniente incrementar el contacto entre personas.En un artículo aparecido la pasada semana en la revista “Journal of Archaeological  Method and Theory”, se reportan los resultados de un estudio sobre las primeras etapas de desarrollo de áreas urbanas de baja densidad en Tongatapu. Tongatapu es la isla principal de Tonga, un país insular del Pacífico sur, de 100,000 habitantes, que por su  tamaño ocupa el lugar 186 entre los países del mundo. El artículo fue publicado por Phillip Parton y Geoffrey Clark de la Universidad Nacional de Australia.Como comentan Parton y Clark, Tonga fue primeramente ocupado en el año 900 a.C. y floreció desde el año 1300 d.C, hasta su colapso a inicios del siglo XIX, parcialmente por el impacto de enfermedades introducidas por extranjeros. Durante su esplendor, Tonga desarrolló arquitectura monumental, redes de tráfico comercial e instituciones políticas y sociales. Adicionalmente, dado su aislamiento en medio del océano Pacífico, los asentamientos urbanos primitivos de Tonga se desarrollaron sin influencia externa, y su estudio revela información valiosa sobre la evolución urbana. En su investigación, Parton y Clark utilizaron datos topográficos de Tongatapu obtenidos mediante la técnica de Lidar, que consiste en lanzar un haz de luz láser desde un avión hacia la superficie del terreno, midiendo el tiempo que tarda en regresar después de reflejarse en dicha superficie. A partir de este tiempo, es posible determinar las  elevaciones y depresiones del terreno. El uso del Lidar proporciona una enorme cantidad de datos que no es posible obtener por medio de los arqueológicos tradicionales. En particular, Parton y Clark estaban interesados en localizar montículos artificiales de tierra que son comunes en Tobgatapu y que se sabe fueron construidos con diferentes propósitos, ya sea como tumbas, o como plataformas para la construcción de casas habitación o espacios públicos. Además de los montículos, el Lidar revela redes de caminos que los enlazan, lo mismo que fortificaciones y construcciones para practicar deporte.Parton y Clark describen el proceso de urbanización de Tobgatapu, que se iniciaría cuando se incrementa la población dentro de los límites de las áreas pobladas  y genera lo que llama efectos de aglomeración: “La aglomeración causa cambios en la forma en la cual están construidos los asentamientos a medida que los pobladores empiezan a hacer un uso más eficiente del espacio y hacen un balance de las ventajas del cambio con los costos que implica. Efectos de aglomeración más grandes estimulan el desarrollo de instituciones sociales a medida que los asentamientos se adaptan a interacciones sociales cada vez mayores. Las instituciones sociales también provocan cambios en la forma en que está construido un asentamiento, al competir por espacios para realizar sus funciones con otros usos de la tierra, como el residencial, de subsistencia y otros usos productivos”. Basados en su estudio, Parton y Clark concluyen que los asentamientos en el Pacífico tienen un potencial considerable para contribuir a debates sobre la formación de asentamientos, la urbanización y la sostenibilidad, y contribuirán a nuestro conocimiento sobre urbanización y desarrollo de sociedades complejas. Ciertamente, entre mejor entendamos el proceso de formación de las ciudades podremos desarrollar medidas para contrarrestar las desventajas que representa vivir en una de ellas.  Una ciudad, sin embargo, es un objeto de estudio muy complejo -incluso más que el clima del planeta- y entender que botones presionar para que cambie en una dirección u otra, no será  algo que logremos en un futuro cercano. No obstante, esperemos que Monterrey pueda mejorar, aunque sea un poquito, el tráfico de la ciudad durante el mundial de futbol"
```
### Dividing the text into blocks
The reason for doing this is because the model only accepts short texts of no more than approximately 250 tokens. For this reason we divide the text into blocks that contain the same number of tokens.

```python
tokens = tokenizer.tokenize(text) #split the text in tokens
len_tokens = len(tokens)#quantity of tokens in the text 
print(f'Tokens size: {len_tokens}')#959 para este caso
sizeblock = 240 #the number of tokens that each block will have
```
#### Creating the blocks
```python
#Here we check if the amount of tokens is greater than the proposed limit
if len_tokens > sizeblock:
    n_parts = math.ceil(len_tokens/sizeblock)#number of blocks
    partes = {}
    aux = 0 #we will use this auxilary as index for each block
    rest = len_tokens #variable to see how much tokens have not been saved in the diffent parts.

    for i in range(n_parts):
        total = rest - sizeblock
        #In the first iteration, rest has to be bigger than sizeblock
        if rest >= sizeblock:
            partes[i] = tokens[aux:(sizeblock*(i+1))] #we always add the sizeblock tokens if the condition pass
            partes[i] = tokenizer.convert_tokens_to_string(partes[i]) #we transform the tokens into text
            aux=(sizeblock*(i+1)) #we add an additional 1 to avoid start with the same token

        else: #if the result of the substraction is less than the sizeblock
            partes[i] = tokens[aux:len_tokens] #here we add last tokens.
            partes[i] = tokenizer.convert_tokens_to_string(partes[i])#we transform the tokens into text
        
        rest=total #here we update the calue of rest
```
In this part we already have blocks of maximum size "sizeblock" tokens. So in the next part, we are going to translate each block, and than mix each block in one list.
```python

translation = {}

for i, text in partes.items():
    # Input the text of each block into the model and tokenizer
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    # Generate the translation
    translated_tokens = model.generate(**inputs)
    translation[i] = tokenizer.decod
```

Here we can see the original text in spanish and english for each block 
```python
for i in range(n_parts):
    print(f"Original{i+1}: {partes[i]}")
    print('\n')
    print(f"Traducción{i+1}: {translation[i]}")
    print('\n')
    print('\n')
```
Join all strings into a single string with a space as a separator
```python
translations = translation.values()
full_translations = ' '.join(translations)

print(full_translations)
```


