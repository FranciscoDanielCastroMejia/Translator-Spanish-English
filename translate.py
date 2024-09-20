import re #para la division de texto 
from transformers import MarianMTModel, MarianTokenizer
import math
import json

import time
from progressbar import ProgressBar


# Nombre del modelo
model_name = 'Helsinki-NLP/opus-mt-es-en'

# Cargar el tokenizador y el modelo
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


# Cargamos el diccionario  qeu contiene los links y los articulos sin resumen ni traduccion 
with open('Bases de datos/links_article_not_ST.json', 'r') as file:
    links_article_not_ST = json.load(file)

links_article_T_not_S = {} #diccionario final, en donde se agregara el texto traducido a ingles con su respectivo link

sizeblock = 240 #size of each block of tokens





"""


This tool uses a pre-trained model for Spanish-to-English translation, allowing text of any size to be translated without being limited by the model's token restrictions. It handles large texts by splitting them into manageable parts and reassembling the translation seamlessly.


print("Inicio de traductor")

#inicializamos la barra de progreso
barra = ProgressBar(maxval=len(links_article_not_ST)).start()
barrita=0 

for link, text in links_article_not_ST.items():

    print(f'traduccion numero {barrita}')

    tokens = tokenizer.tokenize(text)#sacamos la cantidad de tokens que contiene cada articulo
    len_tokens = len(tokens)#quantity of tokens in the text

    #En esta parte dividiremos el texto en bloques del size necesario para que logre hacer bien la traduccion. 

    if len_tokens > sizeblock:
        n_parts = math.ceil(len_tokens/sizeblock)#numero de partes
        partes = {} #diccionario donde se almacenara las diferentes partes para hacer el resumen
        aux = 0 #ayudará al indice de cada arreglo
        
        rest = len_tokens #variable to see how much tokens have not been saved in the diffent parts.

        for i in range(n_parts):
            
            total = rest - sizeblock

            if rest >= sizeblock:
                partes[i] = tokens[aux:(sizeblock*(i+1))] #we always add the sizeblock tokens if the condition pass
                #print(len(partes[i]))
                partes[i] = tokenizer.convert_tokens_to_string(partes[i]) #we transform the tokens into text
                aux=(sizeblock*(i+1)) #we add an additional 1 to avoid start with the same token

            else: #if the result of the substraction is less than the sizeblock
                partes[i] = tokens[aux:len_tokens] #here we add last tokens.
                partes[i] = tokenizer.convert_tokens_to_string(partes[i])#we transform the tokens into text
                #print(len(partes[i]))
            
            rest=total
    
    #In this part we already have blocks of maximum size "sizeblock" tokens. So in the next part, 
    # we are going to translate each block, and than mix each block in one list.
            
   
    max_length = 512 # Ajustar el máximo de tokens permitidos
    translation = {}

    #En esta parte se genera la traduccion de cada bloque
    for i,  part_text in partes.items():

        # Preparar el texto para el modelo
        inputs = tokenizer(part_text, return_tensors="pt", max_length=max_length, truncation=True)
        # Generar la traducción
        translated_tokens = model.generate(**inputs)
        translation[i] = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    #A continuacion, juntamos los bloques en uno solo
    
    # Obtener todas las cadenas del diccionario
    translations = translation.values()

    # Unir todas las cadenas en una sola cadena con un espacio como separador
    full_translations = ' '.join(translations)

    #agregar las traducciones a un nuevo diccionario "links_article_T_not_S"
     
    links_article_T_not_S[link] = full_translations

    #actualizacion de la barra de progreso
    barrita+=1
    barra.update(barrita)

barra.finish()

#Aqui guardamos las traducciones
with open("Bases de datos/links_article_T_not_S.json", 'w') as file:
    json.dump(links_article_T_not_S, file, indent=4)



    
"""













#esta seccion de codigo es para realizar la traduccion de un solo texto


"""# Texto en español que quieres traducir
text = "Como sabemos, esta semana se celebró un juego de futbol en la ciudad de Monterrey entre los equipos Monterrey e Inter Miami, con Leo Messi en sus filas.  Con motivo de este partido, visitó Monterrey el famoso jugador argentino de futbol Mario Kempes, quién fue campeón del mundo con Argentina en 1978. Durante su visita, Kempes se quejó del tráfico de la ciudad de la siguiente manera: “Lo que hemos visto está bien, lo que pasa es que bueno el tráfico generalmente aquí en México es bastante complicado, y Monterrey no es menos, espero que de cualquier manera que a medida que se vaya acercando esa gran posibilidad de Mundial, se hagan arreglos para que el ida y vuelta sea más fácil”. Por lo demás, sabemos que el tráfico de automóviles no es un problema exclusivo de nuestro país y que lo experimentan muchas ciudades a lo largo del mundo.  Por otro lado, si bien el tráfico no es la única incomodidad que enfrentamos quienes vivimos en áreas urbanas y nos faltarían dedos de las manos para enumerarlas, hemos decidido permanecer en las ciudades por las ventajas que también tienen. De hecho, las ciudades nacieron hace miles de años porque en algún momento resultó conveniente incrementar el contacto entre personas.En un artículo aparecido la pasada semana en la revista “Journal of Archaeological  Method and Theory”, se reportan los resultados de un estudio sobre las primeras etapas de desarrollo de áreas urbanas de baja densidad en Tongatapu. Tongatapu es la isla principal de Tonga, un país insular del Pacífico sur, de 100,000 habitantes, que por su  tamaño ocupa el lugar 186 entre los países del mundo. El artículo fue publicado por Phillip Parton y Geoffrey Clark de la Universidad Nacional de Australia.Como comentan Parton y Clark, Tonga fue primeramente ocupado en el año 900 a.C. y floreció desde el año 1300 d.C, hasta su colapso a inicios del siglo XIX, parcialmente por el impacto de enfermedades introducidas por extranjeros. Durante su esplendor, Tonga desarrolló arquitectura monumental, redes de tráfico comercial e instituciones políticas y sociales. Adicionalmente, dado su aislamiento en medio del océano Pacífico, los asentamientos urbanos primitivos de Tonga se desarrollaron sin influencia externa, y su estudio revela información valiosa sobre la evolución urbana. En su investigación, Parton y Clark utilizaron datos topográficos de Tongatapu obtenidos mediante la técnica de Lidar, que consiste en lanzar un haz de luz láser desde un avión hacia la superficie del terreno, midiendo el tiempo que tarda en regresar después de reflejarse en dicha superficie. A partir de este tiempo, es posible determinar las  elevaciones y depresiones del terreno. El uso del Lidar proporciona una enorme cantidad de datos que no es posible obtener por medio de los arqueológicos tradicionales. En particular, Parton y Clark estaban interesados en localizar montículos artificiales de tierra que son comunes en Tobgatapu y que se sabe fueron construidos con diferentes propósitos, ya sea como tumbas, o como plataformas para la construcción de casas habitación o espacios públicos. Además de los montículos, el Lidar revela redes de caminos que los enlazan, lo mismo que fortificaciones y construcciones para practicar deporte.Parton y Clark describen el proceso de urbanización de Tobgatapu, que se iniciaría cuando se incrementa la población dentro de los límites de las áreas pobladas  y genera lo que llama efectos de aglomeración: “La aglomeración causa cambios en la forma en la cual están construidos los asentamientos a medida que los pobladores empiezan a hacer un uso más eficiente del espacio y hacen un balance de las ventajas del cambio con los costos que implica. Efectos de aglomeración más grandes estimulan el desarrollo de instituciones sociales a medida que los asentamientos se adaptan a interacciones sociales cada vez mayores. Las instituciones sociales también provocan cambios en la forma en que está construido un asentamiento, al competir por espacios para realizar sus funciones con otros usos de la tierra, como el residencial, de subsistencia y otros usos productivos”. Basados en su estudio, Parton y Clark concluyen que los asentamientos en el Pacífico tienen un potencial considerable para contribuir a debates sobre la formación de asentamientos, la urbanización y la sostenibilidad, y contribuirán a nuestro conocimiento sobre urbanización y desarrollo de sociedades complejas. Ciertamente, entre mejor entendamos el proceso de formación de las ciudades podremos desarrollar medidas para contrarrestar las desventajas que representa vivir en una de ellas.  Una ciudad, sin embargo, es un objeto de estudio muy complejo -incluso más que el clima del planeta- y entender que botones presionar para que cambie en una dirección u otra, no será  algo que logremos en un futuro cercano. No obstante, esperemos que Monterrey pueda mejorar, aunque sea un poquito, el tráfico de la ciudad durante el mundial de futbol"
#text = "Como sabemos, esta semana se celebró un juego de futbol."
#text = "Como sabemos, esta semana se celebró un juego de futbol en la ciudad de Monterrey entre los equipos Monterrey e Inter Miami, con Leo Messi en sus filas. Con motivo de este partido, visitó Monterrey el famoso jugador argentino de futbol Mario Kempes, quién fue campeón del mundo con Argentina en 1978. Durante su visita, Kempes se quejó del tráfico de la ciudad de la siguiente manera: “Lo que hemos visto está bien, lo que pasa es que bueno el tráfico generalmente aquí en México es bastante complicado, y Monterrey no es menos, espero que de cualquier manera que a medida que se vaya acercando esa gran posibilidad de Mundial, se hagan arreglos para que el ida y vuelta sea más fácil”. Por lo demás, sabemos que el tráfico de automóviles no es un problema exclusivo de nuestro país y que lo experimentan muchas ciudades a lo largo del mundo. Por otro lado, si bien el tráfico no es la única incomodidad que enfrentamos quienes vivimos en áreas urbanas y nos faltarían dedos de las manos para enumerarlas, hemos decidido permanecer en las ciudades por las ventajas que también tienen. De hecho, las ciudades nacieron hace miles de años porque en algún momento resultó conveniente incrementar el contacto entre personas"

#tokenizacion
tokens = tokenizer.tokenize(text)
len_tokens = len(tokens)#quantity of tokens in the text
print(f'Tokens size: {len_tokens}')#959 para este caso
sizeblock = 240

#En esta parte dividiremos el texto en bloques del size necesario para que logre hacer bien la traduccion. 

if len_tokens > sizeblock:
    n_parts = math.ceil(len_tokens/sizeblock)#numero de partes
    partes = {}
    aux = 0 #ayudará al indice de cada arreglo
    
    rest = len_tokens #variable to see how much tokens have not been saved in the diffent parts.

    for i in range(n_parts):
        
        total = rest - sizeblock

        if rest >= sizeblock:
            partes[i] = tokens[aux:(sizeblock*(i+1))] #we always add the sizeblock tokens if the condition pass
            #print(len(partes[i]))
            partes[i] = tokenizer.convert_tokens_to_string(partes[i]) #we transform the tokens into text
            aux=(sizeblock*(i+1)) #we add an additional 1 to avoid start with the same token

        else: #if the result of the substraction is less than the sizeblock
            partes[i] = tokens[aux:len_tokens] #here we add last tokens.
            partes[i] = tokenizer.convert_tokens_to_string(partes[i])#we transform the tokens into text
            #print(len(partes[i]))
        
        rest=total


#In this part we already have blocks of maximum size "sizeblock" tokens. So in the next part, 
# we are going to translate each block, and than mix each block in one list.
        


# Ajustar el máximo de tokens permitidos
max_length = 512  # Puedes ajustar este valor según tus necesidades
translation = {}

for i, text in partes.items():
    
    #print(text)
    #print('\n')

    # Preparar el texto para el modelo
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    # Generar la traducción
    translated_tokens = model.generate(**inputs)
    translation[i] = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)



for i in range(n_parts):

    print(f"Original{i+1}: {partes[i]}")
    print('\n')
    print(f"Traducción{i+1}: {translation[i]}")
    print('\n')
    print('\n')


# Obtener todas las cadenas del diccionario
translations = translation.values()

# Unir todas las cadenas en una sola cadena con un espacio como separador
full_translations = ' '.join(translations)

print(full_translations)"""




