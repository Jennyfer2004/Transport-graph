import streamlit as st 
import PIL as pl
import os
import io
def main():
    st.title("Redes de Tranporte")
    st.write("")

    st.write("")

    with open('satelital.jpg', 'rb') as f:
        datos_imagen = f.read()
    imagen = pl.Image.open(io.BytesIO(datos_imagen)).resize((900,700))
    st.image(imagen,width=500)

    st.write("Imagina que cada vez que te subes a un medio de transporte, no solo estás cambiando de lugar, sino que estás navegando por un vasto océano de conexiones invisibles. ¡Vamos a desentrañar esta red mágica!")
    st.write("En el corazón de cada ciudad, en cada rincón del campo, se despliega una red de transporte vibrante y dinámica. Desde los autos que surcan las vías como flechas veloces, por ir al trabajo o regresar a casa, hasta los autobuses que serpentean por las calles, cada medio de transporte tiene su propio relato que contar. Pero, ¿qué sería de todos estos medios de transporte y las personas que se mueven en ellos si no existieran las carreteras?") 
    st.write("Las redes de transporte son fundamentales para el funcionamiento de cualquier comunidad. No solo facilitan la movilidad de personas y mercancías, sino que también influyen en el desarrollo económico, social y cultural de un pueblo o ciudad. ")
    st.write("Al seleccionar a la izquierda la opción de \"Red Interactiva\" te encuentras frente a un mapa vibrante que parece cobrar vida. Las líneas que se entrelazan son más que simples rutas; son los hilos invisibles que tejen nuestra sociedad. Cada nodo en este entramado es una puerta abierta a nuevas experiencias.")
    st.write("En ella, no olvides observa cómo las rutas se desvían de una línea recta. Estas desviaciones no son caprichos; son decisiones estratégicas que pueden llevarte a lugares inesperados. Cada curva cuenta una historia sobre cómo se adaptó la red a su entorno, como un artista que da forma a su obra maestra.")


opciones = ["Introdución" ,"Red Interactiva"]
selection = st.sidebar.radio('Selecciona una opción:', opciones)


if selection == "Red Interactiva":
    with open("grafo.py", encoding="UTF-8") as f:
        exec(f.read())
else:
    main()