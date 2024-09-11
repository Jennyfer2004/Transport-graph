import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import random
import time
import math

def create_transport_graph(rows, cols,calle,ave, weights):
    G = nx.grid_2d_graph(rows, cols)
    
    # Agregar desviaciones aleatorias a las posiciones de los nodos
    pos = {(x, y): (x + random.uniform(-calle, calle), y + random.uniform(-ave, ave)) for x, y in G.nodes()}

    weight_dict = {}

    # Asignar el mismo peso y grosor a todas las aristas en la misma columna
    for x in range(rows):
        weight_dict[x] = weights[x % len(weights)]

    for y in range(cols):
        weight_dict[y + rows] = weights[y % len(weights)]

    # Asignar pesos y grosores a las aristas
    edge_weights = {}
    for edge in G.edges():
        x0, y0 = edge[0]
        x1, y1 = edge[1]
        
        # Determinar el peso basado en fila o columna
        if x0 == x1:  # Arista vertical
            weight = weight_dict[x0]
        elif y0 == y1:  # Arista horizontal
            weight = weight_dict[y1 + rows]
        else:
            weight = 1  # Peso por defecto para aristas diagonales (si se usan)
            thickness = 1  # Grosor por defecto para aristas diagonales (si se usan)
        
        edge_weights[edge] = weight
        G[edge[0]][edge[1]]['weight'] = weight

    return G, pos, edge_weights, edge_weights

def draw_graph(G, pos, edge_weights, edge_thickness,centrality_value=None ,moving_objects=None):
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        thickness = edge_thickness[edge]
        trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=thickness, color='#888'),
            hoverinfo='text',
            text=[f'Peso: {edge_weights[edge]}'],  # Mostrar peso al pasar el cursor
            mode='lines'
        )
        edge_traces.append(trace)

    node_x = []
    node_y = []
    text = []
    node_size=[]
    node_color=[]
    if centrality_value:
        max_centrality=max(centrality_value.values())
        min_centrality=min(centrality_value.values())
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            text.append(str(node))
            centrality =centrality_value[node]
            if centrality-min_centrality==0:
                node_color.append(0)
            else:
                node_color.append((centrality-min_centrality)/(max_centrality-min_centrality))
            #print(node_color)
            node_size.append(centrality*50)
    else:
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            text.append(str(node))
            node_size.append(10)
            node_color.append(0.5)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=text,
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            color=node_color,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    if moving_objects:
        moving_object_traces = []
        for edge, objects_on_edge in moving_objects.items():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            num_objects = len(objects_on_edge)
        
            for i, obj in enumerate(objects_on_edge):
            # Colocar los objetos uno detrás de otro en la misma arista
                ratio = (i + 1) / (num_objects + 1)  # Evitar solapamiento
                obj_x = x0 * (1 - ratio) + x1 * ratio
                obj_y = y0 * (1 - ratio) + y1 * ratio
            
                moving_object_trace = go.Scatter(
                    x=[obj_x],
                    y=[obj_y],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=obj['color']
                ),
                name=f'Moving Object {obj["id"]}'
                )
                moving_object_traces.append(moving_object_trace)

            fig = go.Figure(data=edge_traces + [node_trace] + moving_object_traces,
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )
    else:
        fig = go.Figure(data=edge_traces + [node_trace] ,
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )
    return fig

def calculate_centrality(G,nombre):
    if nombre =="Degree Centrality":
        return nx.degree_centrality(G)
    elif nombre=="Closeness Centrality" :
        return nx.closeness_centrality(G)
    elif nombre=="Betweenness Centrality":
        return nx.betweenness_centrality(G)
    elif nombre=="Eigenvector Centrality": 
        return nx.eigenvector_centrality(G)
    elif nombre=="Minimo Cenected Time":
        conection_times={}
        for node in G.nodes():
           lengths=nx.single_source_shortest_path_length(G,node)
           total_time=sum(lengths.values())
           conection_times[node]=0.3-(total_time/1000)
        return conection_times
    
def get_random_closest_edge(pos, current_node, previous_node):
    # Obtener las aristas adyacentes al nodo actual
    adjacent_edges = list(G.edges(current_node))

    possible_edges = []
   
    for edge in adjacent_edges:
        next_node = edge[1] if edge[0] == current_node else edge[0]

        if next_node == previous_node:
            if len(adjacent_edges)==1 :
                return edge
            continue
        possible_edges.append(edge)

    # Si hay opciones, elegir una al azar
    if possible_edges:
        return random.choice(possible_edges)
    else:
        return None


def main():
        global G  

        st.title("Explorando la Infraestructura Urbana")
        st.sidebar.header("Configuración de la Red")
        calle = st.sidebar.select_slider(
        "Selecciona el grado de desviación que van a tener las calles",
        options=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        value=0
        )    
        ave = st.sidebar.select_slider(
        "Selecciona el grado de desviación que van a tener las avenidas",
        options=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        value=0
        )
        input_text = st.sidebar.text_area(
        "Escribe las dimensiones que deseas para los carriles separados por comas:",
        value=", ".join(map(str, [2,4,6,8])),
        placeholder="Presiona Enter después de ingresar", height=1)

        if input_text:
            try:
                numbers = [float(x.strip()) for x in input_text.split(',')]
            except ValueError:
                st.sidebar.error("Por favor, introduce solo números separados por comas.")

        rows = st.sidebar.number_input("Elija la cantidad de Calles", min_value=1, value=5)
        cols = st.sidebar.number_input("Elija la cantidad de Avenidas", min_value=1, value=5)
        num_objects = st.sidebar.number_input("Elija la cantidad de Objetos en Movimiento", min_value=1, value=3)

        if "graph" not in st.session_state or st.session_state.rows != rows or st.session_state.cols != cols or st.session_state.ave != ave or st.session_state.calle != calle or st.session_state.numbers != numbers:
        # Crear el grafo de la red de transporte

            G, pos, edge_weights, edge_thickness = create_transport_graph(rows, cols,calle,ave,numbers)
            st.session_state.graph = G
            st.session_state.pos = pos
            st.session_state.edge_weights = edge_weights
            st.session_state.edge_thickness = edge_thickness
            st.session_state.rows = rows
            st.session_state.cols = cols
            st.session_state.ave = ave
            st.session_state.calle = calle
            st.session_state.numbers=numbers
        else:
            G = st.session_state.graph
            pos = st.session_state.pos
            edge_weights = st.session_state.edge_weights
            edge_thickness = st.session_state.edge_thickness
        st.sidebar.subheader("Opciones de centralidad")
        centrality_option = st.sidebar.selectbox("Selecciona el tipo de centralidad que desea visualizar :", ["Ninguna","Degree Centrality","Minimo Cenected Time","Betweenness Centrality","Closeness Centrality","Eigenvector Centrality"])

        if centrality_option:
            centrality_value =calculate_centrality(G,centrality_option)
            print(centrality_value)
        elif centrality_option=="Ninguna":
            centrality_value =None
        st.sidebar.write("Seleccione si desea ver :")
        agree = st.sidebar.checkbox("Dimensiones de los carriles")

# Variables para almacenar los totales
        total_col = {}
        total_fil = {}

# Verifica el estado del checkbox "agree"
        if agree:
    # Reinicia los totales cada vez que se selecciona "agree"
            total_col.clear()
            total_fil.clear()

            for edge, weight in edge_weights.items():
                x0, y0 = edge[0]
                x1, y1 = edge[1]
                if x0 == x1:
                    fil = x1  
                    if fil not in total_fil:
                        total_fil[fil] = weight
                elif y0 == y1:
                    col = y0  # Determinar la columna basada en la posición de las aristas
                    if col not in total_col:
                        total_col[col] = weight

    # Opción para dimensiones de las avenidas
            fil = st.sidebar.checkbox("Dimensiones de las avenidas")
            if fil:
                st.sidebar.subheader("Pesos Totales por Calle")
                for col, total_weight in total_col.items():
                    st.sidebar.write(f"Total de carriles de la avenida {col}: {total_weight}")

    # Opción para dimensiones de las calles
            col = st.sidebar.checkbox("Dimensiones de las calles")
            if col:
                st.sidebar.subheader("Pesos Totales por avenida")
                for fil, total_weight in total_fil.items():
                    st.sidebar.write(f"Total de carriles de la calle {fil}: {total_weight}")
        else:
            total_col.clear()
            total_fil.clear()

        # Opción para eliminar aristas
        st.sidebar.subheader("Eliminación de aristas")
        edges = list(G.edges())
        selected_edge = st.sidebar.selectbox("Selecciona una arista para eliminar:", edges, format_func=lambda e: f"{e[0]}-{e[1]}")

        if st.sidebar.button("Eliminar Arista"):
            if selected_edge in G.edges():
                G.remove_edge(*selected_edge)
                edge_weights.pop(selected_edge, None)
                edge_thickness.pop(selected_edge, None)
            else:
                st.error(f"Arista {selected_edge} no encontrada")

        # Opción para eliminar nodos
        st.sidebar.subheader("Eliminación de Nodos")
        nodes = list(G.nodes())
        selected_node = st.sidebar.selectbox("Selecciona un nodo para eliminar:", nodes)

        if st.sidebar.button("Eliminar Nodos"):
            if selected_node in G.nodes():
                G.remove_node(selected_node)
                edges_to_eliminate=list(G.edges(selected_node))
                for edge in edges_to_eliminate:
                    edge_weights.pop(edge, None)
                    edge_thickness.pop(edge, None)
                pos={node:pos[node] for node in G.nodes()}
                st.session_state.pos=pos
            else:
                st.error(f"Nodo {selected_node} no encontrado")

        # Dibujar el grafo con los objetos en movimiento
        fig = draw_graph(G, pos, edge_weights, edge_thickness, centrality_value)
        graph_placeholder = st.empty()
        graph_placeholder.plotly_chart(fig)

        st.sidebar.subheader("Visualización del Recorrido")
        if st.sidebar.button("Inicializar Movimiento"):
            moving_objects = {}
            for i in range(num_objects):
                edge = random.choice(list(G.edges()))
                if edge not in moving_objects:
                   moving_objects[edge] = []
                moving_objects[edge].append({
            'id': i,
            'pos': [(pos[edge[0]][0] + pos[edge[1]][0]) / 2, 
                    (pos[edge[0]][1] + pos[edge[1]][1]) / 2],
            'current_edge': edge,
            'previous_node': edge[0],
            'color': 'red' if i == 0 else f'rgba({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)}, 1)'
            })
            while True:
                new_moving_objects = {}
                for edge, objects_on_edge in moving_objects.items():
                    for obj in objects_on_edge:
                   
                   # Moverse a una arista adyacente aleatoria
                        closest_edge = get_random_closest_edge(pos, obj['current_edge'][1], obj['previous_node'])
                        if closest_edge is None:
                            continue
    
                        obj['previous_node'] = obj['current_edge'][1]
                        obj['current_edge'] = closest_edge

                        if closest_edge not in new_moving_objects:
                            new_moving_objects[closest_edge] = []
                        new_moving_objects[closest_edge].append(obj)

                moving_objects = new_moving_objects

            # Actualizar el grafo
                fig = draw_graph(G, pos, edge_weights, edge_thickness,centrality_value, moving_objects)
                graph_placeholder.plotly_chart(fig)

            # Esperar un momento antes de la siguiente actualización
                time.sleep(2)

if __name__ == "__main__":
    main()