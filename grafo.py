import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import random
import time
import math

def create_transport_graph(rows, cols):
    G = nx.grid_2d_graph(rows, cols)
    
    # Agregar desviaciones aleatorias a las posiciones de los nodos
    pos = {(x, y): (x + random.uniform(-0.5, 0.5), y + random.uniform(-0.5, 0.5)) for x, y in G.nodes()}

    # Definir pesos y grosores por fila y columna
    weights = [2, 4, 6, 8]  # Pesos disponibles
    thicknesses = [2, 4, 6, 8]  # Grosor disponible
    weight_dict = {}
    thickness_dict = {}

    # Asignar el mismo peso y grosor a todas las aristas en la misma columna
    for x in range(rows):
        weight_dict[x] = weights[x % len(weights)]
        thickness_dict[x] = thicknesses[x % len(thicknesses)]

    for y in range(cols):
        weight_dict[y + rows] = weights[y % len(weights)]
        thickness_dict[y + rows] = thicknesses[y % len(thicknesses)]

    # Asignar pesos y grosores a las aristas
    edge_weights = {}
    edge_thickness = {}
    for edge in G.edges():
        x0, y0 = edge[0]
        x1, y1 = edge[1]
        
        # Determinar el peso basado en fila o columna
        if x0 == x1:  # Arista vertical
            weight = weight_dict[x0]
            thickness = thickness_dict[x0]
        elif y0 == y1:  # Arista horizontal
            weight = weight_dict[y1 + rows]
            thickness = thickness_dict[y1 + rows]
        else:
            weight = 1  # Peso por defecto para aristas diagonales (si se usan)
            thickness = 1  # Grosor por defecto para aristas diagonales (si se usan)
        
        edge_weights[edge] = weight
        edge_thickness[edge] = thickness
        G[edge[0]][edge[1]]['weight'] = weight

    return G, pos, edge_weights, edge_thickness

def draw_graph(G, pos, edge_weights, edge_thickness, moving_objects):
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
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(str(node))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=text,
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

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
    return fig

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

    st.title("Red de Transporte ")
    st.sidebar.header("Configuración de la Red")
    rows = st.sidebar.number_input("Número de Filas", min_value=1, value=5)
    cols = st.sidebar.number_input("Número de Columnas", min_value=1, value=5)
    num_objects = st.sidebar.number_input("Cantidad de Objetos en Movimiento", min_value=1, value=3)
    if "graph" not in st.session_state or st.session_state.rows != rows or st.session_state.cols != cols:
        # Crear el grafo de la red de transporte
        G, pos, edge_weights, edge_thickness = create_transport_graph(rows, cols)
        st.session_state.graph = G
        st.session_state.pos = pos
        st.session_state.edge_weights = edge_weights
        st.session_state.edge_thickness = edge_thickness
        st.session_state.rows = rows
        st.session_state.cols = cols
    else:
        G = st.session_state.graph
        pos = st.session_state.pos
        edge_weights = st.session_state.edge_weights
        edge_thickness = st.session_state.edge_thickness
        
    agree = st.sidebar.checkbox("Dimensiones de los carriles")
    if agree:
        total_col = {}
        total_fil = {}

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
        
        fil = st.sidebar.checkbox("Dimensiones de las avenidas")
        if fil:
            st.sidebar.subheader("Pesos Totales por Columna")
            for col, total_weight in total_col.items():
                st.sidebar.write(f"Total de carriles de la avenida {col}: {total_weight}")

        col = st.sidebar.checkbox("Dimensiones de las calles")
        if col:
            st.sidebar.subheader("Pesos Totales por Fila")
            for fil, total_weight in total_fil.items():
                st.sidebar.write(f"Total de carriles de la calle {fil}: {total_weight}")

    # Opción para eliminar aristas
    st.sidebar.subheader("Eliminar Aristas")
    edges = list(G.edges())
    selected_edge = st.sidebar.selectbox("Selecciona una arista para eliminar:", edges, format_func=lambda e: f"{e[0]}-{e[1]}")
    
    if st.sidebar.button("Eliminar Arista"):
        if selected_edge in G.edges():
            G.remove_edge(*selected_edge)
            edge_weights.pop(selected_edge, None)
            edge_thickness.pop(selected_edge, None)
        else:
            st.error(f"Arista {selected_edge} no encontrada")
    # Opción para eliminar aristas
    st.sidebar.subheader("Eliminar Nodos")
    nodes = list(G.nodes())
    selected_node = st.sidebar.selectbox("Selecciona una arista para eliminar:", nodes)
    
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

    # Inicializar las posiciones de los objetos en movimiento en aristas aleatorias
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

    # Dibujar el grafo con los objetos en movimiento
    fig = draw_graph(G, pos, edge_weights, edge_thickness, moving_objects)
    graph_placeholder = st.empty()
    graph_placeholder.plotly_chart(fig)

    # Animar los objetos en movimiento
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
        fig = draw_graph(G, pos, edge_weights, edge_thickness, moving_objects)
        graph_placeholder.plotly_chart(fig)

        # Esperar un momento antes de la siguiente actualización
        time.sleep(2)
    
if __name__ == "__main__":
    main()