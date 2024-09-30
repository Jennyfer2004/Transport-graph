import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import plotly.graph_objects as go
import time
import itertools

# Crear un grafo de cuadrícula y conectarlos con n aristas
def create_connected_grid_graphs(rows1, cols1, rows2, cols2, num_edges, row_deviation1=0, col_deviation1=0, row_deviation2=0, col_deviation2=0):
    G1 = nx.grid_2d_graph(rows1, cols1)  # Primer grafo de cuadrícula
    G2 = nx.grid_2d_graph(rows2, cols2)  # Segundo grafo de cuadrícula

    max_col1 = max([node[1] for node in G1.nodes()])
    mapping = {node: (node[0], node[1] + max_col1 + 2) for node in G2.nodes()}  # Desplazar el segundo grafo
    G2 = nx.relabel_nodes(G2, mapping)
    pos1 = {(x, y): (x + random.uniform(-row_deviation1, row_deviation1), y + random.uniform(-col_deviation1, col_deviation1)) for x, y in G1.nodes()}
    pos2 = {(x, y): (x + random.uniform(-row_deviation2, row_deviation2), y + random.uniform(-col_deviation2, col_deviation2)) for x, y in G2.nodes()}

    # Crear un grafo combinado
    G = nx.compose(G1, G2)

    # Conectar los grafos con n aristas
    edges_added = 0
    G1_nodes = [node for node in G1.nodes() if node[1] == cols1 - 1]  # Nodos del borde derecho de G1
    G2_nodes = [node for node in G2.nodes() if node[1] == max_col1 + 2]  # Nodos del borde izquierdo de G2
    
    while edges_added < num_edges:
        node1 = random.choice(G1_nodes)  # Seleccionar un nodo al azar del borde derecho de G1
        node2 = random.choice(G2_nodes)  # Seleccionar un nodo al azar del borde izquierdo de G2
        if not G.has_edge(node1, node2):  # Evitar aristas duplicadas
            G.add_edge(node1, node2)
            edges_added += 1
    pos={**pos1, **pos2}
    
    return G,pos

def draw_graph(G, pos, edge_weights, edge_thickness,node_semaforo_status={},centrality_value=None ,moving_objects=None,vel_edges_bool={}):
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
            if node in node_semaforo_status:
                text.append(str(node) + "     " + node_semaforo_status[node])
            else:
                text.append(str(node))
            centrality =centrality_value[node]
            if centrality-min_centrality==0:
                node_color.append(0)
            else:
                node_color.append((centrality-min_centrality)/(max_centrality-min_centrality))
            node_size.append(centrality*50)
    else:
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            if node in node_semaforo_status:
                text.append(str(node) + "     " + node_semaforo_status[node])
            else:
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
                ratio = (i + 1) / (num_objects + 1)  
                obj_x = x0 * (1 - ratio) + x1 * ratio
                obj_y = y0 * (1 - ratio) + y1 * ratio
                if obj['id'] in vel_edges_bool:
                    if vel_edges_bool[(obj['id'],edge)]==False:
                        print(obj_x,obj_y)
                        obj_x=x0 * (1 - 0.1) + x1 * 0.1
                        obj_y=y0 * (1 - 0.1) + y1 * 0.1
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

def gird_2d(rows, cols,calle,ave, weights=[2,4,6,8]):
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
            weight = 1  
            
        edge_weights[edge] = weight
        G[edge[0]][edge[1]]['weight'] = weight

    return G, pos, edge_weights, edge_weights
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

def get_random_closest_edge(G, current_node, previous_node,edges,vias,bol,stop,vel=True,vel_cols=True,vel_edge_bool=True):
    # Obtener las aristas adyacentes al nodo actual
    adjacent_edges = list(G.edges(current_node))
    possible_edges = []
   
    for edge in adjacent_edges:
        if bol:
            estado_de_edge = G.edges[edge]['estado']
            if edge[0] == current_node and estado_de_edge:
                next_node = edge[1]
            elif estado_de_edge:
                next_node = edge[0]
            else:
                continue
        else:
             next_node = edge[1] if edge[0] == current_node else edge[0]

        if next_node == previous_node:
            if len(adjacent_edges)==1 :
                return edge
            continue
        possible_edges.append(edge)
    if not vel or not vel_cols or not vel_edge_bool:
        return edges
    if vias or not stop or not vel:
        return edges

    # Si hay opciones, elegir una al azar
    if possible_edges:
        return random.choice(possible_edges)
    else:
        return edges
    
def inicializar_grafo_con_aristas(G,arist):
    # Obtén la lista de todas las aristas
    aristas = list(G.edges())
    for i, edge in enumerate(aristas):
        if i in arist:
            G.edges[edge]['estado'] = False  # Verde
        else:
            G.edges[edge]['estado'] = True  # Rojo

def ciclo_arista():
    # Alterna entre los estados de la arista: True (Verde) y False (Rojo)
    estados = itertools.cycle([True, False])
    while True:
        yield next(estados)

def update_estado_aristas(G, arista_estado_generadores):
    for arista, estado_gen in arista_estado_generadores.items():
        G.edges[arista]['estado'] = next(estado_gen)  # Cambia el estado de la arista a True o False

def inicializar_generadores_aristas(G):
    
    return {
        edge: ciclo_arista() for edge in G.edges()
    }

def generate_graph_witrh_weights(G,weights):
    edge_weights={}
    edge_thickness={}

    for edge in G.edges():
        weight=random.choice(weights)
        edge_weights[edge]=weight
        edge_thickness[edge]=weight
    return edge_weights,edge_thickness

def expand_city(G, expansion_prob):
    nodes_to_expand = list(G.nodes)
    new_edges = []

    for node in nodes_to_expand:
        # Buscar vecinos fuera del límite actual del grafo
        neighbors = [(node[0]+1, node[1]), (node[0]-1, node[1]), 
                     (node[0], node[1]+1), (node[0], node[1]-1)]
        
        for neighbor in neighbors:
            if neighbor not in G.nodes and random.random() < expansion_prob:
                new_edges.append((node, neighbor))
    
    G.add_edges_from(new_edges)
    return new_edges
def main():
    global G 
    st.title('Generador de Redes de Transporte')
    st.sidebar.title("Interacción con la Red")
# Selección del tipo de grafo

    grafo_tipo = st.sidebar.selectbox(
    'Selecciona el tipo de red de transporte que desee:',
    [ 'Grid 2D Graph',"Double Grid 2D Graph"]
)
    
    st.sidebar.header("Configuración de la Red")

    input_text = st.sidebar.text_area(
        "Escribe las dimensiones que deseas para los carriles separados por comas:",
        value=", ".join(map(str, [2,4,6,8])),
        placeholder="Presiona Enter después de ingresar", height=1)
    if input_text:
            try:
                numbers = [float(x.strip()) for x in input_text.split(',')]
            except ValueError:
                numbers = [2,4,6,8]
                st.sidebar.error("Por favor, introduce solo números separados por comas.")

    node_semaforo_status = {}
        
    if grafo_tipo == 'Grid 2D Graph':

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


        rows = st.sidebar.number_input("Elija la cantidad de Calles", min_value=1, value=5)
        cols = st.sidebar.number_input("Elija la cantidad de Avenidas", min_value=2, value=5)

        if "graph" not in st.session_state or (st.session_state.numbers != numbers or st.session_state.rows != rows or st.session_state.cols != cols or st.session_state.ave != ave or st.session_state.calle != calle) and grafo_tipo == 'Grid 2D Graph':
            G, pos, edge_weights, edge_thickness = gird_2d(rows, cols,calle,ave,numbers)
            st.session_state.rows = rows
            st.session_state.cols = cols
            st.session_state.ave = ave
            st.session_state.calle = calle
            st.session_state.node_semaforo_status=node_semaforo_status
            st.session_state.rows1 = 1
            st.session_state.rows2 = 1
            st.session_state.cols1 = 1
            st.session_state.cols2 = 1
            st.session_state.num_edges = 1    
            st.session_state.row_dev_1 = 1
            st.session_state.row_dev_2 = 1
            st.session_state.col_dev_1 = 1
            st.session_state.col_dev_2 = 1
            st.session_state.pos = pos
            st.session_state.edge_weights = edge_weights
            st.session_state.edge_thickness = edge_thickness
            st.session_state.numbers=numbers
            st.session_state.graph = G
            
        else:
            G = st.session_state.graph
            pos = st.session_state.pos
            edge_weights = st.session_state.edge_weights
            edge_thickness = st.session_state.edge_thickness
        
        num_objects = st.sidebar.number_input("Elija la cantidad de Objetos en Movimiento", min_value=1, value=10)


    elif grafo_tipo == 'Double Grid 2D Graph':

        rows1 = st.sidebar.number_input("Elija la cantidad de Calles del primer grafo", min_value=1, value=3)
        cols1 = st.sidebar.number_input("Elija la cantidad de Avenidas del primer grafo", min_value=1, value=3)

        rows2 = st.sidebar.number_input("Elija la cantidad de Calles del segundo grafo", min_value=1, value=4)
        cols2 = st.sidebar.number_input("Elija la cantidad de Avenidas del segundo grafo", min_value=1, value=4)

        num_edges = st.sidebar.slider("Número de aristas entre los grafos", min_value=1, max_value=10, value=1)
        
        row_dev_1 = st.sidebar.select_slider(
        "Selecciona el grado de desviación que van a tener las calles del primer grafo",
        options=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        value=0
        )    
        col_dev_1 = st.sidebar.select_slider(
        "Selecciona el grado de desviación que van a tener las avenidas del primer grafo",
        options=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        value=0
        )
        row_dev_2 = st.sidebar.select_slider(
        "Selecciona el grado de desviación que van a tener las calles del segundo grafo",
        options=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        value=0
        )    
        col_dev_2 = st.sidebar.select_slider(
        "Selecciona el grado de desviación que van a tener las avenidas del segundo grafo",
        options=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        value=0
        )
        
        if ("graph" not in st.session_state or st.session_state.num_edges != num_edges or st.session_state.row_dev_1 != row_dev_1 or st.session_state.row_dev_2 != row_dev_2 or st.session_state.col_dev_1 != col_dev_1 or st.session_state.col_dev_2 != col_dev_2  or st.session_state.rows1 != rows1 or st.session_state.rows2 != rows2 or st.session_state.cols2 != cols2 or st.session_state.cols1 != cols1) :
            st.session_state.rows1 = rows1
            st.session_state.rows2 = rows2
            st.session_state.cols1 = cols1
            st.session_state.cols2 = cols2
            st.session_state.row_dev_1 = row_dev_1
            st.session_state.row_dev_2 = row_dev_2
            st.session_state.col_dev_1 = col_dev_1
            st.session_state.col_dev_2 = col_dev_2
            st.session_state.num_edges = num_edges
            st.session_state.node_semaforo_status=node_semaforo_status
            st.session_state.rows = 1
            st.session_state.cols = 1
            st.session_state.ave = 1
            st.session_state.calle = 1
            G,pos = create_connected_grid_graphs(rows1, cols1, rows2, cols2, num_edges, row_dev_1, col_dev_1, row_dev_2, col_dev_2)
            st.session_state.graph = G
            st.session_state.pos = pos
            st.session_state.numbers=numbers
            edge_weights,edge_thickness = generate_graph_witrh_weights(G,numbers)
            st.session_state.edge_weights = edge_weights
            st.session_state.edge_thickness = edge_thickness

        else:
            G = st.session_state.graph
            pos = st.session_state.pos
            edge_weights = st.session_state.edge_weights
            edge_thickness = st.session_state.edge_thickness
        
        num_objects = st.sidebar.number_input("Elija la cantidad de Objetos en Movimiento", min_value=1, value=3)
     
    st.sidebar.subheader("Añadir una arista")
    nodes = sorted(list(G.nodes()))

# Seleccionar dos nodos para crear una nueva arista
    node1 = st.sidebar.selectbox("Selecciona la primera intersección:", nodes)
    node2 = st.sidebar.selectbox("Selecciona la segunda intersección:", nodes)
    peso = st.sidebar.text_area(
        "Escribe el grosor que tendrá ",
        value=str(2),
        placeholder="Presiona Enter después de ingresar", height=1)
    
# Añadir la arista si no existe
    if st.sidebar.button("Añadir Arista"):
        if G.has_edge(node1, node2):
            st.warning(f"La arista entre {node1} y {node2} ya existe")
        else:
            try:
                G.add_edge(node1, node2)
                edge_weights[(node1, node2)] = float(peso)
                edge_thickness[(node1, node2)] = float(peso) 
            except:
                G.add_edge(node2, node1)
                edge_weights[(node2, node1)] = float(peso)
                edge_thickness[(node2, node1)] = float(peso) 
        for i in edge_thickness:
            print(type(edge_thickness[i]))
    st.sidebar.subheader("Eliminación de aristas")
    edges = list(G.edges())
    selected_edge = st.sidebar.selectbox("Selecciona una arista para eliminar:", edges, format_func=lambda e: f"{e[0]}-{e[1]}")

    if st.sidebar.button("Eliminar Arista"):
            if selected_edge in G.edges():
                G.remove_edge(*selected_edge)
                edge_weights.pop(selected_edge, None)
                edge_thickness.pop(selected_edge, None)
                st.session_state.graph=G
            else:
                st.error(f"Arista {selected_edge} no encontrada")

        # Opción para eliminar nodos
    st.sidebar.subheader("Eliminación de Intersecciones")
    nodes = sorted(list(G.nodes()))
    selected_node = st.sidebar.selectbox("Selecciona una intersección para eliminar:", nodes)

    if st.sidebar.button("Eliminar Intersección"):
            if selected_node in G.nodes():
                G.remove_node(selected_node)
                if grafo_tipo == 'Grid 2D Graph':
                    edges_to_eliminate=list(G.edges(selected_node))
                    for edge in edges_to_eliminate:
                        edge_weights.pop(edge, None)
                        edge_thickness.pop(edge, None)
                pos={node:pos[node] for node in G.nodes()}
                st.session_state.pos=pos
            else:
                st.error(f"Intersección {selected_node} no encontrado")

    
    def actualizar_estado_aristas():
            update_estado_aristas(G, st.session_state.arista_estado_generadores)
            
    st.sidebar.subheader("Opción de expansión")
    n = st.sidebar.slider('Seleccione cuantas veces desea que se expanda' , value=0)
    expansion_probability = st.sidebar.slider('Seleccione lo probabilidad de que aparezcan caminos nuevos', value=0.3 )
   
    if not "n" in st.session_state:
        st.session_state.n=n
    if not "numbers_edge" in st.session_state:
        numbers_edge={}
        st.session_state.numbers_edge=numbers_edge
    if n!= st.session_state.n :
        st.session_state.n=n
        if grafo_tipo == 'Double Grid 2D Graph':

            for _ in range(n):  # Expandir 10 veces
                numbers_edge=expand_city(G, expansion_probability)
                if 'Double Grid 2D Graph' in st.session_state.numbers_edge:
                    st.session_state.numbers_edge['Double Grid 2D Graph']+=numbers_edge
                else:
                    st.session_state.numbers_edge['Double Grid 2D Graph']=numbers_edge
            calles=random.choice([col_dev_1,col_dev_2])
            aves=random.choice([row_dev_1,row_dev_2])
            for node in G.nodes():
                if node not in pos:
                    pos[node]=(node[0] + random.uniform(-calles, calles), node[1] + random.uniform(-aves, aves))
            for edge in G.edges():
                if edge not in edge_weights:
                    edge_weights[edge]=random.choice(numbers)
                    edge_thickness[edge]=random.choice(numbers)
            
        elif grafo_tipo =='Grid 2D Graph':
            for _ in range(n):  # Expandir 10 veces
                numbers_edge=expand_city(G, expansion_probability)
                if 'Grid 2D Graph' in st.session_state.numbers_edge:
                    st.session_state.numbers_edge['Grid 2D Graph']+=numbers_edge
                else:
                    st.session_state.numbers_edge['Grid 2D Graph']=numbers_edge
            for node in G.nodes():
                if node not in pos:
                    pos[node]=(node[0] + random.uniform(-calle, calle), node[1] + random.uniform(-ave, ave))
            for edge in G.edges():
                if edge not in edge_weights:
                    edge_weights[edge]=random.choice([2, 4])
                    edge_thickness[edge]=random.choice([2, 4])
        st.session_state.graph=G
       
    st.sidebar.subheader("Opciones de centralidad")
    centrality_option = st.sidebar.selectbox("Selecciona el tipo de centralidad que desea visualizar :", ["Ninguna","Degree Centrality","Minimo Cenected Time","Betweenness Centrality","Closeness Centrality","Eigenvector Centrality"])

    if centrality_option:
        centrality_value =calculate_centrality(G,centrality_option)
    elif centrality_option=="Ninguna":
        centrality_value =None

    st.sidebar.subheader("Opciones de señales de transito") 

    all_edges = list(G.edges())
    selected_edges_vel = st.sidebar.multiselect("Introduce los tramos de reducción de velocidad que desea", all_edges, default=[])
        
    all_nodes =sorted(list(G.nodes()))
    filtered_nodes = [node for node in all_nodes if node!=(0,0) and node!=(0,1)]
    selected_node_s = st.sidebar.multiselect("Introduce la intersección de semaforos que desee", filtered_nodes, default=[])
    if selected_node_s:
            for i in selected_node_s:
                node_semaforo_status[i]= "semáforo"

                arist=list(G.edges(i))
                inicializar_grafo_con_aristas(G,arist)
                st.session_state.node_semaforo_status=node_semaforo_status
                st.session_state.arista_estado_generadores = inicializar_generadores_aristas(G)

    filtered_nodes = [node for node in all_nodes if node not in selected_node_s]

    selected_ceda_rows= st.sidebar.multiselect("Introduce la intersección de ceda el paso que desea para las avenidas", filtered_nodes, default=[])
    if selected_ceda_rows:
            for i in selected_ceda_rows:                
                if i in st.session_state.node_semaforo_status:
                    st.session_state.node_semaforo_status[i]== "ceda_el_paso(calle)"
                    node_semaforo_status[i]= "ceda_el_paso"
                    st.session_state.node_semaforo_status=node_semaforo_status
                    continue
                node_semaforo_status[i]= "ceda_el_paso(ave)"
                st.session_state.node_semaforo_status=node_semaforo_status
    selected_ceda_cols= st.sidebar.multiselect("Introduce la intersección de ceda el paso que desea para las calles", filtered_nodes, default=[(0,1)])
    if selected_ceda_cols:
            for i in selected_ceda_cols:
                if i in st.session_state.node_semaforo_status:
                    st.session_state.node_semaforo_status[i]== "ceda_el_paso(ave)"
                    node_semaforo_status[i]= "ceda_el_paso"
                    st.session_state.node_semaforo_status=node_semaforo_status
                    continue
                node_semaforo_status[i]= "ceda_el_paso(calle)"
                st.session_state.node_semaforo_status=node_semaforo_status

    filtered_nodes = [node for node in filtered_nodes if node not in selected_ceda_rows and node not in selected_ceda_cols ]
    selected_stop= st.sidebar.multiselect("Introduce la intersección de pare que desea", filtered_nodes, default=[(0,0)])
    if selected_stop:
            for i in selected_stop:
                node_semaforo_status[i]= "pare"
                st.session_state.node_semaforo_status=node_semaforo_status

    
         #   st.seasion_state.edge_status=edge_status

   
    fig = draw_graph(G, pos, edge_weights, edge_thickness,node_semaforo_status, centrality_value)
    graph_placeholder = st.empty()
    graph_placeholder.plotly_chart(fig)
    if "moving" not in st.session_state:
            st.session_state.moving=False
    st.sidebar.subheader("Visualización del Recorrido")
    ready=st.sidebar.button("Inicializar Movimiento")      
    closed=st.sidebar.button("Detener Movimiento")

    
    estancamientos=[]
            
    if ready or st.session_state.moving:
            if "closed" not in st.session_state:
                st.session_state.closed=True
            else:
                st.session_state.closed=True

            if not st.session_state.moving:
                st.session_state.moving=True
            if "moving_objects" not in st.session_state:
                st.session_state.moving_objects = {}
            moving_objects = st.session_state.moving_objects  # Cargar los objetos ya existentes
            
            current_num_objects = sum(len(objects) for objects in moving_objects.values())  # Total de objetos actuales
            if current_num_objects > num_objects:
                all_objects = [(edge, obj) for edge, objects_on_edge in moving_objects.items() for obj in objects_on_edge]
                 
                all_objects.sort(key=lambda x: x[1]['id']) 
    
    # Eliminar los objetos excedentes
                for _ in range(current_num_objects - num_objects):
                    edge, obj = all_objects.pop()  # Eliminar el último objeto de la lista (puedes ajustar la lógica si lo necesitas)
                    moving_objects[edge].remove(obj)  # Remover el objeto de la arista correspondiente
                    if not moving_objects[edge]:  # Si la arista ya no tiene objetos, eliminar la entrada
                       del moving_objects[edge]

            for i in range(current_num_objects, num_objects):
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
            st.session_state.moving_objects = moving_objects  # Guardar el nuevo estado
            
            count=0
            stop={}
            vel={}
            vel_cols={}
            vel_edges_bool={}
            visited_edges={}
            cc=0
            while st.session_state.moving and not closed and st.session_state.closed:
                new_moving_objects = {}
                
                for edge, objects_on_edge in moving_objects.items():
                    k=[]
                    for obj in objects_on_edge:
                        if obj['id'] not in visited_edges:
                            visited_edges[obj['id']]=[]
                            visited_edges[obj['id']].append({"count":1,"edge":edge})
                        elif visited_edges[obj['id']][0]["edge"]==edge:
                            counter=visited_edges[obj['id']][0]["count"]
                            visited_edges[obj['id']][0]["count"]=counter+1
                        else:
                            visited_edges[obj['id']]=[{"count":1,"edge":edge}]
                            
                        if visited_edges[obj['id']][0]["count"]>=3 :
                            
                            if len(estancamientos)>=1:
                                a=0
                                for i in estancamientos:
                                    if i[0]==obj['id'] and i[1]==edge:
                                        estancamientos[estancamientos.index(i)][2]+=1
                                        a+=1
                                if a==0:
                                    estancamientos.append([obj['id'],edge,visited_edges[obj['id']][0]["count"]])
                            else:
                                estancamientos.append([obj['id'],edge,visited_edges[obj['id']][0]["count"]])

                        if not obj['id'] in vel:
                            vel[obj['id']]=True
                        
                        if not obj['id'] in vel_cols:
                            vel_cols[obj['id']]=True

                        if (obj['id'],edge) not in vel_edges_bool:
                            vel_edges_bool[(obj['id'],edge)]=True                       
                        if edge in selected_edges_vel:
                                if vel_edges_bool[(obj['id'],edge)]==True:
                                    vel_edges_bool[(obj['id'],edge)]=False
                                elif vel_edges_bool[(obj['id'],edge)]==False:
                                    vel_edges_bool[(obj['id'],edge)]=True
                        
                        if edge[1] in node_semaforo_status:
                            if node_semaforo_status[edge[1]]=="ceda_el_paso(ave)" and vel_edges_bool[(obj['id'],edge)]:
                                adjacent_edges = [(u, v) for u, v in G.edges(edge[1]) if u[1] == i and v[1] == i]
                                for i in adjacent_edges:
                                    j=i[1]
                                    if edge[1][0]== obj['previous_node'][0] :
                                        if edge[1][1]!=j[1]:
                                            if i in moving_objects and moving_objects[i]:
                                               
                                               k.append(i)
                                    elif edge[1][1]== obj['previous_node'][1] :
                                        if edge[1][0]==j[0]:
                                            continue
                                        else:
                                            if i in moving_objects and moving_objects[i]:
                                               k.append(i)
                            elif node_semaforo_status[edge[1]]=="ceda_el_paso" and vel_edges_bool[(obj['id'],edge)]:
                                adjacent_edges = list(G.edges(edge[1]))
                                for i in adjacent_edges:
                                    j=i[1]
                                    if edge[1][0]== obj['previous_node'][0] :
                                        if edge[1][1]!=j[1]:
                                            if i in moving_objects and moving_objects[i]:
                                               
                                               k.append(i)
                                    elif edge[1][1]== obj['previous_node'][1] :
                                        if edge[1][0]==j[0]:
                                            continue
                                        else:
                                            if i in moving_objects and moving_objects[i]:
                                               k.append(i)
                            elif node_semaforo_status[edge[1]]=="ceda_el_paso(calle)" and vel_edges_bool[(obj['id'],edge)]:
                                adjacent_edges = [(u, v) for u, v in G.edges(edge[1]) if u[0] == i and v[0] == i]
                     
                                for i in adjacent_edges:
                                    j=i[1]
                                    if edge[1][0]== obj['previous_node'][0] :
                                        if edge[1][1]!=j[1]:
                                            if i in moving_objects and moving_objects[i]:
                                               
                                               k.append(i)
                                    elif edge[1][1]== obj['previous_node'][1] :
                                        if edge[1][0]==j[0]:
                                            continue
                                        else:
                                            if i in moving_objects and moving_objects[i]:
                                               k.append(i)
                        print(moving_objects)
                        print()
                        if not obj['id'] in stop:
                             stop[obj['id']]=True
                        if selected_stop and edge[1] in node_semaforo_status and node_semaforo_status[edge[1]]=="pare" and vel_edges_bool[(obj['id'],edge)]:
                            if stop[obj['id']]==False:
                                stop[obj['id']]=True
                            elif stop[obj['id']]==True:
                                stop[obj['id']]=False

                        if selected_node_s :
                                closest_edge = get_random_closest_edge(G, obj['current_edge'][1], obj['previous_node'],edge,k,True,stop[obj['id']],vel[obj['id']],vel_cols[obj['id']],vel_edges_bool[(obj['id'],edge)])
                        else:
                                closest_edge = get_random_closest_edge(G, obj['current_edge'][1], obj['previous_node'],edge,k,False,stop[obj['id']],vel[obj['id']],vel_cols[obj['id']],vel_edges_bool[(obj['id'],edge)])
                        
                        if closest_edge==obj['current_edge'] :
                            if closest_edge not in new_moving_objects:
                                new_moving_objects[closest_edge] = []
                            new_moving_objects[closest_edge].append(obj)
                            continue
                        if closest_edge==obj['current_edge'] :
                            print("no")
                        obj['previous_node'] = obj['current_edge'][1]
                        obj['current_edge'] = closest_edge
                      
                        if closest_edge not in new_moving_objects:
                            new_moving_objects[closest_edge] = []
                        new_moving_objects[closest_edge].append(obj)

                moving_objects = new_moving_objects
        # Actualizar el grafo
                fig = draw_graph(G, pos, edge_weights, edge_thickness,node_semaforo_status, centrality_value, moving_objects,vel_edges_bool)
                graph_placeholder.plotly_chart(fig)

        # Esperar un momento antes de la siguiente actualización
                time.sleep(2)                
                count+=1
                if count%3==0 and selected_node_s:
                    actualizar_estado_aristas() 
                if not "estancamientos" in st.session_state:
                    st.session_state.estancamientos=estancamientos
                elif estancamientos!=[]:
                    for i in estancamientos:
                        if i not in st.session_state.estancamientos:
                            st.session_state.estancamientos.append(i)
            if closed:
                if "closed" not in st.session_state:
                    st.session_state.closed=False
                st.session_state.moving_objects={}
                moving_objects={}
                edges_stop=[]
                count=0
                if grafo_tipo in st.session_state.numbers_edge:
                    st.write(f"La Red de Transporte se ha extendidio con un total de {len(st.session_state.numbers_edge[grafo_tipo])} caminos nuevos")
                
                for i in st.session_state.estancamientos:
                    count+=1
                    if i[1] not in edges_stop :
                        edges_stop.append(i[1])
                if st.session_state.estancamientos==[]:
                    st.write("No han ocurrido estancamientos significativos del tráfico")
                elif len(edges_stop)==1 :
                    if count==1:
                        st.write(f"Ha ocurrido {len(edges_stop)} estancamiento y se ha detenido {count} vehiculo")
                    else:
                        st.write(f"Ha ocurrido {len(edges_stop)} estancamiento y se han detenido {count} vehiculos")
                       
                else:
                    st.write(f"Han ocurrido {len(edges_stop)} estancamientos y se han detenido {count} vehiculos")
                st.session_state.estancamientos=[]
                
if __name__ =="__main__":
    main()