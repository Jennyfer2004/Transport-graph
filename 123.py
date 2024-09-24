import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import random
import time
import math
import itertools

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

def draw_graph(G, pos, edge_weights, edge_thickness,node_semaforo_status,centrality_value=None ,moving_objects=None):
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
            #text.append(node_semaforo_status[node])
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
    
def get_random_closest_edge(G, current_node, previous_node,edges,vias,bol,stop,vel=True,vel_cols=True):
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
    if not vel or not vel_cols:
        return edges
    if vias or not stop or not vel:
        return edges

    # Si hay opciones, elegir una al azar
    if possible_edges:
        return random.choice(possible_edges)
    else:
        return edges

def assign_semaforos_to_nodes(G,percent):
    total_nodes=len(G.nodes())
    sum_semaforos= int((percent*total_nodes)/100)
    semaforo_nodes= random.sample(G.nodes(),sum_semaforos)
    return semaforo_nodes

def semaforo(nodo_semaforo_status):
    for i in nodo_semaforo_status:
        arist=list(G.edges(i))
        sum_arist=len(arist)
        n=sum_arist*5
        for j in arist:

            print(j)


    return

def update_semaforo(nodo_semaforo_status):
    for nodo, semaforo_gen in nodo_semaforo_status.items():
        nodo_semaforo_status[nodo] = next(semaforo_gen)
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

# Función para actualizar el estado de las aristas
def update_estado_aristas(G, arista_estado_generadores):
    for arista, estado_gen in arista_estado_generadores.items():
        G.edges[arista]['estado'] = next(estado_gen)  # Cambia el estado de la arista a True o False

# Función para inicializar el generador de estados para cada arista
def inicializar_generadores_aristas(G):
    
    return {
        edge: ciclo_arista() for edge in G.edges()
    }
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
        node_semaforo_status = {}
        edge_status={}
        if "graph" not in st.session_state or st.session_state.rows != rows or st.session_state.cols != cols or st.session_state.ave != ave or st.session_state.calle != calle or st.session_state.numbers != numbers:
        # Crear el grafo de la red de transporte
            G, pos, edge_weights, edge_thickness = create_transport_graph(rows, cols,calle,ave,numbers)
            st.session_state.graph = G
            st.session_state.pos = pos
            st.session_state.node_semaforo_status=node_semaforo_status
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
        
        def actualizar_estado_aristas():
            update_estado_aristas(G, st.session_state.arista_estado_generadores)
               
        selected_node_s = st.sidebar.multiselect("Introduce el nodo de semaforos que desea", list(G.nodes()), default=[])
        if selected_node_s:
            for i in selected_node_s:
                node_semaforo_status[i]= "semáforo"

                arist=list(G.edges(i))
                inicializar_grafo_con_aristas(G,arist)
                st.session_state.node_semaforo_status=node_semaforo_status
                st.session_state.arista_estado_generadores = inicializar_generadores_aristas(G)
        all_nodes = list(G.nodes())
        filtered_nodes = [node for node in all_nodes if node not in selected_node_s]

        selected_ceda_rows= st.sidebar.multiselect("Introduce el nodo de ceda el paso que desea para las avenidas", filtered_nodes, default=[(1,1)])
        if selected_ceda_rows:
            for i in selected_ceda_rows:                
                if i in st.session_state.node_semaforo_status:
                    st.session_state.node_semaforo_status[i]== "ceda_el_paso(calle)"
                    node_semaforo_status[i]= "ceda_el_paso"
                    st.session_state.node_semaforo_status=node_semaforo_status
                    continue
                node_semaforo_status[i]= "ceda_el_paso(ave)"
                st.session_state.node_semaforo_status=node_semaforo_status
        selected_ceda_cols= st.sidebar.multiselect("Introduce el nodo de ceda el paso que desea para las calles", filtered_nodes, default=[(0,1)])
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
        selected_stop= st.sidebar.multiselect("Introduce el nodo de pare que desea", filtered_nodes, default=[(0,0)])
        if selected_stop:
            for i in selected_stop:
                node_semaforo_status[i]= "pare"
                st.session_state.node_semaforo_status=node_semaforo_status

        if "nodos_fila" not in st.session_state:
            st.session_state.nodos_fila= []

        selected_vel= st.sidebar.multiselect("Introduce las avenidas que desea restringir la velocidad", list(range(cols)))
        if selected_vel:
            #print(selected_vel)
            for i in selected_vel:
                for j in range(rows):
                    aristas_fila = [(u, v) for u, v in G.edges() if u[1] == i and v[1] == i]
                    st.session_state.nodos_fila=aristas_fila

        if "nodos_cols" not in st.session_state:
            st.session_state.nodos_cols= []

        selected_vel_row= st.sidebar.multiselect("Introduce las calles que desea restringir la velocidad", list(range(rows)))
        if selected_vel_row:
            for i in selected_vel_row:
                for j in range(cols):
                    edge_cols = [(u, v) for u, v in G.edges() if u[0] == i and v[0] == i]
                    st.session_state.nodos_cols=edge_cols
            print(st.session_state.nodos_cols)
         #   st.seasion_state.edge_status=edge_status
        
        if centrality_option:
            centrality_value =calculate_centrality(G,centrality_option)
            
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
                st.session_state.graph = G
                st.session_state.edge_weights = edge_weights
                st.session_state.edge_thickness = edge_thickness
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
                pos.pop(selected_node)
                #pos={node:pos[node] for node in G.nodes()}
                st.session_state.pos=pos
            else:
                st.error(f"Nodo {selected_node} no encontrado")

        #semaforo_nodes=assign_semaforos_to_nodes(G,percent_semaforos)

        # Dibujar el grafo con los objetos en movimiento
        fig = draw_graph(G, pos, edge_weights, edge_thickness,node_semaforo_status, centrality_value)
        graph_placeholder = st.empty()
        graph_placeholder.plotly_chart(fig)
        
        if "moving" not in st.session_state:
            st.session_state.moving=False
        st.sidebar.subheader("Visualización del Recorrido")
        ready=st.sidebar.button("Inicializar Movimiento")      
        closed=st.sidebar.button("Detener Movimiento")

        if ready or st.session_state.moving:
            if not st.session_state.moving:
                st.session_state.moving=True
            if "moving_objects" not in st.session_state:
                st.session_state.moving_objects = {}
            moving_objects = st.session_state.moving_objects  # Cargar los objetos ya existentes
            
            # Añadir nuevos objetos sin sobrescribir los existentes
            # Crear objetos en movimiento solo si hay menos de num_objects en total
            current_num_objects = sum(len(objects) for objects in moving_objects.values())  # Total de objetos actuales
            if current_num_objects > num_objects:
    # Obtener una lista con todos los objetos en movimiento
                all_objects = [(edge, obj) for edge, objects_on_edge in moving_objects.items() for obj in objects_on_edge]
    
    # Ordenarlos por el 'id' o de alguna manera lógica, por ejemplo
                all_objects.sort(key=lambda x: x[1]['id'])  # Ordenar por ID de objeto (puedes cambiar esto si es necesario)
    
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
            print(moving_objects)
    # Mover los objetos (misma lógica que antes)
            count=0
            stop={}
            vel={}
            vel_cols={}
            while st.session_state.moving and not closed:
                new_moving_objects = {}

                for edge, objects_on_edge in moving_objects.items():
                    k=[]
                    for obj in objects_on_edge:
                        #print(st.session_state.nodos_fila)
                        if not obj['id'] in vel:
                            vel[obj['id']]=True
                        if edge in st.session_state.nodos_fila:
                            if vel[obj['id']]==False:
                                vel[obj['id']]=True
                            elif vel[obj['id']]==True:
                                vel[obj['id']]=False
                            #print(vel,st.session_state.nodos_fila,11111,edge)
                        if not obj['id'] in vel_cols:
                            vel_cols[obj['id']]=True
                        if edge in st.session_state.nodos_cols:
                            if vel_cols[obj['id']]==False:
                                vel_cols[obj['id']]=True
                            elif vel_cols[obj['id']]==True:
                                vel_cols[obj['id']]=False

                        if edge[1] in node_semaforo_status:
                            if node_semaforo_status[edge[1]]=="ceda_el_paso(ave)":
                                adjacent_edges = [(u, v) for u, v in G.edges(edge[1]) if u[1] == i and v[1] == i]
                            elif node_semaforo_status[edge[1]]=="ceda_el_paso":
                                adjacent_edges = list(G.edges(edge[1]))
                            elif node_semaforo_status[edge[1]]=="ceda_el_paso(calle)":
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
                                
                        if not obj['id'] in stop:
                             stop[obj['id']]=True
                        if selected_stop and edge[1] in node_semaforo_status and node_semaforo_status[edge[1]]=="pare":
                            #print(stop[obj['id']],"befor")
                            if stop[obj['id']]==False:
                                stop[obj['id']]=True
                            elif stop[obj['id']]==True:
                                stop[obj['id']]=False
                            #print(stop[obj['id']])

                        #print(stop,obj['current_edge'])
                        if selected_node_s :
                                closest_edge = get_random_closest_edge(G, obj['current_edge'][1], obj['previous_node'],edge,k,True,stop[obj['id']],vel[obj['id']],vel_cols[obj['id']])
                        else:
                                closest_edge = get_random_closest_edge(G, obj['current_edge'][1], obj['previous_node'],edge,k,False,stop[obj['id']],vel[obj['id']],vel_cols[obj['id']])
                        
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
                fig = draw_graph(G, pos, edge_weights, edge_thickness,node_semaforo_status, centrality_value, moving_objects)
                graph_placeholder.plotly_chart(fig)

        # Esperar un momento antes de la siguiente actualización
                time.sleep(2)                
                count+=1
                if count%3==0 and selected_node_s:
                    actualizar_estado_aristas() 

if __name__ == "__main__":
    main()