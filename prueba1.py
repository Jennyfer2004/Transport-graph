
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import random

def draw_graph(G, pos, edge_weights, edge_thickness, centrality_value=None, moving_objects=None):
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
    node_size = []
    node_color = []
    if centrality_value:
        max_centrality = max(centrality_value.values())
        min_centrality = min(centrality_value.values())
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            text.append(str(node))
            centrality = centrality_value[node]
            node_color.append((centrality - min_centrality) / (max_centrality - min_centrality) if max_centrality != min_centrality else 0.5)
            node_size.append(centrality * 50)
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

    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )
    return fig

# Crear un grafo de cuadrícula y conectarlos con n aristas
def create_connected_grid_graphs(rows1, cols1, rows2, cols2, num_edges):
    G1 = nx.grid_2d_graph(rows1, cols1)  # Primer grafo de cuadrícula
    G2 = nx.grid_2d_graph(rows2, cols2)  # Segundo grafo de cuadrícula

    # Renombrar los nodos del segundo grafo para que no se solapen con el primero
    max_col1 = max([node[1] for node in G1.nodes()])
    mapping = {node: (node[0], node[1] + max_col1 + 2) for node in G2.nodes()}  # Desplazar el segundo grafo
    G2 = nx.relabel_nodes(G2, mapping)

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

    return G

# Función principal de Streamlit
def main():
    st.title("Visualización de Grafos de Cuadrícula Conectados con n Aristas")

    # Selección de cantidad de grafos
    st.sidebar.header("Parámetros de la visualización")
    num_graphs = st.sidebar.slider("Cantidad de grafos a visualizar", min_value=1, max_value=4, value=2)

# Selección de filas y columnas para cada grafo
    graph_parameters = []
    for i in range(num_graphs):
        st.sidebar.subheader(f"Grafo {i + 1}")
        rows = st.sidebar.slider(f"Filas del grafo {i + 1}", min_value=2, max_value=10, value=3, key=f"rows{i}")
        cols = st.sidebar.slider(f"Columnas del grafo {i + 1}", min_value=2, max_value=10, value=3, key=f"cols{i}")
        num_edges = st.sidebar.slider(f"Número de aristas entre grafos {i + 1}", min_value=1, max_value=10, value=1, key=f"edges{i}")
        graph_parameters.append((rows, cols, num_edges))

    # Crear y visualizar los grafos
    all_graphs = []
    for i in range(num_graphs - 1):
        rows1, cols1, num_edges1 = graph_parameters[i]
        rows2, cols2, num_edges2 = graph_parameters[i + 1]
        G = create_connected_grid_graphs(rows1, cols1, rows2, cols2, num_edges1)
        all_graphs.append(G)

    # Dibuja y muestra cada grafo
    for i, G in enumerate(all_graphs):
        pos = nx.spring_layout(G)
        edge_weights = {edge: 1 for edge in G.edges()}
        edge_thickness = {edge: 2 for edge in G.edges()}
        fig = draw_graph(G, pos, edge_weights, edge_thickness)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()