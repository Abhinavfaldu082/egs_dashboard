import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import pydeck as pdk
import numpy as np
import networkx as nx

# Configure page settings
st.set_page_config(
    page_title="Alviridi ESG Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown(
    """
    <style>
    .main { padding: 0rem; }
    .stButton button { width: 100%; }
    .sidebar .sidebar-content { background-color: #1E1E1E; }
    .streamlit-expanderHeader { background-color: #262730; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Enhanced data loading with more detailed structure
@st.cache_data
def load_data():
    # Load CSV data
    df = pd.read_csv("dummy_sample.csv")

    # Process funds data
    funds_data = {}
    for fund in df["Fund"].unique():
        fund_df = df[df["Fund"] == fund]
        funds_data[fund] = {
            "total_capital": fund_df["Fund Size ($M)"].iloc[0]
            / 1000,  # Convert to billions
            "emissions": {
                "scope1": fund_df["Scope 1 Emissions (tons of CO2e)"].mean(),
                "scope2": fund_df["Scope 2 Emissions (tons of CO2e)"].mean(),
                "scope3": fund_df["Scope 3 Emissions (tons of CO2e)"].mean(),
            },
            "subfunds": {},
            "historical_emissions": [
                {
                    "date": "2020",
                    "scope1": fund_df["Scope 1 Emissions (tons of CO2e)"].mean() * 1.1,
                    "scope2": fund_df["Scope 2 Emissions (tons of CO2e)"].mean() * 1.1,
                    "scope3": fund_df["Scope 3 Emissions (tons of CO2e)"].mean() * 1.1,
                },
                {
                    "date": "2021",
                    "scope1": fund_df["Scope 1 Emissions (tons of CO2e)"].mean() * 1.05,
                    "scope2": fund_df["Scope 2 Emissions (tons of CO2e)"].mean() * 1.05,
                    "scope3": fund_df["Scope 3 Emissions (tons of CO2e)"].mean() * 1.05,
                },
                {
                    "date": "2022",
                    "scope1": fund_df["Scope 1 Emissions (tons of CO2e)"].mean(),
                    "scope2": fund_df["Scope 2 Emissions (tons of CO2e)"].mean(),
                    "scope3": fund_df["Scope 3 Emissions (tons of CO2e)"].mean(),
                },
            ],
        }

        # Create subfunds based on themes
        for _, row in (
            fund_df.groupby("Theme")["Theme Capital Catalyzed ($M)"].sum().items()
        ):
            funds_data[fund]["subfunds"][_] = row / 1000  # Convert to billions

    # Process companies data
    companies_data = {}
    for company in df["Company Name"].unique():
        company_df = df[df["Company Name"] == company]
        companies_data[company] = {
            "location": {
                "lat": 0,  # You would need actual coordinates
                "lon": 0,  # You would need actual coordinates
                "country": company_df["Country"].iloc[0],
            },
            "metrics": {
                "revenue": company_df["Investment ($M)"].mean(),
                "employees": int(
                    company_df["Investment ($M)"].mean() * 10
                ),  # Dummy calculation
                "carbon_footprint": company_df[
                    "Total Emissions by Fund (tons of CO2e)"
                ].mean(),
            },
            "projects": [f"Project {i+1}" for i in range(2)],  # Dummy projects
            "historical_performance": [
                {
                    "year": 2020,
                    "revenue": company_df["Investment ($M)"].mean() * 0.9,
                    "carbon_footprint": company_df[
                        "Total Emissions by Fund (tons of CO2e)"
                    ].mean()
                    * 1.1,
                },
                {
                    "year": 2021,
                    "revenue": company_df["Investment ($M)"].mean() * 0.95,
                    "carbon_footprint": company_df[
                        "Total Emissions by Fund (tons of CO2e)"
                    ].mean()
                    * 1.05,
                },
                {
                    "year": 2022,
                    "revenue": company_df["Investment ($M)"].mean(),
                    "carbon_footprint": company_df[
                        "Total Emissions by Fund (tons of CO2e)"
                    ].mean(),
                },
            ],
        }

    # Process thematic data
    thematic_data = df.groupby("Theme").size().to_dict()

    return funds_data, companies_data, thematic_data


def create_sidebar():
    st.sidebar.title("Search for a company:")
    companies = ["Overview"] + list(
        pd.read_csv("dummy_sample.csv")["Company Name"].unique()
    )
    selected_company = st.sidebar.selectbox("", companies, key="company_select")
    return selected_company


def create_navigation():
    tabs = [
        "alviridi",
        "Global South Investments",
        "Thematic Overview",
        "Emissions Dashboard",
    ]
    selected_tab = st.tabs(tabs)
    return selected_tab


def show_overview_metrics():
    df = pd.read_csv("dummy_sample.csv")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Capital Catalysed ($B)",
            f"{df['Total Capital Committed ($B)'].mean():.2f}",
        )
    with col2:
        st.metric("Total Fund Investments", len(df["Fund"].unique()))
    with col3:
        st.metric("Total Portfolio Companies", len(df["Company Name"].unique()))


def create_fund_charts(funds_data):
    # Create donut charts for fund distribution
    fund_values = [v["total_capital"] for v in funds_data.values()]
    fund_names = list(funds_data.keys())

    fig = go.Figure(
        data=[
            go.Pie(
                labels=fund_names,
                values=fund_values,
                hole=0.4,
                marker_colors=[
                    "rgb(31, 119, 180)",
                    "rgb(255, 127, 14)",
                    "rgb(44, 160, 44)",
                    "rgb(214, 39, 40)",
                    "rgb(148, 103, 189)",
                ],
            )
        ]
    )

    fig.update_layout(
        title="Committed Capital by Fund",
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )

    return fig

def create_fund_hierarchy_visualization(funds_data):
    # Create a network graph visualization
    G = nx.Graph()
    
    # Add nodes for main funds
    for fund_name, fund_data in funds_data.items():
        G.add_node(fund_name, size=fund_data["total_capital"])
        
        # Add nodes and edges for subfunds
        for subfund_name, subfund_value in fund_data["subfunds"].items():
            G.add_node(subfund_name, size=subfund_value)
            G.add_edge(fund_name, subfund_name)
    
    # Convert to plotly figure
    pos = nx.spring_layout(G)
    
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=[], y=[],
        mode='markers+text',
        hoverinfo='text',
        text=[],
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=[],
            color=[],
            line_width=2))

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (node,)
        node_trace['marker']['size'] += (G.nodes[node]['size']/5,)

    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0,l=0,r=0,t=0),
                       plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)',
                   ))
    
    return fig

def create_map(locations_data):
    df = pd.DataFrame(locations_data)
    view_state = pdk.ViewState(latitude=20.5937, longitude=78.9629, zoom=4, pitch=50)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_color="[200, 30, 0, 160]",
        get_radius=50000,
        pickable=True,
    )
    return pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10",
        initial_view_state=view_state,
        layers=[layer],
    )

def create_thematic_overview(thematic_data):
    df = pd.DataFrame(list(thematic_data.items()), columns=["Theme", "Number of Deals"])
    fig = px.bar(df, x="Theme", y="Number of Deals", title="Number of Deals by Theme")
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig

def create_emissions_charts(funds_data):
    emissions_data = []
    for fund_name, data in funds_data.items():
        for scope, value in data["emissions"].items():
            emissions_data.append({"Fund": fund_name, "Scope": scope, "Value": value})

    df_emissions = pd.DataFrame(emissions_data)
    fig = px.bar(
        df_emissions,
        x="Fund",
        y="Value",
        color="Scope",
        title="Emissions by Fund and Scope",
        barmode="group",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig

def create_emissions_timeline(funds_data):
    timeline_data = []
    for fund_name, fund_data in funds_data.items():
        for entry in fund_data["historical_emissions"]:
            timeline_data.append({
                "Fund": fund_name,
                "Date": entry["date"],
                "Scope 1": entry["scope1"],
                "Scope 2": entry["scope2"],
                "Scope 3": entry["scope3"]
            })
    
    df = pd.DataFrame(timeline_data)
    fig = px.line(df, x="Date", y=["Scope 1", "Scope 2", "Scope 3"], 
                  color="Fund", title="Historical Emissions by Fund")
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    return fig

def company_detail_view(company_data):
    st.subheader("Company Overview")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Revenue ($M)", company_data["metrics"]["revenue"])
    with col2:
        st.metric("Employees", company_data["metrics"]["employees"])
    with col3:
        st.metric("Carbon Footprint (tCO2e)", company_data["metrics"]["carbon_footprint"])
    
    # Historical Performance
    hist_df = pd.DataFrame(company_data["historical_performance"])
    fig = px.line(hist_df, x="year", y=["revenue", "carbon_footprint"],
                  title="Historical Performance")
    st.plotly_chart(fig, use_container_width=True)
    
    # Projects
    st.subheader("Active Projects")
    for project in company_data["projects"]:
        st.write(f"- {project}")

def data_editor():
    st.subheader("Data Editor")

    # Load current data
    funds_data, companies_data, _ = load_data()

    # Create tabs for different data sections
    data_tabs = st.tabs(["Funds", "Companies", "Emissions"])

    with data_tabs[0]:
        # Fund editor
        st.write("Edit Fund Data")
        for fund_name in funds_data.keys():
            with st.expander(fund_name):
                new_capital = st.number_input(
                    "Total Capital",
                    value=float(funds_data[fund_name]["total_capital"]),
                    key=f"capital_{fund_name}"
                )

                st.write("Subfunds")
                for subfund_name, subfund_value in funds_data[fund_name]["subfunds"].items():
                    st.number_input(
                        subfund_name,
                        value=float(subfund_value),
                        key=f"subfund_{fund_name}_{subfund_name}"
                    )

    with data_tabs[1]:
        # Company editor
        st.write("Edit Company Data")
        for company_name in companies_data.keys():
            with st.expander(company_name):
                new_revenue = st.number_input(
                    "Revenue",
                    value=float(companies_data[company_name]["metrics"]["revenue"]),
                    key=f"revenue_{company_name}"
                )
                new_employees = st.number_input(
                    "Employees",
                    value=int(companies_data[company_name]["metrics"]["employees"]),
                    key=f"employees_{company_name}"
                )

    with data_tabs[2]:
        # Emissions editor
        st.write("Edit Emissions Data")
        for fund_name in funds_data.keys():
            with st.expander(f"{fund_name} Emissions"):
                st.number_input(
                    "Scope 1",
                    value=float(funds_data[fund_name]["emissions"]["scope1"]),
                    key=f"scope1_{fund_name}"
                )
                st.number_input(
                    "Scope 2",
                    value=float(funds_data[fund_name]["emissions"]["scope2"]),
                    key=f"scope2_{fund_name}"
                )
                st.number_input(
                    "Scope 3",
                    value=float(funds_data[fund_name]["emissions"]["scope3"]),
                    key=f"scope3_{fund_name}"
                )


def main():
    funds_data, companies_data, thematic_data = load_data()
    selected_company = create_sidebar()
    selected_tabs = create_navigation()

    with selected_tabs[0]:  # alviridi tab
        st.title("alviridi Overview")
        show_overview_metrics()

        # New: Fund Hierarchy Visualization
        st.subheader("Fund Structure")
        st.plotly_chart(
            create_fund_hierarchy_visualization(funds_data), use_container_width=True
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_fund_charts(funds_data), use_container_width=True)
        with col2:
            # Create a different view for the second column
            fig = px.bar(
                pd.DataFrame(
                    [(k, v["total_capital"]) for k, v in funds_data.items()],
                    columns=["Fund", "Capital"],
                ),
                x="Fund",
                y="Capital",
                title="Capital Distribution",
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with selected_tabs[1]:  # Global South Investments tab
        st.title("Global South Investments - Overview")

        # Show map
        locations = [v["location"] for v in companies_data.values()]
        st.pydeck_chart(create_map(locations))

        # Add regional metrics
        st.subheader("Regional Distribution")
        col1, col2 = st.columns(2)
        with col1:
            # Country distribution
            countries = [loc["country"] for loc in locations]
            country_counts = pd.Series(countries).value_counts()
            fig = px.pie(
                values=country_counts.values,
                names=country_counts.index,
                title="Investments by Country",
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Show key metrics
            st.metric("Total Countries", len(set(countries)))
            st.metric("Total Companies", len(companies_data))
            total_revenue = sum(
                comp["metrics"]["revenue"] for comp in companies_data.values()
            )
            st.metric("Total Revenue ($M)", f"{total_revenue:.1f}")

    with selected_tabs[2]:  # Thematic Overview tab
        st.title("Thematic Overview")
        if selected_company != "Overview" and selected_company in companies_data:
            # Show detailed company view when a company is selected
            company_detail_view(companies_data[selected_company])
        else:
            # Show thematic overview
            st.plotly_chart(
                create_thematic_overview(thematic_data), use_container_width=True
            )

            # Add thematic breakdown
            st.subheader("Thematic Analysis")
            col1, col2 = st.columns(2)
            with col1:
                # Show top themes
                st.write("Top Themes by Number of Deals")
                sorted_themes = sorted(
                    thematic_data.items(), key=lambda x: x[1], reverse=True
                )
                for theme, deals in sorted_themes[:5]:
                    st.write(f"- {theme}: {deals} deals")

            with col2:
                # Show theme distribution
                fig = px.pie(
                    values=list(thematic_data.values()),
                    names=list(thematic_data.keys()),
                    title="Theme Distribution",
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig, use_container_width=True)

    with selected_tabs[3]:  # Emissions Dashboard tab
        st.title("Emissions Dashboard")

        # Summary metrics
        total_scope1 = sum(fund["emissions"]["scope1"] for fund in funds_data.values())
        total_scope2 = sum(fund["emissions"]["scope2"] for fund in funds_data.values())
        total_scope3 = sum(fund["emissions"]["scope3"] for fund in funds_data.values())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Scope 1 Emissions", f"{total_scope1:.1f}")
        with col2:
            st.metric("Total Scope 2 Emissions", f"{total_scope2:.1f}")
        with col3:
            st.metric("Total Scope 3 Emissions", f"{total_scope3:.1f}")

        # Enhanced emissions visualizations
        st.plotly_chart(create_emissions_charts(funds_data), use_container_width=True)
        st.plotly_chart(create_emissions_timeline(funds_data), use_container_width=True)

        # Data Editor
        if st.checkbox("Show Data Editor"):
            data_editor()


if __name__ == "__main__":
    main()
