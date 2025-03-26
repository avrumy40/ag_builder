import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import io
import json
import random
from os import path, makedirs
from streamlit_elements import elements, mui, html
from streamlit_extras.add_vertical_space import add_vertical_space

# Global font constants for consistent styling
HELVETICA_FONT = 'Helvetica Neue, Helvetica, Arial, sans-serif'

# Set page configuration
st.set_page_config(
    page_title="AG Hierarchy Builder",
    page_icon="🛍️",
    layout="wide"
)

# Initialize session state variables
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'df' not in st.session_state:
    st.session_state.df = None
if 'selected_attributes' not in st.session_state:
    st.session_state.selected_attributes = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'attribute_stats' not in st.session_state:
    st.session_state.attribute_stats = {}
if 'auto_scroll' not in st.session_state:
    st.session_state.auto_scroll = False
# Set default values for visualization and analysis parameters
if 'hist_xlimit' not in st.session_state:
    st.session_state.hist_xlimit = 50
if 'hist_xticks_jumps' not in st.session_state:
    st.session_state.hist_xticks_jumps = 10
if 'enable_hierarchy' not in st.session_state:
    st.session_state.enable_hierarchy = True
# Add missing parameters (previously set in the UI that we removed)
if 'min_products_in_group' not in st.session_state:
    st.session_state.min_products_in_group = 1
if 'max_products_in_group' not in st.session_state:
    st.session_state.max_products_in_group = 1000

# Navigation functions
def go_to_next_step():
    st.session_state.current_step += 1
    # Auto-scroll to top on next render
    st.session_state.auto_scroll = True

def go_to_prev_step():
    st.session_state.current_step -= 1
    # Auto-scroll to top on next render
    st.session_state.auto_scroll = True

def go_to_step(step):
    st.session_state.current_step = step
    # Auto-scroll to top on next render
    st.session_state.auto_scroll = True

# Helper Functions
def build_category(df, cat_cols, hierarchy_mode=False):
    """Create AG column by concatenating values from selected columns
    
    When hierarchy_mode is True, it will also create columns for each level of the hierarchy
    """
    df = df.copy()
    df['ag'] = ''
    
    # If hierarchical mode is enabled, create each level separately
    if hierarchy_mode:
        # Add columns for each hierarchy level
        for i in range(1, len(cat_cols) + 1):
            level_cols = cat_cols[:i]
            level_name = f'ag_level_{i}'
            df[level_name] = ''
            for c in level_cols:
                df[level_name] += df[c].astype(str)
    
    # Create the final AG (containing all levels)
    for c in cat_cols:
        df['ag'] += df[c].astype(str)
        
    return df

def calc_dist(df, comb_str):
    """Calculate distribution statistics for a combination"""
    dist_ = df.groupby('ag')['product_id'].size()
    dist_desc = dist_.describe()
    n_groups = dist_.shape[0]
    n_products = dist_.sum()

    min_products_in_group = min(dist_)
    max_products_in_group = max(dist_)

    df_stats = pd.DataFrame({
        'cols': [comb_str],
        '# Products': [n_products],
        '# AGs': [n_groups],
        'min_products_in_ag': [min_products_in_group],
        'max_products_in_ag': [max_products_in_group],
        'Average Products in AG': [int(dist_desc['mean'])],
        'Median Products in AG': [dist_desc['50%']],
        'Std Products in AG': [dist_desc['std']]
    })

    return df_stats, dist_

def calc_price_stats(df):
    """Calculate price variance and statistics for each AG"""
    if 'price' not in df.columns or 'ag' not in df.columns:
        return None
    
    # Group by AG and calculate price statistics
    price_stats = df.groupby('ag')['price'].agg(['mean', 'std', 'min', 'max', 'count'])
    price_stats.columns = ['Price Mean', 'Price Std', 'Price Min', 'Price Max', 'Count']
    
    # Calculate coefficient of variation (CV) as a normalized measure of dispersion
    price_stats['Price CV'] = price_stats['Price Std'] / price_stats['Price Mean'] * 100
    
    return price_stats

def plot_dist(dist_, df_stats, comb_str, hist_xlimit, hist_xticks_jumps):
    """Plot bar chart showing products per AG distribution using Plotly for interactive tooltips"""
    # Sort the AGs by product count
    sorted_dist = dist_.sort_values(ascending=False)
    
    # Extract actual AG names from the index (for display on axis)
    ag_names = []
    for ag_idx in sorted_dist.index:
        ag_parts = str(ag_idx).split('_')
        if len(ag_parts) > 0:
            # Use first part of AG name (or full name if it's not too long)
            if len(str(ag_idx)) > 10 and len(ag_parts) > 1:
                ag_names.append(ag_parts[0][:10] + "...")
            else:
                ag_names.append(str(ag_idx)[:10])
    
    # Create the Plotly figure
    fig = go.Figure()
    
    # Add the bar chart with hover tooltips showing FULL AG names
    fig.add_trace(go.Bar(
        x=list(range(len(sorted_dist))),
        y=sorted_dist.values,
        marker_color='#38BDF8',
        hoverinfo='text',
        hovertext=[f'AG: {str(ag_idx)}<br>Products: {int(val)}' for ag_idx, val in sorted_dist.items()],
        name='Products per AG'
    ))
    
    # If there are too many AGs, show only a subset of labels
    if len(ag_names) > 30:
        # Show tick marks for the first AGs and then every 5th AG
        visible_ticks = list(range(10)) + list(range(10, len(ag_names), 5))
        ticktext = [ag_names[i] if i < len(ag_names) else '' for i in visible_ticks]
        tickvals = visible_ticks
    else:
        ticktext = ag_names
        tickvals = list(range(len(ag_names)))
    
    # Update layout with styling
    fig.update_layout(
        title={
            'text': f'Products per AG Distribution - {comb_str}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': '#0F172A', 'family': HELVETICA_FONT}
        },
        bargap=0.1,  # Explicitly set bargap to avoid numpy.float64 conversion issues
        xaxis={
            'title': {
                'text': 'Assortment Groups',
                'font': {'size': 12, 'color': '#0F172A', 'family': HELVETICA_FONT}
            },
            'tickfont': {'size': 10, 'color': '#0F172A', 'family': HELVETICA_FONT},
            'tickmode': 'array',
            'tickvals': tickvals,
            'ticktext': ticktext,
            'tickangle': -90
        },
        yaxis={
            'title': {
                'text': '# of Products',
                'font': {'size': 12, 'color': '#0F172A', 'family': HELVETICA_FONT}
            },
            'tickfont': {'size': 10, 'color': '#0F172A', 'family': HELVETICA_FONT},
            'gridcolor': '#94A3B8',
            'gridwidth': 1,
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        margin={'t': 60, 'b': 100, 'l': 60, 'r': 20}
    )
    
    return fig

def plot_ags(dist_, df_stats, comb_str, hist_xlimit):
    """Plot bar chart of products per AG using Plotly for interactive tooltips"""
    # Get top AGs and prepare data
    top_ags = dist_.sort_values(ascending=False)[:hist_xlimit]
    
    # Generate colors from our design system palette
    colors = generate_colors(len(top_ags))
    
    # Extract actual AG names from the index (for display on axis)
    ag_names = []
    for ag_idx in top_ags.index:
        ag_parts = str(ag_idx).split('_')
        if len(ag_parts) > 0:
            # Use first part of AG name (or full name if it's not too long)
            if len(str(ag_idx)) > 15 and len(ag_parts) > 1:
                ag_names.append(ag_parts[0][:15] + "...")
            else:
                ag_names.append(str(ag_idx)[:15])
    
    # Create the Plotly figure
    fig = go.Figure()
    
    # Add the bar chart with hover tooltips showing FULL AG names
    fig.add_trace(go.Bar(
        x=list(range(len(top_ags))),
        y=top_ags.values,
        marker_color=colors,
        hoverinfo='text',
        hovertext=[f'AG: {str(ag_idx)}<br>Products: {int(val)}' for ag_idx, val in top_ags.items()],
        name='Products per AG'
    ))
    
    # Update layout with styling
    fig.update_layout(
        title={
            'text': f'Top AGs by Product Count - {comb_str}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': '#0F172A', 'family': HELVETICA_FONT}
        },
        bargap=0.1,  # Explicitly set bargap to avoid numpy.float64 conversion issues
        xaxis={
            'title': {
                'text': 'Assortment Group',
                'font': {'size': 12, 'color': '#0F172A', 'family': HELVETICA_FONT}
            },
            'tickfont': {'size': 10, 'color': '#0F172A', 'family': HELVETICA_FONT},
            'tickmode': 'array',
            'tickvals': list(range(len(ag_names))),
            'ticktext': ag_names,
            'tickangle': -90
        },
        yaxis={
            'title': {
                'text': '# of Products',
                'font': {'size': 12, 'color': '#0F172A', 'family': HELVETICA_FONT}
            },
            'tickfont': {'size': 10, 'color': '#0F172A', 'family': HELVETICA_FONT},
            'gridcolor': '#94A3B8',
            'gridwidth': 1,
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        margin={'t': 60, 'b': 100, 'l': 60, 'r': 20}
    )
    
    return fig

def plot_price_variance(price_stats, comb_str):
    """Plot price variation for all AGs using Plotly for interactive tooltips"""
    if price_stats is None:
        return None
    
    # Sort all AGs by count
    top_ags = price_stats.sort_values('Count', ascending=False)
    
    # Extract actual AG names from the index (for display on axis)
    ag_names = []
    for ag_idx in top_ags.index:
        ag_parts = str(ag_idx).split('_')
        if len(ag_parts) > 0:
            # Use first part of AG name (or full name if it's not too long)
            if len(str(ag_idx)) > 15 and len(ag_parts) > 1:
                ag_names.append(ag_parts[0][:15] + "...")
            else:
                ag_names.append(str(ag_idx)[:15])
    
    # Create the Plotly figure
    fig = go.Figure()
    
    # Add the bar chart with hover tooltips showing FULL AG names
    fig.add_trace(go.Bar(
        x=list(range(len(top_ags))),
        y=top_ags['Price CV'].values,
        marker_color='#EF4444',
        marker_line_color='#DC2626',
        marker_line_width=1,
        hoverinfo='text',
        hovertext=[f'AG: {ag_idx}<br>Price CV: {val:.1f}%<br>Mean Price: {top_ags["Price Mean"].iloc[i]:.2f}<br>Min Price: {top_ags["Price Min"].iloc[i]:.2f}<br>Max Price: {top_ags["Price Max"].iloc[i]:.2f}' 
                  for i, (ag_idx, val) in enumerate(zip(top_ags.index, top_ags['Price CV'].values))],
        name='Price Variation'
    ))
    
    # Update layout with styling
    fig.update_layout(
        title={
            'text': f'Price Variation by AG - {comb_str}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': '#0F172A', 'family': HELVETICA_FONT}
        },
        bargap=0.1,  # Explicitly set bargap to avoid numpy.float64 conversion issues
        xaxis={
            'title': {
                'text': 'Assortment Group',
                'font': {'size': 12, 'color': '#0F172A', 'family': HELVETICA_FONT}
            },
            'tickfont': {'size': 10, 'color': '#0F172A', 'family': HELVETICA_FONT},
            'tickmode': 'array',
            'tickvals': list(range(len(ag_names))),
            'ticktext': ag_names,
            'tickangle': -90
        },
        yaxis={
            'title': {
                'text': 'Price Coefficient of Variation (%)',
                'font': {'size': 12, 'color': '#0F172A', 'family': HELVETICA_FONT}
            },
            'tickfont': {'size': 10, 'color': '#0F172A', 'family': HELVETICA_FONT},
            'gridcolor': '#94A3B8',
            'gridwidth': 1,
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        margin={'t': 60, 'b': 100, 'l': 60, 'r': 20}
    )
    
    return fig

def save_comb(df, comb_str, output_dir='output'):
    """Save combination data to CSV file"""
    if not path.exists(output_dir):
        makedirs(output_dir)
    
    csv_path = path.join(output_dir, f'{comb_str}.csv')
    df.to_csv(csv_path, index=False)
    return csv_path

def plot_numeric_attribute(attr_name, stats):
    """Plot histogram for numeric attribute using Plotly for interactive tooltips"""
    # Extract histogram data
    counts, bins = stats['histogram']
    
    # Create bin labels for the x-axis and tooltip
    bin_labels = [f"{bins[i]:.2f} - {bins[i+1]:.2f}" for i in range(len(bins)-1)]
    
    # Create the Plotly figure
    fig = go.Figure()
    
    # Add the bar chart with hover tooltips
    fig.add_trace(go.Bar(
        x=bin_labels,
        y=counts,
        marker_color='#38BDF8',  # Neon Sky from our design system
        hoverinfo='text',
        hovertext=[f'Range: {bin_labels[i]}<br>Count: {int(count)}' for i, count in enumerate(counts)],
        name=attr_name
    ))
    
    # Update layout with styling
    fig.update_layout(
        title={
            'text': f'Distribution of {attr_name}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': '#0F172A', 'family': HELVETICA_FONT}
        },
        bargap=0.1,  # Explicitly set bargap to avoid numpy.float64 conversion issues
        xaxis={
            'title': {
                'text': attr_name,
                'font': {'size': 12, 'color': '#0F172A', 'family': HELVETICA_FONT}
            },
            'tickfont': {'size': 10, 'color': '#0F172A', 'family': HELVETICA_FONT},
            'tickangle': -45
        },
        yaxis={
            'title': {
                'text': 'Count',
                'font': {'size': 12, 'color': '#0F172A', 'family': HELVETICA_FONT}
            },
            'tickfont': {'size': 10, 'color': '#0F172A', 'family': HELVETICA_FONT},
            'gridcolor': '#94A3B8',
            'gridwidth': 1,
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        margin={'t': 60, 'b': 80, 'l': 60, 'r': 20}
    )
    
    return fig

def analyze_attributes(df):
    """Analyze available attributes for their distributions"""
    results = {}
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Analyze categorical columns
    for col in categorical_cols:
        if col in ['ag', 'id', 'name', 'description', 'product_description']:
            continue
            
        value_counts = df[col].value_counts()
        results[col] = {
            'type': 'categorical',
            'unique_values': len(value_counts),
            'top_values': value_counts.head(10).to_dict(),
            'null_count': df[col].isna().sum(),
            'example_values': df[col].dropna().sample(min(5, len(df))).tolist()
        }
    
    # Analyze numeric columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            # Fill NAs with 0 for statistical calculations
            calc_series = df[col].fillna(0)
        else:
            calc_series = df[col]
            
        results[col] = {
            'type': 'numeric',
            'min': calc_series.min(),
            'max': calc_series.max(),
            'mean': calc_series.mean(),
            'median': calc_series.median(),
            'std': calc_series.std(),
            'null_count': df[col].isna().sum(),
            'histogram': np.histogram(calc_series, bins=10)
        }
    
    return results

def run_analysis(df, selected_attributes, min_products, max_products, 
                hist_xlimit, hist_xticks_jumps, enable_hierarchy):
    """Run complete analysis on selected attributes"""
    
    # Skip if no attributes selected
    if not selected_attributes:
        return None, None, None
    
    # Create AG column
    df_tmp = build_category(df=df, cat_cols=selected_attributes, hierarchy_mode=enable_hierarchy)
    comb_str = "_".join(selected_attributes)
    
    # Calculate distributions
    df_stats, dist_ = calc_dist(df=df_tmp, comb_str=comb_str)
    
    # Calculate price statistics per AG if price column exists
    price_stats = calc_price_stats(df_tmp) if 'price' in df_tmp.columns else None
    
    return df_tmp, df_stats, dist_, price_stats

# Chart.js visualizations using streamlit-elements
def generate_colors(n):
    """Generate a list of distinct colors using the design system colors"""
    # Base colors from our design system
    palette = [
        '#38BDF8',  # Neon Sky (primary)
        '#22C55E',  # Nano Green
        '#FACC15',  # Pulse Yellow
        '#EF4444',  # Alert Red
        '#94A3B8',  # Slate Grey
        
        # Additional variations of the main colors
        '#0EA5E9',  # Darker Neon Sky
        '#16A34A',  # Darker Nano Green
        '#EAB308',  # Darker Pulse Yellow
        '#DC2626',  # Darker Alert Red
        '#64748B',  # Darker Slate Grey
        
        '#7DD3FC',  # Lighter Neon Sky
        '#4ADE80',  # Lighter Nano Green
        '#FDE047',  # Lighter Pulse Yellow
        '#FCA5A5',  # Lighter Alert Red
        '#CBD5E1',  # Lighter Slate Grey
    ]
    
    # If we need more colors than in the palette, generate variations
    if n > len(palette):
        base_colors = palette[:5]  # Take the 5 primary colors
        for i in range(n - len(palette)):
            base_color = base_colors[i % len(base_colors)]
            # Generate variation by adjusting brightness
            variation = random.uniform(-0.2, 0.2)
            
            # Convert hex to RGB
            r = int(base_color[1:3], 16)
            g = int(base_color[3:5], 16)
            b = int(base_color[5:7], 16)
            
            # Adjust RGB values
            r = max(0, min(255, int(r * (1 + variation))))
            g = max(0, min(255, int(g * (1 + variation))))
            b = max(0, min(255, int(b * (1 + variation))))
            
            # Convert back to hex
            new_color = f'#{r:02x}{g:02x}{b:02x}'
            palette.append(new_color)
    
    return palette[:n]

def plot_hierarchy_level(df_tmp, level, col_name, comb_str):
    """Plot hierarchy level distribution using Plotly for interactive tooltips"""
    level_name = f'ag_level_{level}'
    
    if level_name not in df_tmp.columns:
        return None
    
    # Calculate distribution for this level
    level_dist = df_tmp.groupby(level_name)['product_id'].size()
    top_groups = level_dist.sort_values(ascending=False)[:20]
    
    # Generate colors from our design system palette
    colors = generate_colors(len(top_groups))
    
    # Extract actual group names from the index (for display on axis)
    group_names = []
    for group_idx in top_groups.index:
        # Truncate long names
        if len(str(group_idx)) > 15:
            group_names.append(str(group_idx)[:15] + "...")
        else:
            group_names.append(str(group_idx))
    
    # Create the Plotly figure
    fig = go.Figure()
    
    # Add the bar chart with hover tooltips showing FULL group names
    fig.add_trace(go.Bar(
        x=list(range(len(top_groups))),
        y=top_groups.values,
        marker_color=colors,
        hoverinfo='text',
        hovertext=[f'Group: {str(group_idx)}<br>Products: {int(val)}' for group_idx, val in top_groups.items()],
        name=f'Level {level} Groups'
    ))
    
    # Update layout with styling
    fig.update_layout(
        title={
            'text': f'Top Groups at Level {level}: {col_name}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': '#0F172A', 'family': HELVETICA_FONT}
        },
        bargap=0.1,  # Explicitly set bargap to avoid numpy.float64 conversion issues
        xaxis={
            'title': {
                'text': f'{col_name}',
                'font': {'size': 12, 'color': '#0F172A', 'family': HELVETICA_FONT}
            },
            'tickfont': {'size': 10, 'color': '#0F172A', 'family': HELVETICA_FONT},
            'tickmode': 'array',
            'tickvals': list(range(len(group_names))),
            'ticktext': group_names,
            'tickangle': -90
        },
        yaxis={
            'title': {
                'text': '# of Products',
                'font': {'size': 12, 'color': '#0F172A', 'family': HELVETICA_FONT}
            },
            'tickfont': {'size': 10, 'color': '#0F172A', 'family': HELVETICA_FONT},
            'gridcolor': '#94A3B8',
            'gridwidth': 1,
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        margin={'t': 60, 'b': 100, 'l': 60, 'r': 20}
    )
    
    return fig

def chartjs_dist_histogram(dist_, df_stats, comb_str, hist_xlimit, hist_xticks_jumps):
    """Create Chart.js histogram data for products per AG distribution"""
    
    # Create histogram bins
    bins = min(df_stats['# AGs'].values[0], 50)
    hist_data, bin_edges = np.histogram(dist_, bins=bins, range=(0, hist_xlimit))
    
    # Create labels for each bin (taking the middle value of each bin)
    labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
    
    # Create Chart.js data
    data = {
        "labels": labels,
        "datasets": [
            {
                "label": "Number of AGs",
                "data": hist_data.tolist(),
                "backgroundColor": "#38BDF8",  # Neon Sky from our design system
                "borderColor": "#0EA5E9",      # Darker Neon Sky
                "borderWidth": 1
            }
        ]
    }
    
    # Chart options
    options = {
        "responsive": True,
        "maintainAspectRatio": False,
        "plugins": {
            "title": {
                "display": True,
                "text": f"Distribution of Products per AG - {comb_str}",
                "font": {"size": 16, "family": HELVETICA_FONT}
            },
            "tooltip": {"mode": "index", "intersect": False},
            "legend": {"position": "top"}
        },
        "scales": {
            "x": {
                "title": {"display": True, "text": "# of products in AG"},
                "grid": {"display": False}
            },
            "y": {
                "title": {"display": True, "text": "# of AGs"},
                "grid": {"color": "#E0E0E0"}
            }
        }
    }
    
    return data, options

def chartjs_top_ags(dist_, df_stats, comb_str, hist_xlimit):
    """Create Chart.js data for top AGs by product count"""
    
    # Get top AGs sorted by product count
    top_ags = dist_.sort_values(ascending=False)[:hist_xlimit]
    
    # Generate colors
    colors = generate_colors(len(top_ags))
    
    # Create Chart.js data
    data = {
        "labels": [str(i) for i in range(len(top_ags))],  # Using index numbers as labels
        "datasets": [
            {
                "label": "# of Products",
                "data": top_ags.tolist(),
                "backgroundColor": colors,
                "borderColor": colors,
                "borderWidth": 1
            }
        ]
    }
    
    # Chart options
    options = {
        "responsive": True,
        "maintainAspectRatio": False,
        "plugins": {
            "title": {
                "display": True,
                "text": f"Top AGs by Product Count - {comb_str}",
                "font": {"size": 16, "family": HELVETICA_FONT}
            },
            "tooltip": {"mode": "index", "intersect": False},
            "legend": {"display": False}
        },
        "scales": {
            "x": {
                "title": {"display": True, "text": "AG Index"},
                "grid": {"display": False}
            },
            "y": {
                "title": {"display": True, "text": "# of Products"},
                "grid": {"color": "#E0E0E0"}
            }
        }
    }
    
    return data, options

def chartjs_price_variance(price_stats, comb_str):
    """Create Chart.js data for price variation by AG"""
    if price_stats is None:
        return None, None
    
    # Sort all AGs by count
    top_ags = price_stats.sort_values('Count', ascending=False)
    
    # Create Chart.js data for price coefficient of variation
    data = {
        "labels": [str(i) for i in range(len(top_ags))],  # Using index numbers as labels
        "datasets": [
            {
                "label": "Price Coefficient of Variation (%)",
                "data": top_ags['Price CV'].tolist(),
                "backgroundColor": "#EF4444",  # Alert Red from our design system
                "borderColor": "#DC2626",  # Darker Alert Red
                "borderWidth": 1
            }
        ]
    }
    
    # Chart options
    options = {
        "responsive": True,
        "maintainAspectRatio": False,
        "plugins": {
            "title": {
                "display": True,
                "text": f"Price Variation by AG - {comb_str}",
                "font": {"size": 16}
            },
            "tooltip": {
                "mode": "index", 
                "intersect": False,
                "callbacks": {
                    "label": "function(context) { return 'Price Variation: ' + context.raw.toFixed(2) + '%'; }"
                }
            },
            "legend": {"display": False}
        },
        "scales": {
            "x": {
                "title": {"display": True, "text": "AG Index"},
                "grid": {"display": False}
            },
            "y": {
                "title": {"display": True, "text": "Coefficient of Variation (%)"},
                "grid": {"color": "#E0E0E0"}
            }
        }
    }
    
    return data, options

def chartjs_hierarchy_level(df_tmp, level, col_name, comb_str):
    """Create Chart.js data for hierarchy level analysis"""
    level_name = f'ag_level_{level}'
    
    if level_name not in df_tmp.columns:
        return None, None
    
    # Calculate distribution for this level
    level_dist = df_tmp.groupby(level_name)['product_id'].size()
    top_groups = level_dist.sort_values(ascending=False)[:20]
    
    # Generate colors
    colors = generate_colors(len(top_groups))
    
    # Create Chart.js data
    data = {
        "labels": [str(i) for i in range(len(top_groups))],  # Using index numbers as labels
        "datasets": [
            {
                "label": "# of Products",
                "data": top_groups.tolist(),
                "backgroundColor": colors,
                "borderColor": colors,
                "borderWidth": 1
            }
        ]
    }
    
    # Chart options
    options = {
        "responsive": True,
        "maintainAspectRatio": False,
        "plugins": {
            "title": {
                "display": True,
                "text": f"Top Groups at Level {level}: {col_name}",
                "font": {"size": 16}
            },
            "tooltip": {"mode": "index", "intersect": False},
            "legend": {"display": False}
        },
        "scales": {
            "x": {
                "title": {"display": True, "text": f"Groups at Level {level}"},
                "grid": {"display": False}
            },
            "y": {
                "title": {"display": True, "text": "# of Products"},
                "grid": {"color": "#E0E0E0"}
            }
        }
    }
    
    return data, options

# Page layout - Modern UI with new design system
st.markdown("""
<style>
    /* Modern Neutrals + Accent Color System */
    :root {
        /* Main palette */
        --electric-blue: #3B82F6;  /* Primary accent color */
        --snow-white: #F9FAFB;     /* Background */
        --light-gray: #E5E7EB;     /* Surface */
        --soft-gray: #D1D5DB;      /* Border/Divider */
        --charcoal: #111827;       /* Text (Main) */
        --slate-gray: #6B7280;     /* Text (Secondary) */
        --mint-green: #10B981;     /* Success */
        --amber-yellow: #F59E0B;   /* Warning */
        --rose-red: #EF4444;       /* Error */
        
        /* For backward compatibility */
        --space-navy: var(--charcoal);
        --neon-sky: var(--electric-blue);
        --cloud-white: var(--snow-white);
        --slate-grey: var(--slate-gray);
        --nano-green: var(--mint-green);
        --pulse-yellow: var(--amber-yellow);
        --alert-red: var(--rose-red);
    }

    /* Typography settings */
    body {
        font-family: 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif;
    }
    
    h1, .main-title {
        font-weight: 700 !important;          /* Bold */
        font-size: 32px !important;           /* Size */
        line-height: 40px !important;         /* Line Height */
        color: var(--charcoal) !important;    /* Color */
        text-align: center;
        margin-bottom: 20px !important;
    }

    h2, .section-header {
        font-weight: 600 !important;          /* Semi-Bold */
        font-size: 24px !important;           /* Size */
        line-height: 32px !important;         /* Line Height */
        color: var(--charcoal) !important;    /* Color */
        margin-bottom: 16px !important;
    }

    h3, .card-title {
        font-weight: 500 !important;          /* Medium */
        font-size: 20px !important;           /* Size */
        line-height: 28px !important;         /* Line Height */
        color: var(--charcoal) !important;    /* Color */
        margin-bottom: 12px !important;
    }
    
    .step-title {
        font-family: 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif !important;
        font-size: 20px !important;
        font-weight: 500 !important;
        text-align: center;
        margin-bottom: 0.5rem;
        color: var(--charcoal);
    }

    p, div, span, li {
        font-weight: 400 !important;          /* Regular */
        font-size: 16px !important;           /* Size */
        line-height: 24px !important;         /* Line Height */
    }

    .caption, .help-text {
        font-weight: 400 !important;          /* Regular */
        font-size: 14px !important;           /* Size */
        line-height: 20px !important;         /* Line Height */
        color: var(--slate-gray) !important;  /* Color */
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: var(--electric-blue);
    }
    
    /* Card styling */
    .stat-card {
        background-color: var(--snow-white);
        border: 1px solid var(--soft-gray);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease-in-out;
    }
    
    .stat-card:hover {
        transform: scale(1.02);
    }

    .stat-value {
        font-size: 28px !important;
        font-weight: 600 !important;
        color: var(--electric-blue) !important;
        margin-bottom: 5px;
        text-align: center;
    }

    .stat-label {
        font-size: 14px !important;
        color: var(--slate-gray) !important;
        font-weight: 400 !important;
        text-align: center;
        margin-top: 0.5rem;
    }

    /* Component styling */
    .stButton button {
        font-family: 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif !important;
        font-weight: 500 !important;
        background-color: var(--electric-blue) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }

    .stButton button:hover {
        background-color: #2563EB !important; /* Slightly darker */
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }

    /* Make disabled buttons visibly distinct */
    .stButton button:disabled {
        background-color: var(--soft-gray) !important;
        color: var(--slate-gray) !important;
        cursor: not-allowed !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* Chart container */
    .chart-container {
        background-color: white;
        border: 1px solid var(--soft-gray);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    /* Status messages */
    .success-message {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid var(--mint-green);
        color: #065F46 !important;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 10px 0;
    }

    .warning-message {
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 4px solid var(--amber-yellow);
        color: #92400E !important;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 10px 0;
    }

    .error-message {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid var(--rose-red);
        color: #991B1B !important;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 10px 0;
    }

    /* Tables */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    .dataframe th {
        background-color: var(--charcoal);
        color: var(--snow-white);
        font-weight: 500;
    }
    
    .dataframe tr:hover {
        background-color: var(--snow-white);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: nowrap;
        padding: 0 16px;
        color: var(--slate-gray);
        border-radius: 6px 6px 0 0;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--electric-blue);
        height: 3px;
    }

    .stTabs [aria-selected="true"] {
        color: var(--electric-blue);
        font-weight: 500;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 16px !important;
        font-weight: 500 !important;
        color: var(--charcoal) !important;
    }

    /* Form elements */
    .stSelectbox label, .stMultiSelect label, .stSlider label, .stFileUploader label {
        font-family: 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif !important;
        font-weight: 500 !important;
        color: var(--charcoal) !important;
        font-size: 16px !important;
        margin-bottom: 5px !important;
    }
    
    /* Add font family to select/input elements themselves */
    .stSelectbox, .stMultiSelect, .stSlider, .stFileUploader {
        font-family: 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# Auto-scroll to top when navigating steps
if st.session_state.auto_scroll:
    st.markdown("""
    <script>
        // Auto-scroll to top of page
        window.scrollTo(0, 0);
    </script>
    """, unsafe_allow_html=True)
    # Reset the flag
    st.session_state.auto_scroll = False

st.markdown("<h1 class='main-title'>Assortment Group (AG) Hierarchy Builder</h1>", unsafe_allow_html=True)
steps = ["Upload Data", "Analyze Attributes", "Select Attributes", "View Results"]

# Modern progress indicator
progress_val = (st.session_state.current_step - 1) / 3
st.progress(progress_val)

# Step indicator with better styling
st.markdown(f"<p class='step-title'>Step {st.session_state.current_step}/4: {steps[st.session_state.current_step-1]}</p>", unsafe_allow_html=True)

# Divider with some space
add_vertical_space(1)
st.markdown("---")
add_vertical_space(1)

# Step 1: Upload Data
if st.session_state.current_step == 1:
    st.header("Step 1: Upload Your Product Catalog")
    
    uploaded_file = st.file_uploader("Upload your product catalog (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Create dict of data types for all string columns
            dtype_dict = {
                'id': 'str',
                'name': 'str',
                'description': 'str',
                'product_id': 'str',
                'product_name': 'str',
                'product_description': 'str',
                'categories': 'str',
                'brands': 'str',
                'colors': 'str',
                'markets': 'str',
                'seasons': 'str',
                'styles': 'str',
                'size': 'str',
                'department_name': 'str',
                'department_id': 'str'
            }
            
            # Read data
            df = pd.read_csv(uploaded_file, dtype=dtype_dict)
            
            # Basic cleaning
            # Remove duplicates
            df_original_len = len(df)
            df.drop_duplicates(inplace=True)
            
            # Fill missing values in important columns
            if 'product_id' not in df.columns:
                st.error("The uploaded file must contain a 'product_id' column. Please check your data and try again.")
                st.session_state.df = None
            else:
                # Fill numeric columns with 0
                if 'price' in df.columns:
                    df['price'] = df['price'].fillna(0)
                if 'cost' in df.columns:
                    df['cost'] = df['cost'].fillna(0)
                
                # Fill string columns with 'empty_value'
                str_cols = df.select_dtypes(include=['object']).columns
                for col in str_cols:
                    df[col] = df[col].fillna('empty_value')
                
                # Save to session state
                st.session_state.df = df
                
                # Display data overview
                st.success(f"File uploaded successfully with {len(df)} products")
                st.write(f"Original file shape: {df_original_len} rows")
                st.write(f"After removing duplicates: {len(df)} rows")
                
                st.subheader("Data Preview")
                st.dataframe(df.head(5))
                
                st.subheader("Available Columns")
                col_list = list(df.columns)
                st.write(", ".join(col_list))
                
                
                # Next button
                st.button("Next: Analyze Attributes", on_click=go_to_next_step)
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    else:
        st.info("Please upload a CSV file containing your product catalog. The file should include columns such as product_id, brand, color, and other product attributes.")

# Step 2: Analyze Attributes
elif st.session_state.current_step == 2:
    st.header("Step 2: Analyze Product Attributes")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Check if attribute analysis has been done
        if not st.session_state.attribute_stats:
            with st.spinner("Analyzing attributes..."):
                st.session_state.attribute_stats = analyze_attributes(df)
        
        attribute_stats = st.session_state.attribute_stats
        
        # Display attribute statistics
        st.subheader("Attribute Statistics")
        
        # Create tabs for categorical and numeric attributes
        tab1, tab2 = st.tabs(["Categorical Attributes", "Numeric Attributes"])
        
        with tab1:
            # Display categorical attributes
            categorical_attrs = {k: v for k, v in attribute_stats.items() if v['type'] == 'categorical'}
            
            if categorical_attrs:
                for attr, stats in categorical_attrs.items():
                    with st.expander(f"{attr} - {stats['unique_values']} unique values"):
                        # Display top values
                        st.write("Top values:")
                        
                        # Create a dataframe for better display
                        top_values_df = pd.DataFrame(
                            list(stats['top_values'].items()),
                            columns=['Value', 'Count']
                        )
                        st.dataframe(top_values_df)
                        
                        # Missing values info
                        if stats['null_count'] > 0:
                            st.write(f"Missing values: {stats['null_count']}")
            else:
                st.info("No categorical attributes found in your data.")
        
        with tab2:
            # Display numeric attributes
            numeric_attrs = {k: v for k, v in attribute_stats.items() if v['type'] == 'numeric'}
            
            if numeric_attrs:
                for attr, stats in numeric_attrs.items():
                    with st.expander(f"{attr}"):
                        # Display basic statistics
                        st.write(f"Range: {stats['min']:.2f} to {stats['max']:.2f}")
                        st.write(f"Mean: {stats['mean']:.2f}")
                        st.write(f"Median: {stats['median']:.2f}")
                        st.write(f"Standard Deviation: {stats['std']:.2f}")
                        
                        # Plot histogram using Plotly for interactive tooltips
                        fig = plot_numeric_attribute(attr, stats)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Missing values info
                        if stats['null_count'] > 0:
                            st.write(f"Missing values: {stats['null_count']}")
            else:
                st.info("No numeric attributes found in your data.")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            st.button("Previous: Upload Data", on_click=go_to_prev_step)
        with col2:
            st.button("Next: Select Attributes", on_click=go_to_next_step)
    else:
        st.error("No data available. Please go back and upload a CSV file.")
        st.button("Back to Upload", on_click=lambda: go_to_step(1))

# Step 3: Select Attributes
elif st.session_state.current_step == 3:
    st.header("Step 3: Select Attributes for AG Creation")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Exclude columns that shouldn't be used for AG creation
        excluded_columns = ['ag', 'id', 'name', 'description', 'product_description']
        available_columns = [col for col in df.columns if col not in excluded_columns]
        
        st.write("Select which attributes to use for creating Assortment Groups:")
        st.write("The order of attributes matters - they will form a hierarchy from top to bottom.")
        
        # Determine default values for attribute selection
        default_attributes = []
        if st.session_state.selected_attributes:
            # Use previously selected attributes if they exist
            default_attributes = st.session_state.selected_attributes
        elif 'brands' in available_columns and 'colors' in available_columns:
            # Default to common retail attributes
            default_attributes = ['brands', 'colors']
        elif len(available_columns) >= 2:
            # Default to first two columns
            default_attributes = available_columns[:2]
        else:
            # Use whatever is available
            default_attributes = available_columns
        
        # Function to update session state on selection change
        def update_selected_attributes():
            st.session_state.selected_attributes = st.session_state.attribute_selection
        
        # Select attributes for AG creation with a persistent key
        selected_attributes = st.multiselect(
            "Select Attributes",
            available_columns,
            default=default_attributes,
            key="attribute_selection",
            on_change=update_selected_attributes
        )
        
        # Display the selection
        if selected_attributes:
            st.success(f"You've selected {len(selected_attributes)} attributes: {', '.join(selected_attributes)}")
            
            # Show how many AGs might be created (theoretical maximum)
            max_ags = 1
            for attr in selected_attributes:
                if attr in st.session_state.attribute_stats:
                    stats = st.session_state.attribute_stats[attr]
                    if stats['type'] == 'categorical':
                        max_ags *= stats['unique_values']
            
            st.info(f"These attributes could theoretically create up to {max_ags:,} unique AGs (though the actual number will likely be much lower).")
        else:
            st.warning("Please select at least one attribute to continue.")
        
        # Use the default parameters (no UI controls to modify them)
        # We'll use the session state values directly in the analysis
        
        # Navigation and analysis buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            st.button("Previous: Analyze Attributes", on_click=go_to_prev_step)
        with col2:
            if selected_attributes:
                if st.button("View Results", type="primary"):
                    with st.spinner("Running analysis..."):
                        # Run the analysis
                        df = st.session_state.df
                        # Use the values from session state
                        results = run_analysis(
                            df, 
                            selected_attributes,
                            st.session_state.min_products_in_group,
                            st.session_state.max_products_in_group,
                            st.session_state.hist_xlimit,
                            st.session_state.hist_xticks_jumps,
                            st.session_state.enable_hierarchy
                        )
                        
                        # Store results in session state
                        st.session_state.analysis_results = results
                        
                        # Move to next step
                        go_to_next_step()
                        st.rerun()
            else:
                st.button("View Results", disabled=True)
    else:
        st.error("No data available. Please go back and upload a CSV file.")
        st.button("Back to Upload", on_click=lambda: go_to_step(1))

# Step 4: View Results
elif st.session_state.current_step == 4:
    st.header("Step 4: View Results")
    
    if st.session_state.analysis_results and len(st.session_state.analysis_results) == 4:
        df_tmp, df_stats, dist_, price_stats = st.session_state.analysis_results
        selected_attributes = st.session_state.selected_attributes
        comb_str = "_".join(selected_attributes)
        
        # Main statistics in a modern card
        st.subheader("AG Statistics")
        
        # Display statistics in a styled card
        with st.container():
            # Key metrics in big numbers with a modern look
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="stat-card">
                    <div class="stat-value">{:,}</div>
                    <div class="stat-label">Total Products</div>
                </div>
                """.format(df_stats['# Products'].values[0]), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="stat-card">
                    <div class="stat-value">{:,}</div>
                    <div class="stat-label">Total AGs</div>
                </div>
                """.format(df_stats['# AGs'].values[0]), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="stat-card">
                    <div class="stat-value">{:.1f}</div>
                    <div class="stat-label">Avg Products per AG</div>
                </div>
                """.format(df_stats['Average Products in AG'].values[0]), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="stat-card">
                    <div class="stat-value">{:.1f}</div>
                    <div class="stat-label">Median Products per AG</div>
                </div>
                """.format(df_stats['Median Products in AG'].values[0]), unsafe_allow_html=True)
            
            add_vertical_space(1)
            
            # Full stats in collapsible section
            with st.expander("View detailed statistics"):
                st.dataframe(df_stats, use_container_width=True)

        # Modern Chart.js Visualizations in tabs
        tab1, tab2 = st.tabs(["Products per AG Distribution", "Price Variation"])
        
        with tab1:
            # Distribution of products per AG using Plotly
            st.write("This histogram shows how many AGs have a certain number of products.")
            
            # Create the Plotly chart
            fig = plot_dist(
                dist_, 
                df_stats, 
                comb_str, 
                st.session_state.hist_xlimit, 
                st.session_state.hist_xticks_jumps
            )
            
            # Display in a styled container
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            # Price variation by AG
            if price_stats is not None:
                st.write("This chart shows price variation within each AG (coefficient of variation %).")
                st.write("Lower values indicate more consistent pricing within the AG.")
                
                # Create the matplotlib chart
                fig = plot_price_variance(price_stats, comb_str)
                
                # Display in a styled container
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show price stats table
                with st.expander("View detailed price statistics"):
                    st.dataframe(price_stats.sort_values('Count', ascending=False), use_container_width=True)
            else:
                st.info("Price data not available. Include a 'price' column in your data to see price variation analysis.")
        
        # Hierarchy Analysis with modern visualizations (enabled by default)
        enable_hierarchy = True  # Default value since we removed the UI control
        if enable_hierarchy and len(selected_attributes) > 1:
            st.subheader("Hierarchy Analysis")
            st.write("This analysis shows how products are distributed across different levels of the hierarchy.")
            
            hierarchy_tabs = st.tabs([f"Level {i+1}: {col}" for i, col in enumerate(selected_attributes)])
            
            for i, tab in enumerate(hierarchy_tabs):
                with tab:
                    level_name = f'ag_level_{i+1}'
                    if level_name in df_tmp.columns:
                        # Calculate distribution for this level
                        level_dist = df_tmp.groupby(level_name)['product_id'].size()
                        level_stats = pd.DataFrame({
                            'Level': [i+1],
                            'Attributes': [', '.join(selected_attributes[:i+1])],
                            '# Groups': [level_dist.shape[0]],
                            'Min Products': [min(level_dist)],
                            'Max Products': [max(level_dist)],
                            'Avg Products': [int(level_dist.mean())]
                        })
                        
                        # Display stats for this level with custom styling
                        st.markdown(f"""
                        <div class="stat-card">
                            <h3 style="font-size: 18px; color: var(--space-navy); margin-bottom: 10px;">Level {i+1} Statistics</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        st.dataframe(level_stats, use_container_width=True)
                        
                        # Create matplotlib visualization for this level
                        fig = plot_hierarchy_level(
                            df_tmp,
                            i+1, 
                            selected_attributes[i],
                            comb_str
                        )
                        
                        # Display in a styled container
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.plotly_chart(fig)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create output directory
        output_dir = 'output'
        if not path.exists(output_dir):
            makedirs(output_dir)
        
        # Save results and offer download
        csv_path = save_comb(df=df_tmp, comb_str=comb_str)
        
        # Simple download button without the header
        with open(csv_path, "rb") as file:
            st.download_button(
                label=f"📥 Download AG Results as CSV",
                data=file,
                file_name=f"{comb_str}_ag_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Navigation buttons with better styling
        st.markdown("---")
        st.markdown("### Next Steps")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("⬅️ Previous: Select Attributes", on_click=go_to_prev_step, use_container_width=True)
        with col2:
            st.button("🔄 Start Over", on_click=lambda: go_to_step(1), use_container_width=True)
        with col3:
            st.button("✏️ Try Different Attributes", on_click=lambda: go_to_step(3), use_container_width=True)
    else:
        # Error state with better styling
        st.markdown("""
        <div class="stat-card" style="border-left: 4px solid var(--alert-red);">
            <h3 style="font-size: 18px; color: var(--alert-red); margin-bottom: 10px;">Error</h3>
            <p style="color: var(--space-navy); margin-bottom: 15px;">No analysis results available</p>
        </div>
        """, unsafe_allow_html=True)
        st.error("No analysis results available. Please select attributes and run the analysis first.")
        st.button("Go Back to Select Attributes", on_click=lambda: go_to_step(3), use_container_width=True)

# Footer
st.markdown("---")
st.caption("AG Hierarchy Builder - A tool for retail assortment planning and product grouping")