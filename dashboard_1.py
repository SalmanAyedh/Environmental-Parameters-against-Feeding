import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Climate Effects on Pig Feeding Dashboard",
    page_icon="🐷",
    layout="wide"
)


@st.cache_data
def load_data():
    """Load the combined weekly intake and climate data."""
    try:
        # Load weekly intake data
        intake_file = Path("output_combined_filtered/combined_weekly_intake_filtered.csv")
        if not intake_file.exists():
            st.error(f"Weekly intake file not found: {intake_file}")
            return None, None

        intake_data = pd.read_csv(intake_file)

        # Load climate data
        climate_file = Path("output_combined_filtered/combined_weekly_climate_averages.csv")
        if not climate_file.exists():
            st.error(f"Climate file not found: {climate_file}")
            return intake_data, None

        climate_data = pd.read_csv(climate_file)

        return intake_data, climate_data

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


def merge_climate_intake_data(intake_data, climate_data):
    """Merge climate and intake data on week_year and experiment."""
    if intake_data is None or climate_data is None:
        return None

    # Pivot climate data to have one row per week_year/experiment with columns for each climate type
    climate_pivot = climate_data.pivot_table(
        index=['week_year', 'experiment'],
        columns='climate_type',
        values='avg_value',
        aggfunc='first'
    ).reset_index()

    # Flatten column names
    climate_pivot.columns.name = None

    # Merge with intake data
    merged_data = intake_data.merge(
        climate_pivot,
        on=['week_year', 'experiment'],
        how='inner'
    )

    return merged_data


def create_feeding_climate_plot(data, climate_type, feeding_metric='total_weekly_intake'):
    """Create scatter plot for climate parameter vs feeding metric."""
    if climate_type not in data.columns or feeding_metric not in data.columns:
        return None

    # Remove rows with missing data
    plot_data = data.dropna(subset=[climate_type, feeding_metric])

    if len(plot_data) == 0:
        return None

    # Create scatter plot
    fig = px.scatter(
        plot_data,
        x=climate_type,
        y=feeding_metric,
        color='experiment',
        size='feeding_sessions',
        title=f'{climate_type} vs {feeding_metric.replace("_", " ").title()}',
        labels={
            climate_type: f'{climate_type}',
            feeding_metric: feeding_metric.replace('_', ' ').title(),
            'experiment': 'Experiment'
        },
        hover_data=['pig_id', 'week_year', 'feeding_sessions']
    )

    # Calculate correlation and add trend line
    try:
        correlation = plot_data[climate_type].corr(plot_data[feeding_metric])

        # Add trend line
        z = np.polyfit(plot_data[climate_type], plot_data[feeding_metric], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(plot_data[climate_type].min(), plot_data[climate_type].max(), 100)

        fig.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            name=f'Trend (r={correlation:.3f})',
            line=dict(dash='dash', color='red')
        ))

        # Update title with correlation
        fig.update_layout(
            title=f'{climate_type} vs {feeding_metric.replace("_", " ").title()} (r={correlation:.3f})',
            showlegend=True
        )

    except Exception as e:
        st.warning(f"Could not add trend line for {climate_type}: {e}")

    return fig


def create_feeding_behavior_heatmap(data, climate_types):
    """Create correlation heatmap between climate types and feeding behaviors."""
    feeding_metrics = ['total_weekly_intake', 'feeding_sessions', 'avg_intake_per_session']
    available_metrics = [col for col in feeding_metrics if col in data.columns]

    if not available_metrics:
        return None

    # Calculate correlations
    corr_data = data[climate_types + available_metrics].corr()

    # Extract correlations between climate and feeding metrics
    feeding_climate_corr = corr_data.loc[available_metrics, climate_types]

    fig = px.imshow(
        feeding_climate_corr.values,
        x=feeding_climate_corr.columns,
        y=feeding_climate_corr.index,
        color_continuous_scale='RdBu_r',
        aspect='auto',
        title='Climate Effects on Feeding Behavior (Correlations)',
        zmin=-1, zmax=1
    )

    # Add correlation values as text
    for i, row_name in enumerate(feeding_climate_corr.index):
        for j, col_name in enumerate(feeding_climate_corr.columns):
            val = feeding_climate_corr.loc[row_name, col_name]
            if not np.isnan(val):
                fig.add_annotation(
                    x=j, y=i,
                    text=f'{val:.2f}',
                    showarrow=False,
                    font=dict(color='white' if abs(val) > 0.5 else 'black')
                )

    return fig


def create_climate_time_series(data, climate_types):
    """Create time series showing climate parameters over time with feeding overlay."""
    if not climate_types:
        return None

    # Aggregate by week for cleaner visualization
    weekly_agg = data.groupby(['week_year', 'experiment']).agg({
        **{climate_type: 'mean' for climate_type in climate_types},
        'total_weekly_intake': 'mean',
        'feeding_sessions': 'mean'
    }).reset_index()

    # Sort by week_year
    weekly_agg = weekly_agg.sort_values('week_year')

    # Create subplots
    n_climate = len(climate_types)
    fig = make_subplots(
        rows=n_climate + 1, cols=1,
        subplot_titles=climate_types + ['Average Weekly Intake'],
        shared_xaxes=True,
        vertical_spacing=0.05
    )

    # Add climate parameter traces
    for i, climate_type in enumerate(climate_types):
        for exp in weekly_agg['experiment'].unique():
            exp_data = weekly_agg[weekly_agg['experiment'] == exp]
            fig.add_trace(
                go.Scatter(
                    x=exp_data['week_year'],
                    y=exp_data[climate_type],
                    mode='lines+markers',
                    name=f'{exp} - {climate_type}',
                    legendgroup=exp,
                    showlegend=(i == 0)
                ),
                row=i + 1, col=1
            )

    # Add feeding intake trace
    for exp in weekly_agg['experiment'].unique():
        exp_data = weekly_agg[weekly_agg['experiment'] == exp]
        fig.add_trace(
            go.Scatter(
                x=exp_data['week_year'],
                y=exp_data['total_weekly_intake'],
                mode='lines+markers',
                name=f'{exp} - Intake',
                legendgroup=exp,
                showlegend=False
            ),
            row=n_climate + 1, col=1
        )

    fig.update_layout(
        height=300 * (n_climate + 1),
        title='Climate Parameters and Feeding Intake Over Time',
        showlegend=True
    )

    # Update x-axis labels
    fig.update_xaxes(tickangle=45, row=n_climate + 1, col=1)

    return fig


def create_feeding_distribution_by_climate(data, climate_type):
    """Create feeding distribution plots categorized by climate ranges."""
    if climate_type not in data.columns:
        return None

    # Create climate categories (low, medium, high)
    climate_values = data[climate_type].dropna()
    if len(climate_values) < 3:
        return None

    q33, q66 = climate_values.quantile([0.33, 0.67])

    data_copy = data.copy()
    # Create categorical data - pd.cut doesn't have 'observed' parameter
    data_copy['climate_category'] = pd.cut(
        data_copy[climate_type],
        bins=[-np.inf, q33, q66, np.inf],
        labels=['Low', 'Medium', 'High']
    )

    # Create violin plots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Weekly Intake', 'Feeding Sessions', 'Avg per Session'],
        shared_yaxes=False
    )

    metrics = ['total_weekly_intake', 'feeding_sessions', 'avg_intake_per_session']
    titles = ['Weekly Intake (kg)', 'Feeding Sessions', 'Avg per Session (kg)']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        if metric in data_copy.columns:
            fig.add_trace(
                go.Violin(
                    x=data_copy['climate_category'],
                    y=data_copy[metric],
                    name=title,
                    box_visible=True,
                    meanline_visible=True
                ),
                row=1, col=i + 1
            )

    fig.update_layout(
        title=f'Feeding Behavior Distribution by {climate_type} Level',
        showlegend=False,
        height=400
    )

    return fig


def create_climate_threshold_analysis(data, climate_type):
    """Analyze feeding behavior changes at different climate thresholds."""
    if climate_type not in data.columns:
        return None, None

    climate_data = data[[climate_type, 'total_weekly_intake', 'feeding_sessions', 'avg_intake_per_session']].dropna()

    if len(climate_data) < 20:
        return None, None

    # Create bins and calculate means - FIX: Add observed=True to groupby only
    n_bins = 10
    climate_data = climate_data.copy()  # Create a copy to avoid SettingWithCopyWarning
    climate_data['climate_bin'] = pd.cut(climate_data[climate_type], bins=n_bins)

    bin_stats = climate_data.groupby('climate_bin', observed=True).agg({
        climate_type: 'mean',
        'total_weekly_intake': ['mean', 'std', 'count'],
        'feeding_sessions': ['mean', 'std'],
        'avg_intake_per_session': ['mean', 'std']
    }).round(3)

    # Flatten column names
    bin_stats.columns = ['_'.join(col).strip() for col in bin_stats.columns]
    bin_stats = bin_stats.reset_index()

    # Create threshold analysis plot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'{climate_type} vs Total Weekly Intake',
            f'{climate_type} vs Feeding Sessions',
            f'{climate_type} vs Avg Intake per Session',
            'Change Rate Analysis'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": True}]]
    )

    # Plot 1: Total intake with error bars
    fig.add_trace(
        go.Scatter(
            x=bin_stats[f'{climate_type}_mean'],
            y=bin_stats['total_weekly_intake_mean'],
            error_y=dict(array=bin_stats['total_weekly_intake_std']),
            mode='lines+markers',
            name='Weekly Intake',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    # Plot 2: Feeding sessions
    fig.add_trace(
        go.Scatter(
            x=bin_stats[f'{climate_type}_mean'],
            y=bin_stats['feeding_sessions_mean'],
            error_y=dict(array=bin_stats['feeding_sessions_std']),
            mode='lines+markers',
            name='Feeding Sessions',
            line=dict(color='green')
        ),
        row=1, col=2
    )

    # Plot 3: Avg intake per session
    fig.add_trace(
        go.Scatter(
            x=bin_stats[f'{climate_type}_mean'],
            y=bin_stats['avg_intake_per_session_mean'],
            error_y=dict(array=bin_stats['avg_intake_per_session_std']),
            mode='lines+markers',
            name='Avg per Session',
            line=dict(color='red')
        ),
        row=2, col=1
    )

    # Plot 4: Rate of change analysis
    climate_vals = bin_stats[f'{climate_type}_mean'].values
    intake_vals = bin_stats['total_weekly_intake_mean'].values

    # Calculate rate of change (derivative approximation)
    if len(climate_vals) > 1:
        rate_of_change = np.gradient(intake_vals, climate_vals)

        fig.add_trace(
            go.Scatter(
                x=climate_vals,
                y=rate_of_change,
                mode='lines+markers',
                name='Rate of Change',
                line=dict(color='purple')
            ),
            row=2, col=2
        )

        # Add horizontal line at zero
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)

    fig.update_layout(
        title=f'Climate Threshold Analysis: {climate_type}',
        height=600,
        showlegend=False
    )

    # Find critical thresholds (where rate of change is highest)
    if len(climate_vals) > 1:
        max_change_idx = np.argmax(np.abs(rate_of_change))
        critical_threshold = climate_vals[max_change_idx]

        threshold_info = {
            'critical_threshold': critical_threshold,
            'max_change_rate': rate_of_change[max_change_idx],
            'bin_stats': bin_stats
        }

        return fig, threshold_info

    return fig, None


def create_multi_climate_3d_plot(data, climate_types):
    """Create 3D plot showing interaction between two climate parameters and feeding."""
    if len(climate_types) < 2:
        return None

    # Select two most correlated climate parameters
    correlations = {}
    for climate in climate_types:
        if climate in data.columns and 'total_weekly_intake' in data.columns:
            corr = data[climate].corr(data['total_weekly_intake'])
            if not np.isnan(corr):
                correlations[climate] = abs(corr)

    if len(correlations) < 2:
        return None

    # Get top 2 climate parameters
    sorted_climates = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    climate1, climate2 = sorted_climates[0][0], sorted_climates[1][0]

    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=data[climate1],
        y=data[climate2],
        z=data['total_weekly_intake'],
        mode='markers',
        marker=dict(
            size=5,
            color=data['total_weekly_intake'],
            colorscale='Viridis',
            opacity=0.7,
            colorbar=dict(title='Weekly Intake (kg)')
        ),
        text=data['experiment'],
        hovertemplate=f'<b>{climate1}</b>: %{{x}}<br>' +
                      f'<b>{climate2}</b>: %{{y}}<br>' +
                      '<b>Weekly Intake</b>: %{z}<br>' +
                      '<b>Experiment</b>: %{text}<extra></extra>'
    )])

    fig.update_layout(
        title=f'3D Climate Interaction: {climate1} vs {climate2} vs Feeding',
        scene=dict(
            xaxis_title=climate1,
            yaxis_title=climate2,
            zaxis_title='Total Weekly Intake (kg)'
        ),
        height=600
    )

    return fig


def create_optimal_climate_zones(data, climate_types):
    """Identify optimal climate zones for feeding performance."""
    if len(climate_types) < 2:
        return None, None

    # Check if required columns exist
    required_cols = ['total_weekly_intake', 'feeding_sessions', 'avg_intake_per_session']
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        st.warning(f"Missing columns for optimal zone analysis: {missing_cols}")
        return None, None

    # Calculate feeding efficiency score with error handling
    data_copy = data.copy()

    try:
        # Handle potential division by zero or missing data
        feeding_sessions_safe = data_copy['feeding_sessions'].replace(0, np.nan)

        data_copy['feeding_efficiency'] = (
                data_copy['total_weekly_intake'].fillna(0) * 0.5 +
                data_copy['avg_intake_per_session'].fillna(0) * 0.3 +
                (1 / feeding_sessions_safe).fillna(0) * 0.2  # Prefer fewer, larger meals
        )

        # Remove rows with NaN efficiency scores
        data_copy = data_copy.dropna(subset=['feeding_efficiency'])

        if len(data_copy) < 10:
            st.warning("Insufficient data for optimal zone analysis")
            return None, None

    except Exception as e:
        st.error(f"Error calculating feeding efficiency: {e}")
        return None, None

    # Find top and bottom performers
    top_10_pct = data_copy['feeding_efficiency'].quantile(0.9)
    bottom_10_pct = data_copy['feeding_efficiency'].quantile(0.1)

    optimal_data = data_copy[data_copy['feeding_efficiency'] >= top_10_pct]
    poor_data = data_copy[data_copy['feeding_efficiency'] <= bottom_10_pct]

    if len(optimal_data) == 0 or len(poor_data) == 0:
        st.warning("Not enough data to identify optimal and poor performance zones")
        return None, None

    # Create comparison plot for top 2 climate parameters
    correlations = {}
    for climate in climate_types:
        if climate in data_copy.columns:
            try:
                corr = abs(data_copy[climate].corr(data_copy['feeding_efficiency']))
                if not np.isnan(corr):
                    correlations[climate] = corr
            except Exception:
                continue

    if len(correlations) < 2:
        st.warning("Insufficient climate correlations for optimal zone analysis")
        return None, None

    sorted_climates = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    climate1, climate2 = sorted_climates[0][0], sorted_climates[1][0]

    # Ensure we have data for both climate parameters
    optimal_data_clean = optimal_data.dropna(subset=[climate1, climate2])
    poor_data_clean = poor_data.dropna(subset=[climate1, climate2])

    if len(optimal_data_clean) == 0 or len(poor_data_clean) == 0:
        st.warning(f"No clean data available for {climate1} and {climate2}")
        return None, None

    fig = go.Figure()

    # Add optimal zones
    fig.add_trace(go.Scatter(
        x=optimal_data_clean[climate1],
        y=optimal_data_clean[climate2],
        mode='markers',
        name='Optimal Feeding (Top 10%)',
        marker=dict(color='green', size=8, symbol='circle')
    ))

    # Add poor performance zones
    fig.add_trace(go.Scatter(
        x=poor_data_clean[climate1],
        y=poor_data_clean[climate2],
        mode='markers',
        name='Poor Feeding (Bottom 10%)',
        marker=dict(color='red', size=8, symbol='x')
    ))

    # Add convex hull for optimal zone
    try:
        from scipy.spatial import ConvexHull
        if len(optimal_data_clean) >= 3:
            optimal_points = optimal_data_clean[[climate1, climate2]].values
            hull = ConvexHull(optimal_points)

            # Create hull polygon
            hull_x = optimal_points[hull.vertices, 0]
            hull_y = optimal_points[hull.vertices, 1]
            hull_x = np.append(hull_x, hull_x[0])  # Close the polygon
            hull_y = np.append(hull_y, hull_y[0])

            fig.add_trace(go.Scatter(
                x=hull_x, y=hull_y,
                mode='lines',
                name='Optimal Zone Boundary',
                line=dict(color='green', dash='dash'),
                fill='toself',
                fillcolor='rgba(0,255,0,0.1)'
            ))
    except ImportError:
        st.info("Install scipy for optimal zone boundary visualization")
    except Exception as e:
        st.info(f"Could not create optimal zone boundary: {e}")

    fig.update_layout(
        title=f'Optimal Climate Zones: {climate1} vs {climate2}',
        xaxis_title=climate1,
        yaxis_title=climate2,
        height=500
    )

    # Calculate zone statistics with error handling
    try:
        zone_stats = {
            'optimal_ranges': {
                climate1: f"{optimal_data_clean[climate1].min():.2f} - {optimal_data_clean[climate1].max():.2f}",
                climate2: f"{optimal_data_clean[climate2].min():.2f} - {optimal_data_clean[climate2].max():.2f}"
            },
            'optimal_avg_intake': optimal_data_clean['total_weekly_intake'].mean(),
            'poor_avg_intake': poor_data_clean['total_weekly_intake'].mean(),
            'improvement_potential': ((optimal_data_clean['total_weekly_intake'].mean() -
                                       poor_data_clean['total_weekly_intake'].mean()) /
                                      poor_data_clean['total_weekly_intake'].mean() * 100)
        }

        return fig, zone_stats

    except Exception as e:
        st.error(f"Error calculating zone statistics: {e}")
        return fig, None


def main():
    st.title("🐷 Climate Effects on Pig Feeding Behavior Dashboard")
    st.markdown("Analyze how climate parameters affect pig feeding patterns, intake, and behavior")

    # Load data
    with st.spinner("Loading data..."):
        intake_data, climate_data = load_data()

    if intake_data is None:
        st.stop()

    # Sidebar controls
    st.sidebar.header("Analysis Controls")

    # Show data info
    st.sidebar.subheader("Data Overview")
    st.sidebar.write(f"Pigs: {intake_data['pig_id'].nunique()}")
    st.sidebar.write(f"Experiments: {', '.join(intake_data['experiment'].unique())}")
    st.sidebar.write(f"Weeks: {intake_data['week_year'].nunique()}")

    # Filter by experiment
    experiments = st.sidebar.multiselect(
        "Select Experiments",
        intake_data['experiment'].unique(),
        default=intake_data['experiment'].unique()
    )

    if not experiments:
        st.warning("Please select at least one experiment")
        st.stop()

    # Filter data
    filtered_intake = intake_data[intake_data['experiment'].isin(experiments)]

    if climate_data is not None:
        filtered_climate = climate_data[climate_data['experiment'].isin(experiments)]

        # Merge data
        merged_data = merge_climate_intake_data(filtered_intake, filtered_climate)

        if merged_data is not None and len(merged_data) > 0:
            # Get available climate types
            climate_types = [col for col in merged_data.columns
                             if col not in ['pig_id', 'week', 'year', 'week_year', 'total_weekly_intake',
                                            'feeding_sessions', 'avg_intake_per_session', 'first_date',
                                            'last_date', 'feeding_days', 'stations_used', 'first_feeding_time',
                                            'last_feeding_time', 'total_duration_seconds', 'date_range',
                                            'total_duration_minutes', 'weekly_weight_gain', 'experiment']]

            if climate_types:
                st.sidebar.write(f"Climate types: {len(climate_types)}")

                # Select feeding metric to analyze
                feeding_metrics = ['total_weekly_intake', 'feeding_sessions', 'avg_intake_per_session']
                available_feeding_metrics = [m for m in feeding_metrics if m in merged_data.columns]

                selected_metric = st.sidebar.selectbox(
                    "Primary Feeding Metric",
                    available_feeding_metrics,
                    index=0 if available_feeding_metrics else None
                )

                # Main analysis
                st.header("Climate Effects on Feeding Behavior")

                # Overall correlation heatmap
                st.subheader("Climate-Feeding Correlations Overview")
                heatmap_fig = create_feeding_behavior_heatmap(merged_data, climate_types)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)

                # Time series analysis
                st.subheader("Climate and Feeding Patterns Over Time")
                time_series_fig = create_climate_time_series(merged_data, climate_types)
                if time_series_fig:
                    st.plotly_chart(time_series_fig, use_container_width=True)

                # Individual climate effects
                st.subheader(f"Individual Climate Effects on {selected_metric.replace('_', ' ').title()}")

                # Create plots in a grid
                cols_per_row = 2
                rows_needed = (len(climate_types) + cols_per_row - 1) // cols_per_row

                for row in range(rows_needed):
                    cols = st.columns(cols_per_row)

                    for col_idx in range(cols_per_row):
                        climate_idx = row * cols_per_row + col_idx
                        if climate_idx < len(climate_types):
                            climate_type = climate_types[climate_idx]

                            with cols[col_idx]:
                                fig = create_feeding_climate_plot(merged_data, climate_type, selected_metric)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)

                # Advanced Analysis Tabs
                st.subheader("Advanced Climate-Feeding Analysis")

                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Distribution Analysis",
                    "🎯 Threshold Detection",
                    "🌍 3D Interactions",
                    "⭐ Optimal Zones"
                ])

                with tab1:
                    st.markdown("**Feeding Behavior by Climate Conditions**")
                    selected_climate = st.selectbox(
                        "Select climate parameter to analyze distributions:",
                        climate_types,
                        key="dist_climate"
                    )

                    if selected_climate:
                        dist_fig = create_feeding_distribution_by_climate(merged_data, selected_climate)
                        if dist_fig:
                            st.plotly_chart(dist_fig, use_container_width=True)

                            # Add interpretation
                            climate_values = merged_data[selected_climate].dropna()
                            q33, q66 = climate_values.quantile([0.33, 0.67])

                            st.info(f"""
                            **Climate Categories for {selected_climate}:**
                            - Low: < {q33:.2f}
                            - Medium: {q33:.2f} - {q66:.2f}
                            - High: > {q66:.2f}

                            Compare the feeding behavior distributions to see how {selected_climate} affects pig feeding patterns.
                            """)

                with tab2:
                    st.markdown("**Critical Climate Thresholds**")
                    threshold_climate = st.selectbox(
                        "Select climate parameter for threshold analysis:",
                        climate_types,
                        key="threshold_climate"
                    )

                    if threshold_climate:
                        threshold_fig, threshold_info = create_climate_threshold_analysis(merged_data,
                                                                                          threshold_climate)
                        if threshold_fig:
                            st.plotly_chart(threshold_fig, use_container_width=True)

                            if threshold_info:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "Critical Threshold",
                                        f"{threshold_info['critical_threshold']:.2f}",
                                        help="Climate value where feeding behavior changes most dramatically"
                                    )

                                st.markdown("**Interpretation:**")
                                if threshold_info['max_change_rate'] > 0:
                                    st.success(
                                        f"📈 Feeding increases most rapidly around {threshold_info['critical_threshold']:.2f}")
                                else:
                                    st.warning(
                                        f"📉 Feeding decreases most rapidly around {threshold_info['critical_threshold']:.2f}")

                with tab3:
                    st.markdown("**Multi-Climate Interactions**")
                    if len(climate_types) >= 2:
                        interaction_fig = create_multi_climate_3d_plot(merged_data, climate_types)
                        if interaction_fig:
                            st.plotly_chart(interaction_fig, use_container_width=True)
                            st.info(
                                "🔄 Rotate and zoom the 3D plot to explore how combinations of climate factors affect feeding!")
                        else:
                            st.warning("Unable to create 3D interaction plot")
                    else:
                        st.warning("Need at least 2 climate parameters for interaction analysis")

                with tab4:
                    st.markdown("**Optimal Climate Conditions**")
                    if len(climate_types) >= 2:
                        try:
                            optimal_fig, zone_stats = create_optimal_climate_zones(merged_data, climate_types)
                            if optimal_fig and zone_stats:
                                st.plotly_chart(optimal_fig, use_container_width=True)

                                # Display optimal ranges
                                st.markdown("**🎯 Optimal Climate Ranges:**")
                                col1, col2, col3 = st.columns(3)

                                climate_names = list(zone_stats['optimal_ranges'].keys())
                                with col1:
                                    st.metric(
                                        climate_names[0],
                                        zone_stats['optimal_ranges'][climate_names[0]]
                                    )
                                with col2:
                                    st.metric(
                                        climate_names[1],
                                        zone_stats['optimal_ranges'][climate_names[1]]
                                    )
                                with col3:
                                    st.metric(
                                        "Performance Gap",
                                        f"{zone_stats['improvement_potential']:.1f}%",
                                        help="Potential improvement from optimal vs poor conditions"
                                    )

                                st.success(f"""
                                **Key Insights:**
                                - Optimal feeding performance: {zone_stats['optimal_avg_intake']:.2f} kg/week
                                - Poor performance: {zone_stats['poor_avg_intake']:.2f} kg/week
                                - **Potential improvement: {zone_stats['improvement_potential']:.1f}%** by maintaining optimal climate conditions
                                """)
                            elif optimal_fig:
                                st.plotly_chart(optimal_fig, use_container_width=True)
                                st.info("Optimal zone visualization created, but statistics unavailable")
                            else:
                                st.warning(
                                    "Unable to create optimal zones analysis. This may be due to insufficient data or missing columns.")
                        except Exception as e:
                            st.error(f"Error in optimal zones analysis: {e}")
                            st.info(
                                "This analysis requires columns: total_weekly_intake, feeding_sessions, avg_intake_per_session")
                    else:
                        st.warning("Need at least 2 climate parameters for optimal zone analysis")

                # Summary statistics
                with st.expander("📊 Summary Statistics"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Feeding Metrics")
                        for metric in available_feeding_metrics:
                            avg_val = merged_data[metric].mean()
                            std_val = merged_data[metric].std()
                            st.metric(
                                metric.replace('_', ' ').title(),
                                f"{avg_val:.3f}",
                                f"±{std_val:.3f}"
                            )

                    with col2:
                        st.subheader("Climate Ranges")
                        for climate_type in climate_types:
                            min_val = merged_data[climate_type].min()
                            max_val = merged_data[climate_type].max()
                            mean_val = merged_data[climate_type].mean()
                            st.metric(
                                climate_type,
                                f"{mean_val:.2f}",
                                f"{min_val:.2f} - {max_val:.2f}"
                            )

                # Data download
                with st.expander("📥 Download Data"):
                    csv_data = merged_data.to_csv(index=False)
                    st.download_button(
                        label="Download Climate-Feeding Analysis Data",
                        data=csv_data,
                        file_name="climate_feeding_analysis.csv",
                        mime="text/csv"
                    )
                    st.dataframe(merged_data.head(100))

            else:
                st.warning("No climate types found in the data")
        else:
            st.warning("No data available after merging climate and intake data")

    else:
        st.warning("Climate data not available. Please ensure climate data files are present.")


if __name__ == "__main__":
    main()